"""MultiStrategySwingTradingEnv — v2 selector env.

Steps once per ``StrategyPack`` (one (symbol, date) where at least one
strategy fired). The agent sees the full slate of strategy proposals
and chooses among ``Discrete(N+1)`` actions: 0 = skip, 1..N = take
strategy k's recommendation. If the agent selects strategy k that
didn't fire on this pack, the env applies a small negative penalty
(``illegal_action_penalty``) and skips — encouraging the agent to
attend to the per-slot ``fired`` feature.

Reward shaping:
    take strategy k:    RewardModel.reward_for_take(outcome_k)
    skip:               RewardModel.reward_for_skip(best_counterfactual)
    illegal action:     -illegal_action_penalty   (small constant)

Where ``best_counterfactual`` is the highest-return counterfactual
among the strategies that fired on this pack — i.e., the "what would
the best of you have made" baseline. This makes skip costly when at
least one strategy had a winner.
"""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_swing.domain import (
    CandidateTrade,
    FeatureFrame,
    MarketBar,
    PortfolioState,
)
from rl_swing.rl.env.cost_model import EquityExecutionModel
from rl_swing.rl.env.execution_simulator import ExecutionSimulator
from rl_swing.rl.env.multi_strategy_observation import (
    MultiStrategyObservationBuilder,
)
from rl_swing.rl.env.reward_model import RewardModel
from rl_swing.strategies.multi_strategy_packer import StrategyPack

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
@dataclass
class _PackBars:
    """Bars indexed by symbol for fast simulation. Mirrors v1's EnvBars."""
    by_symbol: dict[str, list[MarketBar]] = field(default_factory=dict)

    def add(self, bars: Sequence[MarketBar]) -> None:
        d = defaultdict(list)
        for b in bars:
            d[b.symbol].append(b)
        for s, lst in d.items():
            lst.sort(key=lambda x: x.timestamp)
            self.by_symbol[s] = lst

    def find_index(self, symbol: str, timestamp) -> int:
        bars = self.by_symbol.get(symbol)
        if not bars:
            return -1
        lo, hi = 0, len(bars) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if bars[mid].timestamp < timestamp:
                lo = mid + 1
            elif bars[mid].timestamp > timestamp:
                hi = mid - 1
            else:
                return mid
        return -1


# ---------------------------------------------------------------------
class _PackEpisodeSampler:
    """Random / chronological window sampler over StrategyPacks. Mirrors
    the candidate samplers from v1 but operates on packs.
    """

    def __init__(
        self,
        packs: Sequence[StrategyPack],
        kind: str = "random",
        window_days: int = 60,
        min_packs: int = 5,
        seed: int = 0,
    ) -> None:
        self.packs = sorted(packs, key=lambda p: (p.as_of, p.symbol))
        self.kind = kind
        self.window_days = int(window_days)
        self.min_packs = int(min_packs)
        self._rng = random.Random(seed)
        self._exhausted = False

    def sample(self) -> tuple[list[StrategyPack], date, date]:
        if not self.packs:
            return [], date.today(), date.today()
        if self.kind == "chronological":
            if self._exhausted:
                return [], date.today(), date.today()
            self._exhausted = True
            return (
                list(self.packs),
                self.packs[0].as_of.date(),
                self.packs[-1].as_of.date(),
            )
        first = self.packs[0].as_of.date()
        last = self.packs[-1].as_of.date()
        if (last - first).days <= self.window_days:
            return list(self.packs), first, last
        for _ in range(20):
            offset_days = self._rng.randint(
                0, max(0, (last - first).days - self.window_days)
            )
            start = first + timedelta(days=offset_days)
            end = start + timedelta(days=self.window_days)
            window = [
                p for p in self.packs
                if start <= p.as_of.date() <= end
            ]
            if len(window) >= self.min_packs:
                return window, start, end
        return list(self.packs), first, last


# ---------------------------------------------------------------------
class MultiStrategySwingTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        bars: Sequence[MarketBar],
        packs: Sequence[StrategyPack],
        feature_frames: Sequence[FeatureFrame],
        feature_names: tuple[str, ...],
        n_strategies: int,
        sampler_kind: str = "random",
        sampler_seed: int = 0,
        sampler_window_days: int = 60,
        cost_model: EquityExecutionModel | None = None,
        reward_model: RewardModel | None = None,
        execution_simulator: ExecutionSimulator | None = None,
        starting_equity: float = 100_000.0,
        max_steps_per_episode: int | None = None,
        illegal_action_penalty: float = 0.5,
        skip_counterfactual_mode: str = "highest_signal",
    ) -> None:
        """
        ``skip_counterfactual_mode`` controls which counterfactual
        outcome the skip-reward mirrors. Choices:

        - ``"highest_signal"`` (default, FIX-26): use the counter-
          factual of the strategy with the highest ``signal_strength``
          among those that fired. Uses prior information only — no
          hindsight peek into realized outcomes. Avoids the max-over-
          noise bias of ``"max"``.
        - ``"max"``: use the maximum-return counterfactual among
          fired strategies. This is a hindsight peek — biased upward
          as the number of fired strategies grows. Kept for
          reproducibility of pre-FIX-26 results; not recommended for
          new experiments.
        - ``"mean"``: average the counterfactuals across fired
          strategies. Removes the max-over-noise bias but treats
          unfired-when-they-shouldn't-have-fired strategies as votes
          (debatable).
        - ``"none"``: pass ``None`` to ``reward_for_skip`` — the
          reward model returns 0 for skips. Use when you want the
          policy to be neutral on skips.
        """
        skip_counterfactual_mode = str(skip_counterfactual_mode)
        if skip_counterfactual_mode not in {"highest_signal", "max", "mean", "none"}:
            raise ValueError(
                f"skip_counterfactual_mode={skip_counterfactual_mode!r} "
                f"not in {{highest_signal, max, mean, none}}"
            )
        super().__init__()
        self.bars = _PackBars()
        self.bars.add(bars)
        self.packs = sorted(packs, key=lambda p: (p.as_of, p.symbol))
        self.feature_frames_by_key = {
            (f.symbol, f.as_of): f for f in feature_frames
        }
        self.n_strategies = int(n_strategies)
        self.observation_builder = MultiStrategyObservationBuilder(
            feature_names=feature_names,
            n_strategies=self.n_strategies,
        )
        self.cost_model = cost_model or EquityExecutionModel()
        self.reward_model = reward_model or RewardModel()
        self.execution_simulator = execution_simulator or ExecutionSimulator()
        self.starting_equity = float(starting_equity)
        self.max_steps_per_episode = max_steps_per_episode
        self.illegal_action_penalty = float(illegal_action_penalty)
        self.skip_counterfactual_mode = skip_counterfactual_mode

        self.sampler_kind = sampler_kind
        self.sampler_seed = sampler_seed
        self.sampler_window_days = sampler_window_days

        # Action: 0 = skip, 1..N = take strategy (k-1).
        self.action_space = spaces.Discrete(1 + self.n_strategies)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.observation_builder.dim,),
            dtype=np.float32,
        )

        self._episode_packs: list[StrategyPack] = []
        self._episode_start: date | None = None
        self._episode_end: date | None = None
        self._idx: int = 0
        self._steps: int = 0

    # ---- gym API ------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        sampler_seed = self.sampler_seed if seed is None else seed
        sampler = _PackEpisodeSampler(
            self.packs, kind=self.sampler_kind,
            window_days=self.sampler_window_days,
            seed=sampler_seed,
        )
        self._episode_packs, self._episode_start, self._episode_end = sampler.sample()
        self._idx = 0
        self._steps = 0
        if not self._episode_packs:
            return (
                np.zeros(self.observation_builder.dim, dtype=np.float32),
                {"empty_episode": True},
            )
        first_pack = self._episode_packs[0]
        return self._build_obs(first_pack), {
            "episode_start": self._episode_start.isoformat(),
            "episode_end": self._episode_end.isoformat(),
            "n_packs": len(self._episode_packs),
            "fired_mask": self.observation_builder.fired_mask(first_pack).tolist(),
        }

    # ---- action masking (FEAT-29) -------------------------------
    def action_masks(self) -> np.ndarray:
        """Boolean mask over the Discrete(N+1) action space.

        FEAT-29: returned to ``sb3-contrib.MaskablePPO`` so the policy
        cannot select non-fired strategy slots. The contract is:

        - index 0 (skip) is **always** True; skipping is always legal.
        - indices 1..N are True iff the strategy at that slot fired
          on the *current* pack (i.e. ``pack.candidates[k-1] is not
          None``).

        Vanilla PPO/DQN ignore this method, so exposing it on the env
        is harmless for the unmasked v002 variant — the masking only
        engages when the trainer wires the env to MaskablePPO.

        Edge case: when there is no current pack (between episodes,
        or if reset returned an empty episode), the mask is all-True
        — sb3-contrib never asks for masks outside an active step,
        but we return something defined just in case.
        """
        if not self._episode_packs or self._idx >= len(self._episode_packs):
            return np.ones(1 + self.n_strategies, dtype=bool)
        pack = self._episode_packs[self._idx]
        mask = np.zeros(1 + self.n_strategies, dtype=bool)
        mask[0] = True  # skip is always legal
        for k in range(self.n_strategies):
            if k < len(pack.candidates) and pack.candidates[k] is not None:
                mask[1 + k] = True
        return mask

    def step(self, action: int):
        if not self._episode_packs:
            return (
                np.zeros(self.observation_builder.dim, dtype=np.float32),
                0.0, True, False, {"reason": "empty_episode"},
            )
        pack = self._episode_packs[self._idx]
        reward, info = self._step_for_pack(pack, int(action))

        self._idx += 1
        self._steps += 1
        terminated = self._idx >= len(self._episode_packs)
        truncated = (
            self.max_steps_per_episode is not None
            and self._steps >= self.max_steps_per_episode
        )
        if not terminated and not truncated:
            next_pack = self._episode_packs[self._idx]
            next_obs = self._build_obs(next_pack)
            info["next_fired_mask"] = self.observation_builder.fired_mask(next_pack).tolist()
        else:
            next_obs = np.zeros(self.observation_builder.dim, dtype=np.float32)
        return next_obs, float(reward), bool(terminated), bool(truncated), info

    # ---- internals ----------------------------------------------
    def _build_obs(self, pack: StrategyPack) -> np.ndarray:
        frame = self.feature_frames_by_key.get((pack.symbol, pack.as_of))
        if frame is None:
            return np.zeros(self.observation_builder.dim, dtype=np.float32)
        ps = PortfolioState(
            as_of=pack.as_of, cash=self.starting_equity,
            equity=self.starting_equity,
        )
        return self.observation_builder.build(pack, frame, ps)

    def _simulate_take(self, candidate: CandidateTrade):
        frame = self.feature_frames_by_key.get((candidate.symbol, candidate.as_of))
        atr_pct = float(frame.values.get("atr_pct_14", 0.02)) if frame else 0.02
        rv20 = float(frame.values.get("realized_vol_20", 0.20)) if frame else 0.20
        vol_percentile = min(1.0, max(0.0, rv20 / 0.6))
        adv = float(frame.values.get("dollar_volume", 0.0)) if frame else 0.0
        notional = self.starting_equity * candidate.base_size_pct
        cost_bps = self.cost_model.cost_bps(
            atr_pct=atr_pct,
            volatility_percentile=vol_percentile,
            in_event_window=False,
            notional=notional,
            avg_dollar_volume=adv,
        )
        bars = self.bars.by_symbol.get(candidate.symbol) or []
        entry_idx = self.bars.find_index(candidate.symbol, candidate.as_of)
        return self.execution_simulator.simulate(
            bars=bars,
            entry_index=entry_idx,
            size_pct=candidate.base_size_pct,
            max_holding_days=candidate.max_holding_days,
            cost_bps=cost_bps,
            atr_pct=atr_pct,
            starting_equity=self.starting_equity,
        )

    def _skip_counterfactual(self, pack: StrategyPack):
        """The counterfactual outcome the skip-reward mirrors.

        Mode is set via ``skip_counterfactual_mode`` constructor arg.
        See FIX-26 issue + class docstring for the rationale on
        each mode. Returns either a TradeOutcome (which
        ``reward_for_skip`` uses to compute its mirrored reward),
        or None (which yields 0 reward).
        """
        mode = self.skip_counterfactual_mode

        if mode == "none":
            return None

        if mode == "highest_signal":
            # Pick the strategy with the highest signal_strength
            # among those that fired — uses prior information only,
            # no hindsight peek into realized returns.
            chosen = None
            best_strength = -1.0
            for c in pack.candidates:
                if c is None:
                    continue
                if c.signal_strength > best_strength:
                    chosen = c
                    best_strength = c.signal_strength
            if chosen is None:
                return None
            return self._simulate_take(chosen)

        if mode == "max":
            # Hindsight-best — biased upward as N grows.
            best = None
            for c in pack.candidates:
                if c is None:
                    continue
                outcome = self._simulate_take(c)
                if outcome is None:
                    continue
                if best is None or outcome.return_pct > best.return_pct:
                    best = outcome
            return best

        if mode == "mean":
            # Average across fired strategies. Synthesize a
            # TradeOutcome with the mean return_pct so the existing
            # reward_for_skip machinery works unchanged. Other fields
            # taken from one representative outcome.
            outcomes = []
            for c in pack.candidates:
                if c is None:
                    continue
                o = self._simulate_take(c)
                if o is not None:
                    outcomes.append(o)
            if not outcomes:
                return None
            mean_ret = sum(o.return_pct for o in outcomes) / len(outcomes)
            mean_asset = sum(o.asset_return_pct for o in outcomes) / len(outcomes)
            from dataclasses import replace
            return replace(
                outcomes[0],
                return_pct=mean_ret,
                raw_return_pct=mean_asset,
                asset_return_pct=mean_asset,
            )

        # Unreachable given the validation in __init__.
        raise AssertionError(f"unknown skip_counterfactual_mode: {mode!r}")

    def _step_for_pack(self, pack: StrategyPack, action: int):
        # action == 0 -> skip
        if action == 0:
            cf = self._skip_counterfactual(pack)
            reward = self.reward_model.reward_for_skip(cf)
            return reward, {
                "action": "skip",
                "n_fired": pack.n_fired,
                "best_cf_return": cf.return_pct if cf else None,
                "symbol": pack.symbol,
                "pack_as_of": pack.as_of.isoformat(),
            }
        # action 1..N -> take strategy (action-1)
        idx = action - 1
        if not (0 <= idx < self.n_strategies):
            return -self.illegal_action_penalty, {
                "action": "illegal_out_of_range",
                "raw_action": int(action),
            }
        chosen = pack.candidates[idx]
        if chosen is None:
            return -self.illegal_action_penalty, {
                "action": "illegal_strategy_not_fired",
                "strategy_idx": idx,
                "symbol": pack.symbol,
                "pack_as_of": pack.as_of.isoformat(),
            }
        outcome = self._simulate_take(chosen)
        if outcome is None:
            return 0.0, {
                "action": "take_no_data",
                "strategy_idx": idx,
                "candidate_id": chosen.candidate_id,
            }
        reward = self.reward_model.reward_for_take(
            outcome, max_holding_days=chosen.max_holding_days
        )
        return reward, {
            "action": "take",
            "strategy_idx": idx,
            "strategy_id": chosen.strategy_id,
            "raw_return": outcome.raw_return_pct,
            "net_return": outcome.return_pct,
            "exit_reason": outcome.exit_reason,
            "cost_bps": outcome.cost_bps,
            "holding_days": outcome.holding_days,
            "candidate_id": chosen.candidate_id,
            "symbol": chosen.symbol,
            # FIX-#51: needed by _evaluate for TradeRecord construction.
            "entry_date": outcome.entry_timestamp.date(),
            "exit_date": outcome.exit_timestamp.date(),
            "size_pct": chosen.base_size_pct,
        }
