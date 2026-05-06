"""SwingTradingEnv — Gymnasium env for the trade-filter MVP.

Each step presents one ``CandidateTrade`` to the agent. The agent
chooses ``skip / take_25 / take_50 / take_100`` and the env simulates
the outcome using the bars it has cached for that symbol.

The episode ends when the candidate sequence is exhausted or when the
``TerminationRule`` triggers (e.g. after a max number of steps in
training mode).

This env is deliberately thin — all the swappable pieces
(EpisodeSampler, ObservationBuilder, ActionMapper, RewardModel,
CostModel, ExecutionSimulator) are passed in.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_swing.domain import (
    CandidateTrade,
    FeatureFrame,
    MarketBar,
    PortfolioState,
)
from rl_swing.rl.env.action_mapper import to_size_multiplier
from rl_swing.rl.env.cost_model import EquityExecutionModel
from rl_swing.rl.env.episode_sampler import (
    ChronologicalSampler,
    Episode,
    RandomWindowSampler,
)
from rl_swing.rl.env.execution_simulator import ExecutionSimulator, TradeOutcome
from rl_swing.rl.env.observation_builder import ObservationBuilder
from rl_swing.rl.env.reward_model import RewardModel

_log = logging.getLogger(__name__)


@dataclass
class EnvBars:
    """Bars indexed by symbol for fast simulation."""
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
        # Binary search by timestamp.
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


class SwingTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        bars: Sequence[MarketBar],
        candidates: Sequence[CandidateTrade],
        feature_frames: Sequence[FeatureFrame],
        feature_names: tuple[str, ...],
        sampler_kind: str = "random",   # "random" | "chronological"
        sampler_seed: int = 0,
        sampler_window_days: int = 60,
        cost_model: EquityExecutionModel | None = None,
        reward_model: RewardModel | None = None,
        execution_simulator: ExecutionSimulator | None = None,
        starting_equity: float = 100_000.0,
        max_steps_per_episode: int | None = None,
    ) -> None:
        super().__init__()
        self.bars = EnvBars()
        self.bars.add(bars)
        self.candidates = sorted(candidates, key=lambda c: (c.as_of, c.symbol))
        self.feature_frames_by_key = {
            (f.symbol, f.as_of): f for f in feature_frames
        }
        self.observation_builder = ObservationBuilder(feature_names=feature_names)
        self.cost_model = cost_model or EquityExecutionModel()
        self.reward_model = reward_model or RewardModel()
        self.execution_simulator = execution_simulator or ExecutionSimulator()
        self.starting_equity = float(starting_equity)
        self.max_steps_per_episode = max_steps_per_episode

        self.sampler_kind = sampler_kind
        self.sampler_seed = sampler_seed
        self.sampler_window_days = sampler_window_days

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.observation_builder.dim,),
            dtype=np.float32,
        )

        self._episode: Episode | None = None
        self._idx: int = 0
        self._steps: int = 0
        self._last_outcome: TradeOutcome | None = None

    # -- gym API -------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        sampler_seed = self.sampler_seed if seed is None else seed
        if self.sampler_kind == "chronological":
            sampler = ChronologicalSampler(self.candidates)
        else:
            sampler = RandomWindowSampler(
                self.candidates,
                window_days=self.sampler_window_days,
                seed=sampler_seed,
            )
        self._episode = sampler.sample()
        self._idx = 0
        self._steps = 0
        if not self._episode.candidates:
            return np.zeros(self.observation_builder.dim, dtype=np.float32), {"empty_episode": True}
        return self._build_obs(self._episode.candidates[0]), {
            "episode_start": self._episode.start.isoformat(),
            "episode_end": self._episode.end.isoformat(),
            "n_candidates": len(self._episode.candidates),
        }

    def step(self, action: int):
        if self._episode is None or not self._episode.candidates:
            return (
                np.zeros(self.observation_builder.dim, dtype=np.float32),
                0.0, True, False, {"reason": "empty_episode"},
            )
        candidate = self._episode.candidates[self._idx]
        size_mult = to_size_multiplier(int(action))
        reward, info = self._step_for_candidate(candidate, size_mult)

        self._idx += 1
        self._steps += 1
        terminated = self._idx >= len(self._episode.candidates)
        truncated = (
            self.max_steps_per_episode is not None
            and self._steps >= self.max_steps_per_episode
        )
        next_obs = (
            self._build_obs(self._episode.candidates[self._idx])
            if not terminated and not truncated
            else np.zeros(self.observation_builder.dim, dtype=np.float32)
        )
        return next_obs, float(reward), bool(terminated), bool(truncated), info

    # -- internals -----------------------------------------------------
    def _build_obs(self, candidate: CandidateTrade) -> np.ndarray:
        frame = self.feature_frames_by_key.get((candidate.symbol, candidate.as_of))
        if frame is None:
            # Cold-start case — return zeros sized to the right shape.
            return np.zeros(self.observation_builder.dim, dtype=np.float32)
        # Simple zero-portfolio state in MVP env. Walk-forward harness
        # tracks portfolio outside the env.
        ps = PortfolioState(
            as_of=candidate.as_of, cash=self.starting_equity,
            equity=self.starting_equity,
        )
        return self.observation_builder.build(candidate, frame, ps)

    def _step_for_candidate(self, candidate: CandidateTrade, size_mult: float):
        frame = self.feature_frames_by_key.get((candidate.symbol, candidate.as_of))
        atr_pct = float(frame.values.get("atr_pct_14", 0.02)) if frame else 0.02
        rv20 = float(frame.values.get("realized_vol_20", 0.20)) if frame else 0.20
        # Approximate volatility percentile inside [0, 1] from realized vol.
        vol_percentile = min(1.0, max(0.0, rv20 / 0.6))
        adv = float(frame.values.get("dollar_volume", 0.0)) if frame else 0.0

        # For TAKE actions, cost is computed against the actually-sized
        # notional (notional-dependent market_impact scales with size).
        # FIX-#54: for SKIP actions, the counterfactual is simulated at
        # FULL base_size_pct, so we compute cost at full notional too —
        # otherwise the skip CF is too optimistic on larger / less-liquid
        # trades because it ignores impact cost. Pre-FIX-#54 the cost was
        # 0 (size_mult=0 → notional=0), zeroing impact.
        cf_size_mult = 1.0 if size_mult <= 0 else size_mult
        notional = self.starting_equity * candidate.base_size_pct * cf_size_mult
        cost_bps = self.cost_model.cost_bps(
            atr_pct=atr_pct,
            volatility_percentile=vol_percentile,
            in_event_window=False,
            notional=notional,
            avg_dollar_volume=adv,
        )

        bars = self.bars.by_symbol.get(candidate.symbol) or []
        entry_idx = self.bars.find_index(candidate.symbol, candidate.as_of)
        if size_mult > 0:
            outcome = self.execution_simulator.simulate(
                bars=bars,
                entry_index=entry_idx,
                size_pct=candidate.base_size_pct * size_mult,
                max_holding_days=candidate.max_holding_days,
                cost_bps=cost_bps,
                atr_pct=atr_pct,
                starting_equity=self.starting_equity,
            )
            self._last_outcome = outcome
            if outcome is None:
                return 0.0, {
                    "action": "take_no_data",
                    "candidate_id": candidate.candidate_id,
                }
            reward = self.reward_model.reward_for_take(
                outcome, max_holding_days=candidate.max_holding_days
            )
            info = {
                "action": "take",
                "size_mult": size_mult,
                "raw_return": outcome.raw_return_pct,
                "net_return": outcome.return_pct,
                "exit_reason": outcome.exit_reason,
                "cost_bps": outcome.cost_bps,
                "holding_days": outcome.holding_days,
                "candidate_id": candidate.candidate_id,
                "symbol": candidate.symbol,
                "strategy_id": candidate.strategy_id,
                # FIX-#51: needed by _evaluate to build TradeRecords
                # for daily-P&L-based checkpoint selection.
                "entry_date": outcome.entry_timestamp.date(),
                "exit_date": outcome.exit_timestamp.date(),
                "size_pct": candidate.base_size_pct * size_mult,
            }
            return reward, info
        # Skip: compute counterfactual at full size (so the agent learns
        # what it would have gotten).
        cf = self.execution_simulator.simulate(
            bars=bars,
            entry_index=entry_idx,
            size_pct=candidate.base_size_pct,
            max_holding_days=candidate.max_holding_days,
            cost_bps=cost_bps,
            atr_pct=atr_pct,
            starting_equity=self.starting_equity,
        )
        reward = self.reward_model.reward_for_skip(cf)
        info = {
            "action": "skip",
            "counterfactual_return": cf.return_pct if cf else None,
            "candidate_id": candidate.candidate_id,
            "symbol": candidate.symbol,
            "strategy_id": candidate.strategy_id,
        }
        return reward, info
