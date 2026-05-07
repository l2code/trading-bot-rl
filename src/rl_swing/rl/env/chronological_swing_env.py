"""ChronologicalSwingEnv — v3 chronological gym.Env (FEAT-32 M1).

Steps once per trading day. Action lattice is a small Discrete(K+1)
where action 0 = no-op (don't open) and actions 1..K = take the
top-k fired packs by signal_strength on today's slate, subject to
the portfolio's gross-exposure + cash budgets.

Per-day reward = today's portfolio P&L (from open MtM + realized
exits) − DD penalty − turnover penalty. Same FIX-#49 calibration as
v002 (target_risk_pct=0.02 etc.); see ``RewardModel``.

This module is the chronological counterpart to ``MultiStrategyEnv``
(per-pack stepping). It deliberately reuses ``MultiStrategyPacker``,
``EquityExecutionModel``, ``MarketBar``, ``StrategyPack`` so the
v002 / v003 split is a difference of decision shape, not of plumbing.

Out of scope for M1:
  - ATR stops mid-position (filed for M3+ if motivated).
  - Action masking (action lattice is small; explore freely first).
  - Per-symbol observation explosion (M1 pools symbol features to a
    small fixed-dim block; #34 set/attention encoder reuse is a
    follow-up).
"""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_swing.domain import FeatureFrame, MarketBar
from rl_swing.rl.env.cost_model import EquityExecutionModel
from rl_swing.rl.env.portfolio_state_tracker import PortfolioStateTracker
from rl_swing.rl.env.reward_model import RewardModel
from rl_swing.strategies.multi_strategy_packer import StrategyPack

_log = logging.getLogger(__name__)


# Observation layout (fixed dim per env construction):
#   [pack_n_fired_today,
#    pack_signal_max, pack_signal_mean, pack_signal_std,
#    pack_signal_gap_top2, pack_all_fired_indicator,
#    portfolio_cash_pct, portfolio_gross_exposure_pct,
#    portfolio_n_open_norm,           (n_open / 10)
#    portfolio_drawdown_pct,
#    portfolio_realized_pnl_pct,
#    day_index_norm,                  (day_idx / total_days)
#   ]
# 12 dims; small + sufficient for M1's smoke. Future milestones can
# stitch in market-feature pooling and the FEAT-34 SlateEncoder.
OBS_DIM = 12


@dataclass
class _DaySlate:
    """All packs that fired on a single trading day, sorted by
    signal_strength desc (top-k semantics for the action lattice)."""
    as_of: date
    packs: list[StrategyPack] = field(default_factory=list)


class ChronologicalSwingEnv(gym.Env):
    """v3 chronological env. Steps per trading day.

    Action space: ``Discrete(1 + max_top_k)``. Action 0 is no-op;
    action k>=1 attempts to open the top-k fired packs of the day
    by signal_strength, subject to budget rules in
    ``PortfolioStateTracker.open_position``.

    Episode boundary: a contiguous window of trading days; ``reset``
    samples a window per ``sampler_kind``. Like the v002 env, we
    support 'random' (random window of ``window_days`` length) and
    'chronological' (the full provided range, single episode).

    Reward shape (per step): same building blocks as v002 but applied
    daily — ``RewardModel.reward_for_take`` and ``reward_for_skip``
    aren't directly reusable here (they're per-trade); M1 computes
    reward inline from daily_pnl + dd_penalty + turnover_penalty.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        bars: Sequence[MarketBar],
        packs: Sequence[StrategyPack],
        feature_frames: Sequence[FeatureFrame],
        cost_model: EquityExecutionModel | None = None,
        reward_model: RewardModel | None = None,
        starting_equity: float = 100_000.0,
        max_top_k: int = 2,
        sampler_kind: str = "random",
        sampler_seed: int = 0,
        sampler_window_days: int = 60,
        max_steps_per_episode: int | None = None,
        episode_min_days: int = 5,
    ) -> None:
        super().__init__()
        # Group bars and packs by date for fast per-day lookup.
        self.bars = list(bars)
        self.packs = list(packs)
        self.feature_frames = list(feature_frames)
        self.cost_model = cost_model or EquityExecutionModel()
        self.reward_model = reward_model or RewardModel()
        self.starting_equity = float(starting_equity)
        self.max_top_k = int(max_top_k)
        self.sampler_kind = sampler_kind
        self.sampler_seed = sampler_seed
        self.sampler_window_days = int(sampler_window_days)
        self.max_steps_per_episode = max_steps_per_episode
        self.episode_min_days = int(episode_min_days)

        # Pre-compute close prices by (symbol, date) for fast lookup.
        self._close_by_symbol_date: dict[tuple[str, date], float] = {}
        for b in self.bars:
            self._close_by_symbol_date[(b.symbol, b.timestamp.date())] = float(b.close)
        self._open_by_symbol_date: dict[tuple[str, date], float] = {}
        for b in self.bars:
            self._open_by_symbol_date[(b.symbol, b.timestamp.date())] = float(b.open)

        # Per-day slate (sorted by signal_strength desc).
        slates_by_day: dict[date, list[StrategyPack]] = defaultdict(list)
        for p in self.packs:
            slates_by_day[p.as_of.date()].append(p)
        for d in slates_by_day:
            # Sort by max signal_strength among fired candidates desc; ties broken by symbol.
            def _max_signal(pack: StrategyPack) -> float:
                return max(
                    (float(c.signal_strength) for c in pack.candidates if c is not None),
                    default=-1.0,
                )
            slates_by_day[d].sort(key=lambda p: (-_max_signal(p), p.symbol))
        self._slates_by_day: dict[date, list[StrategyPack]] = dict(slates_by_day)
        # All trading days for which we have at least one bar.
        all_days: set[date] = set()
        for b in self.bars:
            all_days.add(b.timestamp.date())
        self._trading_days: list[date] = sorted(all_days)

        # Action / observation spaces.
        self.action_space = spaces.Discrete(1 + self.max_top_k)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32,
        )

        # Per-episode state (filled in reset()).
        self._episode_days: list[date] = []
        self._day_idx: int = 0
        self._steps: int = 0
        self.tracker: PortfolioStateTracker = PortfolioStateTracker(
            starting_equity=self.starting_equity,
        )
        self._n_trades_today: int = 0  # for turnover penalty
        self._n_packs_today: int = 0
        self._rng = random.Random(self.sampler_seed)

    # ---- gym API ----------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)
        # Sample the episode window.
        if not self._trading_days:
            return (
                np.zeros(OBS_DIM, dtype=np.float32),
                {"empty_episode": True},
            )
        if self.sampler_kind == "chronological":
            self._episode_days = list(self._trading_days)
        else:
            n = len(self._trading_days)
            window = min(self.sampler_window_days, n)
            start_idx = 0 if n - window <= 0 else self._rng.randint(0, n - window)
            self._episode_days = self._trading_days[start_idx:start_idx + window]
        if len(self._episode_days) < self.episode_min_days:
            return (
                np.zeros(OBS_DIM, dtype=np.float32),
                {"empty_episode": True, "n_days": len(self._episode_days)},
            )
        # Reset portfolio.
        self.tracker = PortfolioStateTracker(starting_equity=self.starting_equity)
        self._day_idx = 0
        self._steps = 0
        return self._build_obs(self._episode_days[0]), {
            "episode_start": self._episode_days[0].isoformat(),
            "episode_end": self._episode_days[-1].isoformat(),
            "n_days": len(self._episode_days),
        }

    def step(self, action: int):
        if not self._episode_days:
            return (
                np.zeros(OBS_DIM, dtype=np.float32),
                0.0, True, False, {"reason": "empty_episode"},
            )
        action = int(action)
        if action < 0 or action >= self.action_space.n:
            # Defensive: clamp to no-op rather than crash on bad action.
            action = 0

        today = self._episode_days[self._day_idx]
        slate = self._slates_by_day.get(today, [])
        self._n_packs_today = len(slate)

        # 1) Open trades per the action lattice.
        n_to_open = min(action, len(slate)) if action > 0 else 0
        n_opened = 0
        for j in range(n_to_open):
            pack = slate[j]
            chosen = self._choose_top_signal_candidate(pack)
            if chosen is None:
                continue
            entry_open = self._open_by_symbol_date.get((chosen.symbol, today))
            if entry_open is None or entry_open <= 0:
                continue
            cost_bps = self._cost_for_candidate(chosen, today)
            ok = self.tracker.open_position(
                symbol=chosen.symbol,
                entry_date=today,
                entry_price=float(entry_open),
                size_pct=float(chosen.base_size_pct),
                max_holding_days=int(chosen.max_holding_days),
                cost_bps_round_trip=float(cost_bps),
                candidate_id=str(chosen.candidate_id),
            )
            if ok:
                n_opened += 1
        self._n_trades_today = n_opened

        # 2) Advance one day (mark-to-market existing + new positions).
        close_today = {sym: c for (sym, d), c in self._close_by_symbol_date.items() if d == today}
        daily_pnl = self.tracker.advance_one_day(today, close_today)

        # 3) Compute reward inline.
        reward = self._reward(daily_pnl, n_opened, len(slate))

        # 4) Tick.
        self._day_idx += 1
        self._steps += 1
        terminated = self._day_idx >= len(self._episode_days)
        truncated = (
            self.max_steps_per_episode is not None
            and self._steps >= self.max_steps_per_episode
        )
        if terminated:
            # Close out any remaining positions at the last day's close.
            self.tracker.close_all(today, close_today)
            next_obs = np.zeros(OBS_DIM, dtype=np.float32)
        else:
            next_obs = self._build_obs(self._episode_days[self._day_idx])
        info = {
            "day": today.isoformat(),
            "action": int(action),
            "n_opened_today": n_opened,
            "n_packs_today": len(slate),
            "daily_pnl_pct": float(daily_pnl),
            **{f"port_{k}": v for k, v in self.tracker.summary().items()},
        }
        return next_obs, float(reward), bool(terminated), bool(truncated), info

    # ---- helpers ----------------------------------------------------
    def _choose_top_signal_candidate(self, pack: StrategyPack):
        """Pick the highest signal_strength candidate within a pack
        (intra-pack tie-break: lower slot_idx wins). Mirrors v002's
        first_fired/highest_signal selectors."""
        best_idx = -1
        best_signal = -1.0
        for i, c in enumerate(pack.candidates):
            if c is None:
                continue
            s = float(c.signal_strength)
            if s > best_signal:
                best_signal = s
                best_idx = i
        if best_idx < 0:
            return None
        return pack.candidates[best_idx]

    def _cost_for_candidate(self, candidate, today: date) -> float:
        """Look up today's frame for candidate.symbol to derive cost_bps.
        Mirrors selector_v002._simulate_take's cost calculation."""
        frame = next(
            (f for f in self.feature_frames
             if f.symbol == candidate.symbol and f.as_of.date() == today),
            None,
        )
        atr_pct = float(frame.values.get("atr_pct_14", 0.02)) if frame else 0.02
        rv20 = float(frame.values.get("realized_vol_20", 0.20)) if frame else 0.20
        vol_percentile = min(1.0, max(0.0, rv20 / 0.6))
        adv = float(frame.values.get("dollar_volume", 0.0)) if frame else 0.0
        notional = self.starting_equity * candidate.base_size_pct
        cost_bps = self.cost_model.cost_bps(
            atr_pct=atr_pct, volatility_percentile=vol_percentile,
            in_event_window=False, notional=notional, avg_dollar_volume=adv,
        )
        # cost_model.cost_bps returns one-side; round-trip = 2x.
        return 2.0 * float(cost_bps)

    def _reward(
        self, daily_pnl_pct: float, n_opened: int, n_packs: int,
    ) -> float:
        """Per-day reward = scaled daily P&L − DD penalty − turnover penalty.

        Scale by 1/target_risk_pct so the reward sits in roughly the same
        magnitude as v002's risk-adjusted reward. DD penalty: same shape
        as v002's drawdown_penalty_weight × current_dd_pct / target_risk_pct.
        Turnover penalty: turnover_penalty_weight × (n_opened / max(1, n_packs)),
        which discourages opening on every fired pack each day.
        """
        rm = self.reward_model
        target_risk = max(rm.target_risk_pct, 1e-9)
        risk_adj = daily_pnl_pct / target_risk
        dd_pen = rm.drawdown_penalty_weight * (
            self.tracker.current_drawdown_pct / target_risk
        )
        turnover_ratio = n_opened / float(n_packs) if n_packs > 0 else 0.0
        turnover_pen = rm.turnover_penalty_weight * turnover_ratio
        reward = risk_adj - dd_pen - turnover_pen
        # Apply v002-style clipping for stability.
        clip = float(getattr(rm, "reward_clip", 5.0) or 5.0)
        return float(max(-clip, min(clip, reward)))

    # ---- action masking (FEAT-32 M3) ---------------------------------
    def action_masks(self) -> np.ndarray:
        """Boolean mask over the ``Discrete(1 + max_top_k)`` action space.

        Returned to ``sb3-contrib.MaskablePPO`` so the policy cannot
        select 'take top-k' actions when fewer than k packs fired today
        (those actions would no-op anyway via the env's
        ``min(action, len(slate))`` clamp; masking lets the policy
        avoid wasting probability mass on non-actions).

        Mask shape: ``[True, fired_slot_0, ..., fired_slot_K-1]`` per
        the FEAT-29 v002 pattern. Concretely:

          - index 0 (no-op) is **always** True; skipping is always legal.
          - index k (1 ≤ k ≤ max_top_k) is True iff today's slate has
            at least k fired packs (i.e. ``len(slate) >= k``), so 'take
            top-k' has something to take.

        Vanilla PPO/DQN ignore this method, so exposing it is harmless
        for the unmasked v003 default — masking only engages when the
        trainer wires the env to MaskablePPO.

        Edge case: when there's no current trading day (between
        episodes, or empty episode), the mask is ``[True, False, ...]``
        — only no-op is legal. sb3-contrib never asks for masks outside
        an active step, but we return something defined just in case.
        """
        n_actions = 1 + self.max_top_k
        if not self._episode_days or self._day_idx >= len(self._episode_days):
            mask = np.zeros(n_actions, dtype=bool)
            mask[0] = True
            return mask
        today = self._episode_days[self._day_idx]
        slate = self._slates_by_day.get(today, [])
        n_fired = len(slate)
        mask = np.zeros(n_actions, dtype=bool)
        mask[0] = True  # no-op always legal
        for k in range(1, n_actions):
            if n_fired >= k:
                mask[k] = True
        return mask

    def _build_obs(self, today: date) -> np.ndarray:
        slate = self._slates_by_day.get(today, [])
        # Pack-slate aggregate stats.
        signals: list[float] = []
        n_fired_packs = 0
        for pack in slate:
            for c in pack.candidates:
                if c is not None:
                    signals.append(float(c.signal_strength))
                    n_fired_packs = max(n_fired_packs, 1)
        n_fired = len(signals)
        if n_fired == 0:
            s_max = s_mean = s_std = s_gap = 0.0
        else:
            signals_sorted = sorted(signals, reverse=True)
            s_max = signals_sorted[0]
            s_mean = sum(signals) / n_fired
            if n_fired >= 2:
                var = sum((s - s_mean) ** 2 for s in signals) / n_fired
                s_std = var ** 0.5
                s_gap = signals_sorted[0] - signals_sorted[1]
            else:
                s_std = 0.0
                s_gap = 0.0
        all_fired_ind = 1.0 if (slate and all(p.n_fired == len(p.candidates) for p in slate)) else 0.0
        port = self.tracker.summary()
        n_days = max(1, len(self._episode_days))
        day_norm = float(self._day_idx) / float(n_days)
        obs = np.array([
            float(n_fired),
            float(s_max),
            float(s_mean),
            float(s_std),
            float(s_gap),
            float(all_fired_ind),
            float(port["cash_pct"]),
            float(port["gross_exposure_pct"]),
            float(port["n_open"] / 10.0),
            float(port["current_drawdown_pct"]),
            float(port["realized_pnl_pct"]),
            float(day_norm),
        ], dtype=np.float32)
        assert obs.shape == (OBS_DIM,)
        return obs
