"""PortfolioStateTracker — per-day portfolio bookkeeping for v3
chronological env (FEAT-32 M1).

v002 stepped once per (symbol, date) pack with a stateless portfolio.
v3 steps once per trading day, with positions evolving across days.
This module owns the portfolio state that has to survive between
env steps: which positions are open, what they cost to enter, what
day they entered, max-holding-days clock, plus rolling cash and
drawdown.

Pure-data + pure-function design. The env owns the tracker's
lifecycle; this module knows nothing about gym, sb3, or yfinance.

Out of scope for M1 (filed for later milestones):
  - ATR stops (M3+ if motivated by results).
  - Per-day cost dynamics from the EquityExecutionModel; we charge
    a single round-trip cost at entry/exit using the entry-day
    cost_bps quote, which mirrors v002's accounting.
  - Position sizing under risk budget beyond gross exposure cap;
    M1 uses the candidate's base_size_pct directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True, slots=True)
class OpenPosition:
    """A single open long position. Frozen — the tracker creates a
    fresh OpenPosition with an updated days_held when advancing days.
    """
    symbol: str
    entry_date: date
    entry_price: float
    size_pct: float            # fraction of starting equity at risk
    max_holding_days: int      # exit when days_held >= this
    cost_bps_round_trip: float  # both sides; accounted once at entry
    days_held: int = 0
    candidate_id: str = ""     # optional traceability

    def with_held(self, days_held: int) -> OpenPosition:
        return OpenPosition(
            symbol=self.symbol,
            entry_date=self.entry_date,
            entry_price=self.entry_price,
            size_pct=self.size_pct,
            max_holding_days=self.max_holding_days,
            cost_bps_round_trip=self.cost_bps_round_trip,
            days_held=days_held,
            candidate_id=self.candidate_id,
        )

    def return_pct_at_close(self, close: float) -> float:
        """Portfolio-scaled return contribution from this position
        if marked at ``close``. Cost drag NOT included — caller
        accounts for it at entry/exit."""
        if self.entry_price <= 0:
            return 0.0
        asset_ret = (close - self.entry_price) / self.entry_price
        return self.size_pct * asset_ret

    def is_due_to_exit(self, on_day_index: int, entry_day_index: int) -> bool:
        """Exit when days_held >= max_holding_days. Tracker advances
        days_held; here we provide the rule for clarity / testing."""
        return (on_day_index - entry_day_index) >= self.max_holding_days


@dataclass
class ClosedTrade:
    """Realized trade — pushed by the tracker when a position exits.
    Used by the env's metrics builder to construct the per-day P&L
    series compatible with ``acceptance_gate.evaluate_gate``."""
    symbol: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    size_pct: float
    cost_bps_round_trip: float
    candidate_id: str = ""

    @property
    def asset_return_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price

    @property
    def net_return_pct(self) -> float:
        """Portfolio-scaled, cost-adjusted return. Mirrors
        ExecutionSimulator's net_return semantics."""
        cost_drag = self.cost_bps_round_trip / 10_000.0
        return self.size_pct * (self.asset_return_pct - cost_drag)


@dataclass
class PortfolioStateTracker:
    """Per-day portfolio bookkeeping.

    The tracker is mutable across env steps. Each call to
    ``advance_one_day`` consumes the current day's close prices,
    marks every open position to market, exits any due, accrues
    realized P&L into the day's slot, and updates cash + drawdown.

    Cash semantics:
      - starting_equity is the reference notional used to scale
        size_pct into dollar exposure. Cash is tracked as a fraction
        of starting_equity (so cash=1.0 means 100% of starting equity
        is uninvested).
      - Opening a position at size_pct reduces cash by size_pct +
        cost (one side at entry).
      - Closing a position at size_pct returns cash equal to
        size_pct × (1 + return_pct) − cost (other side at exit).
      - Negative cash means leverage; the env can refuse to open
        new trades when cash drops below a threshold (M1: refuse
        when cash < 0.05).

    Drawdown semantics:
      - Tracked as: peak portfolio value (cash + open MtM at close)
        and current portfolio value. drawdown_pct = (peak - cur) / peak.
      - Reset never; episode-level peak persists across the episode.
    """
    starting_equity: float = 100_000.0
    open_positions: list[OpenPosition] = field(default_factory=list)
    closed_trades: list[ClosedTrade] = field(default_factory=list)
    cash_pct: float = 1.0          # fraction of starting_equity uninvested
    realized_pnl_pct: float = 0.0  # cumulative since episode start
    daily_pnl_history: list[tuple[date, float]] = field(default_factory=list)
    peak_value_pct: float = 1.0    # max(cash + open_mtm_pct) so far
    current_drawdown_pct: float = 0.0
    n_trades_opened: int = 0
    n_trades_closed: int = 0

    # ------------------------------------------------------------------
    def open_position(
        self,
        *,
        symbol: str,
        entry_date: date,
        entry_price: float,
        size_pct: float,
        max_holding_days: int,
        cost_bps_round_trip: float,
        candidate_id: str = "",
    ) -> bool:
        """Try to open a position. Returns True if the position
        was opened, False if budget didn't fit.

        Budget rule (M1):
          - Refuse if size_pct would push gross_exposure_pct > 1.0.
          - Refuse if cash_pct after the entry-side cost drops below 0.05.
        """
        gross = sum(p.size_pct for p in self.open_positions)
        cost_drag_one_side = (cost_bps_round_trip / 2.0) / 10_000.0
        cash_after = self.cash_pct - size_pct - (size_pct * cost_drag_one_side)
        if gross + size_pct > 1.0:
            return False
        if cash_after < 0.05:
            return False
        self.open_positions.append(OpenPosition(
            symbol=symbol,
            entry_date=entry_date,
            entry_price=float(entry_price),
            size_pct=float(size_pct),
            max_holding_days=int(max_holding_days),
            cost_bps_round_trip=float(cost_bps_round_trip),
            days_held=0,
            candidate_id=candidate_id,
        ))
        self.cash_pct = cash_after
        self.n_trades_opened += 1
        return True

    def advance_one_day(
        self,
        as_of: date,
        close_by_symbol: dict[str, float],
    ) -> float:
        """Mark all open positions to market on close, exit any due,
        record the day's realized + unrealized P&L. Returns today's
        portfolio P&L as a fraction of starting_equity (positive =
        gain).
        """
        # 1) increment days_held on every open position
        new_open: list[OpenPosition] = [p.with_held(p.days_held + 1) for p in self.open_positions]
        # 2) compute mtm and identify exits
        unrealized_today = 0.0
        retained: list[OpenPosition] = []
        realized_today = 0.0
        for p in new_open:
            close = close_by_symbol.get(p.symbol)
            if close is None:
                # No data today (e.g. market holiday for this symbol).
                # Keep the position; mtm contribution is 0.
                retained.append(p)
                continue
            ret_today = p.return_pct_at_close(close)
            if p.days_held >= p.max_holding_days:
                # Exit at today's close.
                cost_drag = p.cost_bps_round_trip / 10_000.0
                net_pct = p.size_pct * ((close - p.entry_price) / p.entry_price - cost_drag)
                realized_today += net_pct
                self.closed_trades.append(ClosedTrade(
                    symbol=p.symbol, entry_date=p.entry_date, exit_date=as_of,
                    entry_price=p.entry_price, exit_price=close,
                    size_pct=p.size_pct,
                    cost_bps_round_trip=p.cost_bps_round_trip,
                    candidate_id=p.candidate_id,
                ))
                # Return cash for this position (size + return − exit cost)
                cost_drag_one_side = (p.cost_bps_round_trip / 2.0) / 10_000.0
                exit_cash = p.size_pct * (1.0 + ret_today / max(p.size_pct, 1e-9))
                # Above is convoluted; clearer form:
                exit_cash = p.size_pct + (p.size_pct * (close - p.entry_price) / p.entry_price)
                self.cash_pct += exit_cash - (p.size_pct * cost_drag_one_side)
                self.n_trades_closed += 1
            else:
                retained.append(p)
                unrealized_today += ret_today
        self.open_positions = retained
        self.realized_pnl_pct += realized_today
        # 3) Today's portfolio P&L = realized exits + delta unrealized.
        # We approximate delta unrealized by the daily move: today's
        # unrealized minus yesterday's. To avoid storing yesterday's,
        # we instead define daily P&L as realized_today + d/dt(unrealized)
        # = realized_today + sum_p (return_pct_today - return_pct_yesterday).
        # Simpler approximation for M1: daily P&L ≈ realized_today +
        # net change in cumulative open-MtM since last call. Maintain
        # a state "last_open_mtm" to compute that delta.
        last_mtm = getattr(self, "_last_open_mtm_pct", 0.0)
        daily_pnl = realized_today + (unrealized_today - last_mtm)
        self._last_open_mtm_pct = unrealized_today
        self.daily_pnl_history.append((as_of, daily_pnl))
        # 4) DD bookkeeping
        portfolio_value_pct = self.cash_pct + unrealized_today
        if portfolio_value_pct > self.peak_value_pct:
            self.peak_value_pct = portfolio_value_pct
        peak = max(self.peak_value_pct, 1e-9)
        self.current_drawdown_pct = max(0.0, (peak - portfolio_value_pct) / peak)
        return daily_pnl

    def close_all(self, as_of: date, close_by_symbol: dict[str, float]) -> float:
        """Close every open position at today's close. Used for
        episode-end accounting. Returns realized P&L from these
        closures (does NOT include final-day unrealized — call
        advance_one_day first if needed)."""
        realized = 0.0
        for p in self.open_positions:
            close = close_by_symbol.get(p.symbol)
            if close is None:
                # Skip; in practice the env should provide closes for
                # all symbols on the final day.
                continue
            cost_drag = p.cost_bps_round_trip / 10_000.0
            net_pct = p.size_pct * ((close - p.entry_price) / p.entry_price - cost_drag)
            realized += net_pct
            self.closed_trades.append(ClosedTrade(
                symbol=p.symbol, entry_date=p.entry_date, exit_date=as_of,
                entry_price=p.entry_price, exit_price=close,
                size_pct=p.size_pct,
                cost_bps_round_trip=p.cost_bps_round_trip,
                candidate_id=p.candidate_id,
            ))
            cost_drag_one_side = (p.cost_bps_round_trip / 2.0) / 10_000.0
            exit_cash = p.size_pct + (p.size_pct * (close - p.entry_price) / p.entry_price)
            self.cash_pct += exit_cash - (p.size_pct * cost_drag_one_side)
            self.n_trades_closed += 1
        self.open_positions = []
        self.realized_pnl_pct += realized
        return realized

    # ------------------------------------------------------------------
    # Diagnostics consumed by the obs builder.
    @property
    def gross_exposure_pct(self) -> float:
        return sum(p.size_pct for p in self.open_positions)

    @property
    def n_open(self) -> int:
        return len(self.open_positions)

    def summary(self) -> dict[str, float]:
        return {
            "cash_pct": float(self.cash_pct),
            "gross_exposure_pct": float(self.gross_exposure_pct),
            "n_open": float(self.n_open),
            "current_drawdown_pct": float(self.current_drawdown_pct),
            "realized_pnl_pct": float(self.realized_pnl_pct),
            "n_trades_opened": float(self.n_trades_opened),
            "n_trades_closed": float(self.n_trades_closed),
        }
