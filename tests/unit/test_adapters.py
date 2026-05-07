"""Adapter tests: providers, brokers, sqlite storage."""
from __future__ import annotations

from datetime import date, datetime

import pytest

from rl_swing.adapters.broker.alpaca_live_broker import AlpacaLiveBrokerAdapter
from rl_swing.adapters.broker.alpaca_paper_broker import AlpacaPaperBrokerAdapter
from rl_swing.adapters.broker.noop_shadow_broker import NoOpShadowBrokerAdapter
from rl_swing.adapters.broker.simulated_broker import SimulatedBrokerAdapter
from rl_swing.adapters.data.parquet_provider import ParquetProvider
from rl_swing.adapters.data.synthetic_provider import SyntheticProvider
from rl_swing.adapters.data.wrds_parquet_provider import WrdsParquetProvider
from rl_swing.adapters.data.yfinance_provider import YFinanceProvider
from rl_swing.adapters.storage.sqlite_repositories import (
    attach_audit_repo_to_bus,
)
from rl_swing.domain import (
    AuditEvent,
    BrokerOrder,
    CandidateTrade,
    EventType,
    FillEvent,
    MarketBar,
    OrderIntent,
    PolicyDecision,
    ReconciliationBreak,
    RiskDecision,
)
from rl_swing.runtime.event_bus import InMemoryEventBus


# --- synthetic provider ---------------------------------------------------
@pytest.mark.parametrize("regime", ["momentum", "mean_reversion", "random_walk"])
def test_synthetic_provider_each_regime(regime):
    p = SyntheticProvider(regime=regime, seed=11)
    bars = list(p.get_bars(["AAA", "BBB"], date(2020, 1, 1), date(2020, 6, 30)))
    assert len(bars) > 0
    assert all(b.symbol in {"AAA", "BBB"} for b in bars)
    assert all(b.high >= b.close >= 0 for b in bars)


def test_synthetic_provider_unknown_regime_raises():
    p = SyntheticProvider(regime="bogus", seed=1)
    with pytest.raises(ValueError):
        list(p.get_bars(["AAA"], date(2020, 1, 1), date(2020, 6, 30)))


def test_synthetic_provider_intraday_unsupported():
    p = SyntheticProvider(regime="momentum", seed=1)
    out = list(p.get_bars(["AAA"], date(2020, 1, 1), date(2020, 1, 5), timeframe="5m"))
    assert out == []


def test_synthetic_provider_empty_when_dates_have_no_weekday():
    p = SyntheticProvider(regime="momentum", seed=1)
    # Saturday only.
    out = list(p.get_bars(["AAA"], date(2020, 1, 4), date(2020, 1, 4)))
    assert out == []


def test_synthetic_provider_snapshot_id_stable():
    p = SyntheticProvider(regime="momentum", seed=11)
    a = p.get_snapshot_id(["A", "B"], date(2020, 1, 1), date(2020, 6, 30))
    b = p.get_snapshot_id(["B", "A"], date(2020, 1, 1), date(2020, 6, 30))
    assert a == b
    c = p.get_snapshot_id(["A"], date(2020, 1, 1), date(2020, 6, 30))
    assert a != c


# --- yfinance provider (no real network — exercise cache + fallbacks) ----
def test_yfinance_unsupported_timeframe_raises():
    p = YFinanceProvider(use_cache=False)
    with pytest.raises(NotImplementedError):
        list(p.get_bars(["AAPL"], date(2020, 1, 1), date(2020, 1, 5), timeframe="5m"))


def test_yfinance_returns_nothing_when_yfinance_unavailable(monkeypatch, tmp_path):
    p = YFinanceProvider(cache_dir=str(tmp_path), use_cache=False)
    # Simulate import failure inside _download by stubbing yf.download to raise.
    import sys
    fake = type(sys)("yfinance")
    def _fail(*a, **k): raise RuntimeError("nope")
    fake.download = _fail
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    bars = list(p.get_bars(["AAPL"], date(2020, 1, 1), date(2020, 1, 5)))
    assert bars == []


def test_yfinance_reads_cached_parquet(tmp_path):
    import pandas as pd
    p = YFinanceProvider(cache_dir=str(tmp_path), use_cache=True)
    cache = tmp_path / "FOO__2020-01-01__2020-01-05__adj=1.parquet"
    df = pd.DataFrame({
        "Open": [100.0, 101.0],
        "High": [102.0, 103.0],
        "Low": [99.0, 100.0],
        "Close": [101.0, 102.0],
        "Adj Close": [101.0, 102.0],
        "Volume": [1e6, 1.1e6],
    }, index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    df.to_parquet(cache)
    bars = list(p.get_bars(["FOO"], date(2020, 1, 1), date(2020, 1, 5)))
    assert len(bars) == 2
    assert bars[0].symbol == "FOO"


def test_yfinance_covering_cache_satisfies_shifted_request(tmp_path, monkeypatch):
    """FIX-#83: a cached file whose range fully covers the requested
    range should be re-used even if the filename doesn't exact-match
    the request. The classic failure was an off-by-one warmup_start
    triggering redundant yfinance fetches.

    We stub yfinance to raise on download so the test fails loudly if
    the cache lookup falls through to the network path.
    """
    import sys

    import pandas as pd
    p = YFinanceProvider(cache_dir=str(tmp_path), use_cache=True)
    # Cached: covers 2020-01-01..2020-01-10 (wider).
    cache = tmp_path / "FOO__2020-01-01__2020-01-10__adj=1.parquet"
    df = pd.DataFrame({
        "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "High": [102.0, 103.0, 104.0, 105.0, 106.0],
        "Low":  [ 99.0, 100.0, 101.0, 102.0, 103.0],
        "Close":[101.0, 102.0, 103.0, 104.0, 105.0],
        "Adj Close":[101.0, 102.0, 103.0, 104.0, 105.0],
        "Volume":[1e6]*5,
    }, index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06",
                              "2020-01-07", "2020-01-08"]))
    df.to_parquet(cache)
    # Stub yfinance.download to raise so any fall-through is loud.
    fake = type(sys)("yfinance")
    def _fail(*a, **k):
        raise AssertionError(
            "FIX-#83 regression: covering cache should have been used; "
            "fell through to network fetch instead."
        )
    fake.download = _fail
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    # Request a SHIFTED narrower range that the cached file covers.
    bars = list(p.get_bars(["FOO"], date(2020, 1, 2), date(2020, 1, 7)))
    # Should pick up the 4 bars in [2020-01-02, 2020-01-07] (Jan-2,3,6,7).
    assert len(bars) == 4
    assert bars[0].timestamp.date() == date(2020, 1, 2)
    assert bars[-1].timestamp.date() == date(2020, 1, 7)


def test_yfinance_covering_cache_picks_smallest(tmp_path, monkeypatch):
    """When multiple cached files cover the request, prefer the
    smallest one (least filtering work on read)."""
    import sys

    import pandas as pd
    p = YFinanceProvider(cache_dir=str(tmp_path), use_cache=True)
    # Wide covering: 2020-01-01..2020-01-30 (30 days).
    wide = tmp_path / "FOO__2020-01-01__2020-01-30__adj=1.parquet"
    pd.DataFrame({
        "Open":[100.0]*5, "High":[101.0]*5, "Low":[99.0]*5,
        "Close":[100.5]*5, "Adj Close":[100.5]*5, "Volume":[1e6]*5,
    }, index=pd.to_datetime(["2020-01-02","2020-01-03","2020-01-06","2020-01-07","2020-01-08"])).to_parquet(wide)
    # Narrow covering: 2020-01-01..2020-01-15 (14 days). Smaller; should be picked.
    narrow = tmp_path / "FOO__2020-01-01__2020-01-15__adj=1.parquet"
    pd.DataFrame({
        "Open":[200.0]*5, "High":[201.0]*5, "Low":[199.0]*5,
        "Close":[200.5]*5, "Adj Close":[200.5]*5, "Volume":[2e6]*5,
    }, index=pd.to_datetime(["2020-01-02","2020-01-03","2020-01-06","2020-01-07","2020-01-08"])).to_parquet(narrow)
    fake = type(sys)("yfinance")
    def _raise(*a, **k):
        raise AssertionError("network fetch should not occur")
    fake.download = _raise
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    bars = list(p.get_bars(["FOO"], date(2020, 1, 2), date(2020, 1, 7)))
    # Should have picked the narrow file (close prices ~200, not ~100).
    assert all(b.close > 150.0 for b in bars), "should have picked narrow covering file"


def test_yfinance_no_covering_cache_falls_through_to_fetch(tmp_path, monkeypatch):
    """If no cached file covers the requested range, fall through to
    the network fetch path."""
    import sys

    p = YFinanceProvider(cache_dir=str(tmp_path), use_cache=True)
    # Cached file that does NOT cover the request (wrong year).
    import pandas as pd
    cache = tmp_path / "FOO__2019-01-01__2019-12-31__adj=1.parquet"
    pd.DataFrame({
        "Open":[100.0],"High":[101.0],"Low":[99.0],"Close":[100.5],
        "Adj Close":[100.5],"Volume":[1e6],
    }, index=pd.to_datetime(["2019-06-15"])).to_parquet(cache)
    called = {"download": False}
    fake = type(sys)("yfinance")
    def _fake_download(*a, **k):
        called["download"] = True
        return None
    fake.download = _fake_download
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    list(p.get_bars(["FOO"], date(2020, 1, 1), date(2020, 12, 31)))
    assert called["download"] is True, "should have called yfinance.download"


def test_yfinance_snapshot_id_stable(tmp_path):
    p = YFinanceProvider(cache_dir=str(tmp_path))
    a = p.get_snapshot_id(["A", "B"], date(2020, 1, 1), date(2020, 6, 30))
    b = p.get_snapshot_id(["B", "A"], date(2020, 1, 1), date(2020, 6, 30))
    assert a == b


# --- WRDS parquet provider -----------------------------------------------
def test_wrds_provider_returns_nothing_when_cache_missing(tmp_path):
    p = WrdsParquetProvider(cache_dir=str(tmp_path / "nope"))
    bars = list(p.get_bars(["AAPL"], date(2020, 1, 1), date(2020, 6, 30)))
    assert bars == []


def test_wrds_provider_unsupported_timeframe_raises(tmp_path):
    p = WrdsParquetProvider(cache_dir=str(tmp_path))
    with pytest.raises(NotImplementedError):
        list(p.get_bars(["AAPL"], date(2020, 1, 1), date(2020, 1, 5), timeframe="5m"))


def test_wrds_provider_loads_minimal_fixture(tmp_path):
    import pandas as pd
    cache = tmp_path / "wrds"
    cache.mkdir()
    df = pd.DataFrame({
        "PERMNO": [10001, 10001, 10002],
        "TICKER": ["AAA", "AAA", "BBB"],
        "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-02"]),
        "PRC": [100.0, 101.0, 50.0],
        "VOL": [1e6, 2e6, 5e5],
        "CFACPR": [1.0, 1.0, 1.0],
    })
    df.to_parquet(cache / "crsp_dsf.parquet")

    # Reset class-level cache so the fixture is re-read.
    WrdsParquetProvider._df_cache = None

    p = WrdsParquetProvider(cache_dir=str(cache))
    bars = list(p.get_bars(["AAA", "BBB"], date(2020, 1, 1), date(2020, 6, 30)))
    assert any(b.symbol == "AAA" for b in bars)
    assert any(b.symbol == "BBB" for b in bars)


def test_wrds_provider_handles_negative_prc_flagged(tmp_path):
    import pandas as pd
    cache = tmp_path / "wrds"
    cache.mkdir()
    df = pd.DataFrame({
        "PERMNO": [10001],
        "TICKER": ["AAA"],
        "date": pd.to_datetime(["2020-01-02"]),
        "PRC": [-100.0],   # bid/ask average -> flagged
        "VOL": [1e6],
        "CFACPR": [1.0],
    })
    df.to_parquet(cache / "crsp_dsf.parquet")
    WrdsParquetProvider._df_cache = None

    p = WrdsParquetProvider(cache_dir=str(cache))
    bars = list(p.get_bars(["AAA"], date(2020, 1, 1), date(2020, 6, 30)))
    assert any("price_unreliable" in b.quality_flags for b in bars)


def test_wrds_provider_snapshot_id():
    p = WrdsParquetProvider(cache_dir="/no/such")
    s = p.get_snapshot_id(["A"], date(2020, 1, 1), date(2020, 1, 5))
    assert isinstance(s, str)


# --- parquet provider ----------------------------------------------------
def test_parquet_provider_round_trip(tmp_path):
    import pandas as pd
    p = ParquetProvider(cache_dir=str(tmp_path))
    df = pd.DataFrame({
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [1e6, 1.1e6],
        "adjusted_close": [101.0, 102.0],
    }, index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    df.to_parquet(tmp_path / "FOO.parquet")
    bars = list(p.get_bars(["FOO"], date(2020, 1, 1), date(2020, 1, 5)))
    assert len(bars) == 2


def test_parquet_provider_skips_missing_files(tmp_path):
    p = ParquetProvider(cache_dir=str(tmp_path))
    bars = list(p.get_bars(["NOPE"], date(2020, 1, 1), date(2020, 1, 5)))
    assert bars == []


def test_parquet_provider_unsupported_timeframe_raises(tmp_path):
    p = ParquetProvider(cache_dir=str(tmp_path))
    with pytest.raises(NotImplementedError):
        list(p.get_bars(["X"], date(2020, 1, 1), date(2020, 1, 5), timeframe="5m"))


def test_parquet_provider_snapshot_id():
    p = ParquetProvider(cache_dir="x")
    assert isinstance(p.get_snapshot_id(["A"], date(2020, 1, 1), date(2020, 6, 30)), str)


# --- shadow broker --------------------------------------------------------
def _intent(symbol="AAPL", **kw) -> OrderIntent:
    base = dict(
        intent_id="i1", as_of=datetime(2024, 1, 2),
        symbol=symbol, side="buy", quantity=10,
        order_type="market", time_in_force="day",
        limit_price=None, source_decision_id="d", environment="shadow",
        client_order_id="cid-1",
    )
    base.update(kw)
    return OrderIntent(**base)


def test_shadow_broker_accepts_orders():
    b = NoOpShadowBrokerAdapter()
    o = b.submit_order(_intent())
    assert o.status == "ACCEPTED"
    assert b.list_open_orders() == []
    assert b.list_positions() == []
    snap = b.get_account_snapshot()
    assert snap.cash == 0.0
    b.cancel_order("anything")  # no-op


# --- alpaca stubs ---------------------------------------------------------
@pytest.mark.parametrize("cls", [AlpacaPaperBrokerAdapter, AlpacaLiveBrokerAdapter])
def test_alpaca_adapters_raise_not_implemented(cls):
    a = cls()
    with pytest.raises(NotImplementedError):
        a.submit_order(_intent(environment="paper"))
    with pytest.raises(NotImplementedError):
        a.cancel_order("x")
    with pytest.raises(NotImplementedError):
        a.list_open_orders()
    with pytest.raises(NotImplementedError):
        a.list_positions()
    with pytest.raises(NotImplementedError):
        a.get_account_snapshot()


# --- simulated broker -----------------------------------------------------
def test_simulated_broker_rejects_without_oracle():
    b = SimulatedBrokerAdapter(starting_cash=10_000)
    o = b.submit_order(_intent(environment="backtest"))
    assert o.status == "REJECTED"


def test_simulated_broker_fills_buy_then_sells():
    b = SimulatedBrokerAdapter(starting_cash=10_000)
    b.set_price_oracle(lambda sym, ts: 100.0)
    o1 = b.submit_order(_intent(symbol="AAPL", side="buy", quantity=10))
    assert o1.status == "FILLED"
    assert b.fills
    positions = b.list_positions()
    assert len(positions) == 1 and positions[0].quantity == 10
    snap = b.get_account_snapshot()
    assert snap.cash < 10_000
    o2 = b.submit_order(_intent(symbol="AAPL", side="sell", quantity=5,
                                client_order_id="cid-2"))
    assert o2.status == "FILLED"
    pos = b.list_positions()
    assert pos and pos[0].quantity == 5


def test_simulated_broker_rejects_when_oracle_returns_none():
    b = SimulatedBrokerAdapter()
    b.set_price_oracle(lambda *a: None)
    o = b.submit_order(_intent())
    assert o.status == "REJECTED"


def test_simulated_broker_rejects_insufficient_cash():
    b = SimulatedBrokerAdapter(starting_cash=10)
    b.set_price_oracle(lambda *a: 100.0)
    o = b.submit_order(_intent(quantity=100))
    assert o.status == "REJECTED"


def test_simulated_broker_cancel_changes_status():
    b = SimulatedBrokerAdapter()
    b.set_price_oracle(lambda *a: 100.0)
    o = b.submit_order(_intent())
    # Already FILLED — cancel is a no-op for terminal statuses.
    b.cancel_order(o.broker_order_id)
    assert b.list_open_orders() == []


def test_simulated_broker_sell_unknown_position_warns_and_skips(caplog):
    b = SimulatedBrokerAdapter()
    b.set_price_oracle(lambda *a: 100.0)
    o = b.submit_order(_intent(side="sell", symbol="DOES_NOT_EXIST"))
    # Position unknown -> we just warn and don't crash.
    assert o.status == "FILLED"  # fill from oracle's perspective
    assert b.list_positions() == []


def test_simulated_broker_partial_close_keeps_position():
    b = SimulatedBrokerAdapter(starting_cash=100_000)
    b.set_price_oracle(lambda *a: 100.0)
    b.submit_order(_intent(symbol="X", side="buy", quantity=10))
    b.submit_order(_intent(symbol="X", side="sell", quantity=4, client_order_id="cid-2"))
    pos = b.list_positions()
    assert pos and pos[0].quantity == pytest.approx(6.0)


# --- sqlite storage -------------------------------------------------------
def test_sqlite_bundle_round_trip_audit(tmp_db):
    bus = InMemoryEventBus()
    attach_audit_repo_to_bus(tmp_db.audit, bus)
    ev = AuditEvent(
        event_id="e1", event_type=EventType.PIPELINE_STARTED,
        timestamp=datetime(2024, 1, 2),
        correlation_id="run1", payload={"a": 1},
        run_id="run1", environment="research", tags=("test",),
    )
    bus.publish(ev)
    replayed = tmp_db.audit.replay("run1")
    assert len(replayed) == 1
    assert replayed[0].payload == {"a": 1}
    assert replayed[0].event_type == EventType.PIPELINE_STARTED


def test_sqlite_bundle_round_trip_bars_and_candidates(tmp_db):
    bar = MarketBar(
        symbol="AAPL", timestamp=datetime(2024, 1, 2), timeframe="1d",
        open=100, high=101, low=99, close=100.5, volume=1e6,
        adjusted_close=100.5, source="test",
    )
    n = tmp_db.bars.save_bars([bar])
    assert n == 1
    out = tmp_db.bars.load_bars(["AAPL"], date(2024, 1, 1), date(2024, 1, 31))
    assert out and out[0].symbol == "AAPL"

    cand = CandidateTrade(
        candidate_id="c1", as_of=datetime(2024, 1, 2), symbol="AAPL",
        strategy_id="m", direction="long", entry_timing="next_open",
        base_size_pct=0.10, max_holding_days=10, stop_rule_id=None,
        exit_rule_id="x", signal_strength=0.5,
    )
    assert tmp_db.candidates.save_candidates([cand]) == 1
    loaded = tmp_db.candidates.load_candidates("any")
    assert loaded and loaded[0].candidate_id == "c1"


def test_sqlite_bars_empty_inputs_safe(tmp_db):
    assert tmp_db.bars.save_bars([]) == 0
    assert tmp_db.bars.load_bars([], date(2024, 1, 1), date(2024, 1, 31)) == []
    assert tmp_db.bars.load_bars(["X"], date(2024, 1, 1), date(2024, 1, 31), timeframe="5m") == []


def test_sqlite_decisions_round_trip(tmp_db):
    pd_ = PolicyDecision(
        decision_id="d1", candidate_id="c1", as_of=datetime(2024, 1, 2),
        model_id="m", action="take_50", confidence=0.7,
        target_size_pct=0.05, raw_action=2, observation_hash="h",
        explanation={"raw_action": 2},
    )
    rd = RiskDecision(
        decision_id="rd1", candidate_id="c1", policy_decision_id="d1",
        approved=True, final_size_pct=0.04, blocked_reasons=(),
        applied_rules=("max_single_position",),
    )
    assert tmp_db.decisions.save_policy_decisions([pd_]) == 1
    assert tmp_db.decisions.save_risk_decisions([rd]) == 1
    loaded = tmp_db.decisions.load_policy_decisions("any")
    assert loaded and loaded[0].decision_id == "d1"


def test_sqlite_orders_and_fills_round_trip(tmp_db):
    bo = BrokerOrder(
        internal_order_id="i1", broker_order_id="bo1", client_order_id="c1",
        environment="paper", symbol="AAPL", side="buy",
        order_type="market", time_in_force="day",
        requested_qty=10, limit_price=None, status="FILLED",
        submitted_at=datetime(2024, 1, 2), updated_at=datetime(2024, 1, 2),
    )
    assert tmp_db.orders.save_orders([bo]) == 1
    f = FillEvent(
        fill_id="f1", internal_order_id="i1", broker_order_id="bo1",
        symbol="AAPL", side="buy", filled_qty=10, filled_avg_price=100.0,
        filled_at=datetime(2024, 1, 2),
    )
    assert tmp_db.orders.save_fills([f]) == 1


def test_sqlite_reconciliation_round_trip(tmp_db):
    b = ReconciliationBreak(
        recon_id="r1", recon_at=datetime(2024, 1, 2),
        environment="paper", break_type="quantity_mismatch",
        severity="WARN", description="off by one",
    )
    assert tmp_db.reconciliation.save_breaks([b]) == 1


def test_sqlite_empty_inputs_for_all_repos(tmp_db):
    assert tmp_db.candidates.save_candidates([]) == 0
    assert tmp_db.decisions.save_policy_decisions([]) == 0
    assert tmp_db.decisions.save_risk_decisions([]) == 0
    assert tmp_db.orders.save_orders([]) == 0
    assert tmp_db.orders.save_fills([]) == 0
    assert tmp_db.reconciliation.save_breaks([]) == 0
    assert tmp_db.audit.append_events([]) == 0
