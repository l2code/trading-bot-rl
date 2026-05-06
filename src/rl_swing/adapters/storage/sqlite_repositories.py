"""SQLite-backed repositories.

The schema mirrors ``spec §16.2``. We use plain ``sqlite3`` (stdlib) so
the package has no driver dependency for development. A Postgres-based
adapter can be added under the same ports without touching the rest of
the system.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

from rl_swing.domain import (
    AuditEvent,
    BrokerOrder,
    CandidateTrade,
    EventType,
    FillEvent,
    MarketBar,
    PolicyDecision,
    ReconciliationBreak,
    RiskDecision,
)

_log = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS assets (
  symbol TEXT PRIMARY KEY,
  name TEXT,
  asset_type TEXT,
  exchange TEXT,
  sector TEXT,
  is_active INTEGER,
  is_tradeable INTEGER,
  first_seen_date TEXT,
  last_seen_date TEXT
);

CREATE TABLE IF NOT EXISTS bars_daily (
  symbol TEXT NOT NULL,
  bar_date TEXT NOT NULL,
  open REAL,
  high REAL,
  low REAL,
  close REAL,
  adjusted_close REAL,
  volume REAL,
  source TEXT NOT NULL,
  ingested_at TEXT,
  quality_status TEXT,
  PRIMARY KEY (symbol, bar_date, source)
);

CREATE TABLE IF NOT EXISTS features_daily (
  feature_id TEXT PRIMARY KEY,
  symbol TEXT NOT NULL,
  as_of_date TEXT NOT NULL,
  feature_config_version TEXT NOT NULL,
  features_json TEXT NOT NULL,
  source_snapshot_id TEXT,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS candidate_trades (
  candidate_id TEXT PRIMARY KEY,
  symbol TEXT NOT NULL,
  as_of_date TEXT NOT NULL,
  strategy_id TEXT NOT NULL,
  direction TEXT NOT NULL,
  signal_score REAL,
  base_size_pct REAL,
  max_holding_days INTEGER,
  metadata_json TEXT,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS rl_decisions (
  decision_id TEXT PRIMARY KEY,
  candidate_id TEXT NOT NULL,
  model_id TEXT NOT NULL,
  as_of_date TEXT NOT NULL,
  action TEXT NOT NULL,
  target_size_pct REAL,
  confidence REAL,
  observation_hash TEXT,
  decision_metadata_json TEXT,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS risk_decisions (
  risk_decision_id TEXT PRIMARY KEY,
  candidate_id TEXT NOT NULL,
  policy_decision_id TEXT NOT NULL,
  approved INTEGER NOT NULL,
  final_size_pct REAL,
  blocked_reasons_json TEXT,
  applied_rules_json TEXT,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS broker_orders (
  internal_order_id TEXT PRIMARY KEY,
  broker_order_id TEXT,
  client_order_id TEXT UNIQUE,
  environment TEXT,
  symbol TEXT,
  side TEXT,
  order_type TEXT,
  time_in_force TEXT,
  requested_qty REAL,
  limit_price REAL,
  status TEXT,
  submitted_at TEXT,
  updated_at TEXT,
  raw_request_json TEXT,
  raw_response_json TEXT
);

CREATE TABLE IF NOT EXISTS broker_fills (
  fill_id TEXT PRIMARY KEY,
  internal_order_id TEXT,
  broker_order_id TEXT,
  symbol TEXT,
  side TEXT,
  filled_qty REAL,
  filled_avg_price REAL,
  filled_at TEXT,
  raw_fill_json TEXT
);

CREATE TABLE IF NOT EXISTS reconciliation_events (
  recon_id TEXT PRIMARY KEY,
  recon_at TEXT NOT NULL,
  environment TEXT,
  break_type TEXT,
  severity TEXT,
  description TEXT,
  expected_json TEXT,
  actual_json TEXT,
  resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS audit_events (
  event_id TEXT PRIMARY KEY,
  event_type TEXT NOT NULL,
  ts TEXT NOT NULL,
  correlation_id TEXT,
  run_id TEXT,
  environment TEXT,
  schema_version TEXT,
  tags_json TEXT,
  payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_bars_symbol_date ON bars_daily(symbol, bar_date);
CREATE INDEX IF NOT EXISTS idx_features_symbol_date ON features_daily(symbol, as_of_date);
CREATE INDEX IF NOT EXISTS idx_candidates_run ON candidate_trades(strategy_id, as_of_date);
CREATE INDEX IF NOT EXISTS idx_audit_run ON audit_events(run_id);
"""


def _parse_url(url: str) -> str:
    """``sqlite:///foo.sqlite`` -> ``foo.sqlite``."""
    if url.startswith("sqlite:///"):
        return url[len("sqlite:///"):]
    if url.startswith("sqlite://"):
        return url[len("sqlite://"):]
    return url


def _to_iso(ts: datetime | date) -> str:
    if isinstance(ts, datetime):
        return ts.isoformat()
    return datetime(ts.year, ts.month, ts.day).isoformat()


def _from_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


class SqliteStorageBundle:
    """Single object holding all repositories so the container can pass
    one storage handle down to services."""

    def __init__(self, database_url: str = "sqlite:///data/cache/rl_swing.sqlite") -> None:
        self.path = _parse_url(database_url)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()
        # Repositories
        self.bars = SqliteBarRepository(self)
        self.candidates = SqliteCandidateRepository(self)
        self.decisions = SqliteDecisionRepository(self)
        self.orders = SqliteOrderRepository(self)
        self.reconciliation = SqliteReconciliationRepository(self)
        self.audit = SqliteAuditRepository(self)

    @contextmanager
    def connect(self):
        # Each call gets its own connection — sqlite3 is happy with that
        # and it sidesteps thread-affinity issues during tests.
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)


class _Base:
    def __init__(self, bundle: SqliteStorageBundle) -> None:
        self.bundle = bundle


class SqliteBarRepository(_Base):
    def save_bars(self, bars: Iterable[MarketBar]) -> int:
        rows = [
            (
                b.symbol, b.timestamp.date().isoformat(),
                b.open, b.high, b.low, b.close,
                b.adjusted_close, b.volume,
                b.source, datetime.utcnow().isoformat(),
                "PASS" if not b.quality_flags else "WARN",
            )
            for b in bars
        ]
        if not rows:
            return 0
        with self.bundle.connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO bars_daily
                   (symbol, bar_date, open, high, low, close, adjusted_close,
                    volume, source, ingested_at, quality_status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
        return len(rows)

    def load_bars(
        self,
        symbols: list[str],
        start: date,
        end: date,
        timeframe: str = "1d",
    ) -> list[MarketBar]:
        if timeframe != "1d":
            return []
        if not symbols:
            return []
        placeholders = ",".join("?" for _ in symbols)
        with self.bundle.connect() as conn:
            cur = conn.execute(
                f"""SELECT symbol, bar_date, open, high, low, close,
                          adjusted_close, volume, source, quality_status
                       FROM bars_daily
                       WHERE symbol IN ({placeholders})
                         AND bar_date BETWEEN ? AND ?
                       ORDER BY symbol, bar_date""",
                (*symbols, start.isoformat(), end.isoformat()),
            )
            rows = cur.fetchall()
        out: list[MarketBar] = []
        for r in rows:
            sym, d, o, h, lo, c, ac, v, src, qs = r
            flags: tuple[str, ...] = () if qs == "PASS" else (qs.lower(),)
            out.append(
                MarketBar(
                    symbol=sym, timestamp=datetime.fromisoformat(d),
                    timeframe=timeframe,
                    open=float(o), high=float(h), low=float(lo),
                    close=float(c),
                    adjusted_close=float(ac) if ac is not None else None,
                    volume=float(v), source=src,
                    quality_flags=flags,
                )
            )
        return out


class SqliteCandidateRepository(_Base):
    def save_candidates(self, candidates: Iterable[CandidateTrade]) -> int:
        rows = [
            (
                c.candidate_id, c.symbol,
                _to_iso(c.as_of), c.strategy_id, c.direction,
                c.signal_strength, c.base_size_pct, c.max_holding_days,
                json.dumps(c.metadata, default=str),
                datetime.utcnow().isoformat(),
            )
            for c in candidates
        ]
        if not rows:
            return 0
        with self.bundle.connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO candidate_trades
                   (candidate_id, symbol, as_of_date, strategy_id, direction,
                    signal_score, base_size_pct, max_holding_days,
                    metadata_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
        return len(rows)

    def load_candidates(self, run_id: str) -> list[CandidateTrade]:
        # ``run_id`` correlation tracking is on the audit table; this
        # repo is keyed by candidate_id. The pipeline service is
        # responsible for joining the two when needed.
        with self.bundle.connect() as conn:
            cur = conn.execute(
                """SELECT candidate_id, symbol, as_of_date, strategy_id, direction,
                          signal_score, base_size_pct, max_holding_days, metadata_json
                       FROM candidate_trades"""
            )
            return [
                CandidateTrade(
                    candidate_id=r[0],
                    as_of=_from_iso(r[2]),
                    symbol=r[1],
                    strategy_id=r[3],
                    direction=r[4],
                    entry_timing="next_open",
                    base_size_pct=float(r[6]),
                    max_holding_days=int(r[7]),
                    stop_rule_id=None,
                    exit_rule_id="default_exit",
                    signal_strength=float(r[5]),
                    metadata=json.loads(r[8]) if r[8] else {},
                )
                for r in cur.fetchall()
            ]


class SqliteDecisionRepository(_Base):
    def save_policy_decisions(self, decisions: Iterable[PolicyDecision]) -> int:
        rows = [
            (
                d.decision_id, d.candidate_id, d.model_id, _to_iso(d.as_of),
                d.action, d.target_size_pct, d.confidence,
                d.observation_hash, json.dumps(d.explanation, default=str),
                datetime.utcnow().isoformat(),
            )
            for d in decisions
        ]
        if not rows:
            return 0
        with self.bundle.connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO rl_decisions
                   (decision_id, candidate_id, model_id, as_of_date, action,
                    target_size_pct, confidence, observation_hash,
                    decision_metadata_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
        return len(rows)

    def save_risk_decisions(self, decisions: Iterable[RiskDecision]) -> int:
        rows = [
            (
                d.decision_id, d.candidate_id, d.policy_decision_id,
                int(d.approved), d.final_size_pct,
                json.dumps(list(d.blocked_reasons)),
                json.dumps(list(d.applied_rules)),
                datetime.utcnow().isoformat(),
            )
            for d in decisions
        ]
        if not rows:
            return 0
        with self.bundle.connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO risk_decisions
                   (risk_decision_id, candidate_id, policy_decision_id,
                    approved, final_size_pct, blocked_reasons_json,
                    applied_rules_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                rows,
            )
        return len(rows)

    def load_policy_decisions(self, run_id: str) -> list[PolicyDecision]:
        with self.bundle.connect() as conn:
            cur = conn.execute(
                """SELECT decision_id, candidate_id, model_id, as_of_date, action,
                          target_size_pct, confidence, observation_hash,
                          decision_metadata_json
                       FROM rl_decisions"""
            )
            out = []
            for r in cur.fetchall():
                expl = json.loads(r[8]) if r[8] else {}
                out.append(
                    PolicyDecision(
                        decision_id=r[0], candidate_id=r[1],
                        as_of=_from_iso(r[3]), model_id=r[2],
                        action=r[4],
                        confidence=float(r[6]) if r[6] is not None else None,
                        target_size_pct=float(r[5]),
                        raw_action=expl.get("raw_action", -1),
                        observation_hash=r[7] or "",
                        explanation=expl,
                    )
                )
            return out


class SqliteOrderRepository(_Base):
    def save_orders(self, orders: Iterable[BrokerOrder]) -> int:
        rows = [
            (
                o.internal_order_id, o.broker_order_id, o.client_order_id,
                o.environment, o.symbol, o.side, o.order_type, o.time_in_force,
                o.requested_qty, o.limit_price, o.status,
                _to_iso(o.submitted_at), _to_iso(o.updated_at),
                json.dumps(o.raw_request, default=str),
                json.dumps(o.raw_response, default=str),
            )
            for o in orders
        ]
        if not rows:
            return 0
        with self.bundle.connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO broker_orders
                   (internal_order_id, broker_order_id, client_order_id,
                    environment, symbol, side, order_type, time_in_force,
                    requested_qty, limit_price, status,
                    submitted_at, updated_at,
                    raw_request_json, raw_response_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
        return len(rows)

    def save_fills(self, fills: Iterable[FillEvent]) -> int:
        rows = [
            (
                f.fill_id, f.internal_order_id, f.broker_order_id,
                f.symbol, f.side, f.filled_qty, f.filled_avg_price,
                _to_iso(f.filled_at), json.dumps(f.raw_fill, default=str),
            )
            for f in fills
        ]
        if not rows:
            return 0
        with self.bundle.connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO broker_fills
                   (fill_id, internal_order_id, broker_order_id, symbol, side,
                    filled_qty, filled_avg_price, filled_at, raw_fill_json)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                rows,
            )
        return len(rows)


class SqliteReconciliationRepository(_Base):
    def save_breaks(self, breaks: Iterable[ReconciliationBreak]) -> int:
        rows = [
            (
                b.recon_id, _to_iso(b.recon_at), b.environment,
                b.break_type, b.severity, b.description,
                json.dumps(b.expected, default=str),
                json.dumps(b.actual, default=str),
                _to_iso(b.resolved_at) if b.resolved_at else None,
            )
            for b in breaks
        ]
        if not rows:
            return 0
        with self.bundle.connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO reconciliation_events
                   (recon_id, recon_at, environment, break_type, severity,
                    description, expected_json, actual_json, resolved_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                rows,
            )
        return len(rows)


class SqliteAuditRepository(_Base):
    def append_events(self, events: Iterable[AuditEvent]) -> int:
        rows = [
            (
                e.event_id, e.event_type.value, _to_iso(e.timestamp),
                e.correlation_id, e.run_id, e.environment, e.schema_version,
                json.dumps(list(e.tags)),
                json.dumps(e.payload, default=str),
            )
            for e in events
        ]
        if not rows:
            return 0
        with self.bundle.connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO audit_events
                   (event_id, event_type, ts, correlation_id, run_id, environment,
                    schema_version, tags_json, payload_json)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                rows,
            )
        return len(rows)

    def replay(self, run_id: str) -> list[AuditEvent]:
        with self.bundle.connect() as conn:
            cur = conn.execute(
                """SELECT event_id, event_type, ts, correlation_id, run_id,
                          environment, schema_version, tags_json, payload_json
                       FROM audit_events WHERE run_id = ? ORDER BY ts""",
                (run_id,),
            )
            return [
                AuditEvent(
                    event_id=r[0],
                    event_type=EventType(r[1]),
                    timestamp=_from_iso(r[2]),
                    correlation_id=r[3] or "",
                    run_id=r[4] or "",
                    environment=r[5] or "",
                    schema_version=r[6] or "v1",
                    tags=tuple(json.loads(r[7])) if r[7] else (),
                    payload=json.loads(r[8]) if r[8] else {},
                )
                for r in cur.fetchall()
            ]


def attach_audit_repo_to_bus(audit_repo: SqliteAuditRepository, bus) -> None:
    """Subscribe a listener that persists every published event."""
    bus.subscribe(lambda ev: audit_repo.append_events([ev]))
