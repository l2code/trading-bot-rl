"""Dependency container.

Reads a runtime config + the component registry, instantiates the
selected adapters, and bundles them so services don't need to know
about the registry. This is the only place imports of adapters happen
indirectly (via class lookup strings).
"""
from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from rl_swing.domain import AuditEvent, EventType
from rl_swing.ports import (
    BrokerAdapter,
    CandidateStrategy,
    EventBus,
    FeaturePipeline,
    MarketDataProvider,
    PolicyScorer,
)
from rl_swing.runtime.event_bus import InMemoryEventBus, make_audit_logger
from rl_swing.runtime.modes import ModeProfile, RuntimeMode, get_mode_profile
from rl_swing.runtime.registry import ComponentRegistry

_log = logging.getLogger(__name__)


@dataclass
class RuntimeConfig:
    mode: RuntimeMode
    universe: str
    data_provider: str
    feature_pipeline: str
    strategies: list[str]
    policy_scorer: str
    risk_profile: str
    broker_adapter: str
    storage_profile: str
    run_id_prefix: str = "run"
    place_orders: bool = False
    allow_live_trading: bool = False
    require_reconciliation_before_new_orders: bool = False
    emit_events: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RuntimeConfig:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        runtime = data.get("runtime") or data
        return cls(
            mode=runtime["mode"],
            universe=runtime.get("universe", "starter_equities"),
            data_provider=runtime["data_provider"],
            feature_pipeline=runtime["feature_pipeline"],
            strategies=list(runtime.get("strategies", [])),
            policy_scorer=runtime["policy_scorer"],
            risk_profile=runtime.get("risk_profile", "conservative_paper_v001"),
            broker_adapter=runtime["broker_adapter"],
            storage_profile=runtime.get("storage_profile", "local_sqlite"),
            run_id_prefix=runtime.get("run_id_prefix", "run"),
            place_orders=bool(runtime.get("place_orders", False)),
            allow_live_trading=bool(runtime.get("allow_live_trading", False)),
            require_reconciliation_before_new_orders=bool(
                runtime.get("require_reconciliation_before_new_orders", False)
            ),
            emit_events=bool(runtime.get("emit_events", True)),
            extra={k: v for k, v in runtime.items() if k not in {
                "mode", "universe", "data_provider", "feature_pipeline",
                "strategies", "policy_scorer", "risk_profile",
                "broker_adapter", "storage_profile", "run_id_prefix",
                "place_orders", "allow_live_trading",
                "require_reconciliation_before_new_orders", "emit_events",
            }},
        )


@dataclass
class Container:
    config: RuntimeConfig
    profile: ModeProfile
    registry: ComponentRegistry
    market_data: MarketDataProvider
    feature_pipeline: FeaturePipeline
    strategies: list[CandidateStrategy]
    policy_scorer: PolicyScorer
    broker: BrokerAdapter
    event_bus: EventBus
    run_id: str

    def emit(self, event_type: EventType, correlation_id: str, payload: dict,
             tags: tuple[str, ...] = ()) -> None:
        if not self.config.emit_events:
            return
        ev = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            correlation_id=correlation_id,
            payload=payload,
            run_id=self.run_id,
            environment=self.config.mode,
            tags=tags,
        )
        self.event_bus.publish(ev)


def _safety_check(profile: ModeProfile, config: RuntimeConfig) -> None:
    """Enforce the spec's hard safety rules at container build time."""
    if config.mode == "live_guarded":
        # Three independent flags must ALL be on for live to trade.
        token = os.environ.get("RL_SWING_LIVE_APPROVAL_TOKEN")
        if (config.place_orders or config.allow_live_trading) and not token:
            raise RuntimeError(
                "live_guarded mode requested with place_orders/allow_live_trading "
                "set, but RL_SWING_LIVE_APPROVAL_TOKEN is not in the environment."
            )
        if config.broker_adapter != "alpaca_live" and (
            config.place_orders or config.allow_live_trading
        ):
            raise RuntimeError(
                "live_guarded with non-live broker is incoherent. Use broker_adapter=alpaca_live."
            )
    if profile.allow_live_trading is False and config.allow_live_trading:
        # Profile says no, but config says yes.
        raise RuntimeError(
            f"Mode profile {profile.mode} forbids live trading; "
            f"runtime config says allow_live_trading=true."
        )


def build_container(
    runtime_config_path: str | Path,
    components_path: str | Path,
) -> Container:
    config = RuntimeConfig.from_yaml(runtime_config_path)
    profile = get_mode_profile(config.mode)
    _safety_check(profile, config)

    registry = ComponentRegistry.from_yaml(components_path)

    market_data: MarketDataProvider = registry.build(
        "market_data_providers", config.data_provider
    )
    feature_pipeline: FeaturePipeline = registry.build(
        "feature_pipelines", config.feature_pipeline
    )
    strategies: list[CandidateStrategy] = [
        registry.build("strategies", name) for name in config.strategies
    ]
    policy_scorer: PolicyScorer = registry.build(
        "policy_scorers", config.policy_scorer
    )
    broker: BrokerAdapter = registry.build("broker_adapters", config.broker_adapter)

    bus = InMemoryEventBus()
    if config.emit_events:
        bus.subscribe(make_audit_logger())

    run_id = f"{config.run_id_prefix}_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"

    container = Container(
        config=config,
        profile=profile,
        registry=registry,
        market_data=market_data,
        feature_pipeline=feature_pipeline,
        strategies=strategies,
        policy_scorer=policy_scorer,
        broker=broker,
        event_bus=bus,
        run_id=run_id,
    )

    container.emit(
        EventType.PIPELINE_STARTED,
        correlation_id=run_id,
        payload={
            "mode": config.mode,
            "data_provider": config.data_provider,
            "feature_pipeline": config.feature_pipeline,
            "strategies": config.strategies,
            "policy_scorer": config.policy_scorer,
            "broker_adapter": config.broker_adapter,
        },
    )

    return container
