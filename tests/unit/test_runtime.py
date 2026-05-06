"""runtime/* — registry, modes, dependency container, event bus, CLI."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
import yaml

from rl_swing.domain import AuditEvent, EventType
from rl_swing.runtime.dependency_container import (
    RuntimeConfig,
    _safety_check,
    build_container,
)
from rl_swing.runtime.event_bus import InMemoryEventBus, make_audit_logger
from rl_swing.runtime.modes import MODE_PROFILES, get_mode_profile
from rl_swing.runtime.registry import ComponentRegistry, ComponentSpec


# --- modes ----------------------------------------------------------------
def test_mode_profiles_known():
    for m in ("research", "shadow", "paper", "live_guarded"):
        p = get_mode_profile(m)
        assert p.mode == m
    assert "research" in MODE_PROFILES


def test_mode_profile_unknown_raises():
    with pytest.raises(ValueError):
        get_mode_profile("not_a_mode")  # type: ignore[arg-type]


def test_mode_profile_paper_places_orders_and_research_does_not():
    assert MODE_PROFILES["paper"].place_orders is True
    assert MODE_PROFILES["research"].place_orders is False
    assert MODE_PROFILES["live_guarded"].place_orders is False


# --- registry -------------------------------------------------------------
def test_component_registry_round_trip(tmp_path: Path):
    cfg = {
        "components": {
            "policy_scorers": {
                "random": {
                    "class": "rl_swing.rl.agents.baseline_scorers.RandomPolicyScorer",
                    "params": {"model_id": "x", "seed": 7},
                }
            }
        }
    }
    p = tmp_path / "components.yaml"
    p.write_text(yaml.safe_dump(cfg))
    reg = ComponentRegistry.from_yaml(p)
    assert "policy_scorers" in reg.categories()
    assert reg.names("policy_scorers") == ["random"]
    spec = reg.get_spec("policy_scorers", "random")
    assert isinstance(spec, ComponentSpec)
    inst = reg.build("policy_scorers", "random")
    assert inst.model_id == "x"


def test_component_registry_missing_class_raises():
    with pytest.raises(ValueError):
        ComponentRegistry.from_dict({"components": {"x": {"y": {}}}})


def test_component_registry_missing_dotted_path_raises(tmp_path: Path):
    cfg = {"components": {"x": {"y": {"class": "nopath", "params": {}}}}}
    reg = ComponentRegistry.from_dict(cfg)
    with pytest.raises(ValueError):
        reg.build("x", "y")


def test_component_registry_unknown_lookup_raises():
    reg = ComponentRegistry({})
    with pytest.raises(KeyError):
        reg.get_spec("nope", "missing")


# --- event bus ------------------------------------------------------------
def test_event_bus_publish_to_subscribed_listeners():
    bus = InMemoryEventBus()
    received: list[AuditEvent] = []
    listener = lambda ev: received.append(ev)  # noqa: E731
    bus.subscribe(listener)
    bus.subscribe(listener)  # idempotent
    ev = AuditEvent(
        event_id="e1", event_type=EventType.PIPELINE_STARTED,
        timestamp=datetime.utcnow(), correlation_id="c", payload={},
        run_id="r", environment="research",
    )
    bus.publish(ev)
    assert received == [ev]


def test_event_bus_unsubscribe():
    bus = InMemoryEventBus()
    seen: list[AuditEvent] = []
    listener = lambda ev: seen.append(ev)  # noqa: E731
    bus.subscribe(listener)
    bus.unsubscribe(listener)
    bus.unsubscribe(listener)  # idempotent
    bus.publish(AuditEvent(
        event_id="x", event_type=EventType.PIPELINE_COMPLETED,
        timestamp=datetime.utcnow(), correlation_id="c", payload={},
        run_id="r", environment="research",
    ))
    assert seen == []


def test_event_bus_swallows_listener_exceptions():
    bus = InMemoryEventBus()

    def bad(_):
        raise RuntimeError("boom")

    received = []
    bus.subscribe(bad)
    bus.subscribe(received.append)
    ev = AuditEvent(
        event_id="x", event_type=EventType.PIPELINE_STARTED,
        timestamp=datetime.utcnow(), correlation_id="c", payload={},
        run_id="r", environment="research",
    )
    bus.publish(ev)  # must not raise
    assert received == [ev]


def test_audit_logger_listener_runs():
    bus = InMemoryEventBus()
    bus.subscribe(make_audit_logger())
    bus.publish(AuditEvent(
        event_id="x", event_type=EventType.PIPELINE_STARTED,
        timestamp=datetime.utcnow(), correlation_id="c", payload={"a": 1},
        run_id="r", environment="research",
    ))


# --- runtime config + container ------------------------------------------
def _runtime_yaml(tmp_path: Path, **overrides) -> Path:
    cfg = {
        "runtime": {
            "mode": "research",
            "universe": "synthetic",
            "data_provider": "synthetic_momentum",
            "feature_pipeline": "equities_features_v001",
            "strategies": ["momentum_20_60"],
            "policy_scorer": "random",
            "broker_adapter": "shadow",
            "storage_profile": "local_sqlite",
            **overrides,
        }
    }
    p = tmp_path / "runtime.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_runtime_config_from_yaml_round_trip(tmp_path: Path):
    p = _runtime_yaml(tmp_path)
    cfg = RuntimeConfig.from_yaml(p)
    assert cfg.mode == "research"
    assert cfg.data_provider == "synthetic_momentum"
    assert "momentum_20_60" in cfg.strategies


def test_runtime_config_extra_kwargs_kept(tmp_path: Path):
    p = _runtime_yaml(tmp_path, custom_key="abc")
    cfg = RuntimeConfig.from_yaml(p)
    assert cfg.extra.get("custom_key") == "abc"


def test_safety_check_blocks_live_with_no_token(tmp_path, monkeypatch):
    monkeypatch.delenv("RL_SWING_LIVE_APPROVAL_TOKEN", raising=False)
    cfg = RuntimeConfig(
        mode="live_guarded", universe="u", data_provider="x",
        feature_pipeline="y", strategies=[], policy_scorer="z",
        risk_profile="r", broker_adapter="alpaca_live",
        storage_profile="s", place_orders=True, allow_live_trading=False,
    )
    profile = get_mode_profile("live_guarded")
    with pytest.raises(RuntimeError):
        _safety_check(profile, cfg)


def test_safety_check_blocks_live_with_non_live_broker(monkeypatch):
    monkeypatch.setenv("RL_SWING_LIVE_APPROVAL_TOKEN", "yes")
    cfg = RuntimeConfig(
        mode="live_guarded", universe="u", data_provider="x",
        feature_pipeline="y", strategies=[], policy_scorer="z",
        risk_profile="r", broker_adapter="alpaca_paper",
        storage_profile="s", place_orders=True, allow_live_trading=True,
    )
    profile = get_mode_profile("live_guarded")
    with pytest.raises(RuntimeError):
        _safety_check(profile, cfg)


def test_safety_check_blocks_research_mode_demanding_live():
    cfg = RuntimeConfig(
        mode="research", universe="u", data_provider="x",
        feature_pipeline="y", strategies=[], policy_scorer="z",
        risk_profile="r", broker_adapter="simulated",
        storage_profile="s", allow_live_trading=True,
    )
    profile = get_mode_profile("research")
    with pytest.raises(RuntimeError):
        _safety_check(profile, cfg)


def test_safety_check_passes_for_research():
    cfg = RuntimeConfig(
        mode="research", universe="u", data_provider="x",
        feature_pipeline="y", strategies=[], policy_scorer="z",
        risk_profile="r", broker_adapter="simulated", storage_profile="s",
    )
    _safety_check(get_mode_profile("research"), cfg)  # should not raise


def test_build_container_research_mode(tmp_path: Path):
    components = {
        "components": {
            "market_data_providers": {
                "synthetic_momentum": {
                    "class": "rl_swing.adapters.data.synthetic_provider.SyntheticProvider",
                    "params": {"regime": "momentum", "seed": 11},
                }
            },
            "feature_pipelines": {
                "equities_features_v001": {
                    "class": "rl_swing.features.pipelines.CoreDailyPipeline",
                    "params": {"feature_version": "features_v001_core_daily"},
                }
            },
            "strategies": {
                "momentum_20_60": {
                    "class": "rl_swing.strategies.momentum.MomentumStrategy",
                    "params": {"strategy_id": "momentum_20_60"},
                }
            },
            "policy_scorers": {
                "random": {
                    "class": "rl_swing.rl.agents.baseline_scorers.RandomPolicyScorer",
                    "params": {"model_id": "rand"},
                }
            },
            "broker_adapters": {
                "shadow": {
                    "class": "rl_swing.adapters.broker.noop_shadow_broker.NoOpShadowBrokerAdapter",
                    "params": {},
                }
            },
        }
    }
    cp = tmp_path / "components.yaml"
    cp.write_text(yaml.safe_dump(components))
    rp = _runtime_yaml(tmp_path, run_id_prefix="research_daily")
    container = build_container(rp, cp)
    assert container.config.mode == "research"
    assert container.broker.broker_id == "shadow_noop"
    assert container.run_id.startswith("research_daily_")
