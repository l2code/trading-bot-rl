"""Component registry — resolves abstract names to concrete classes.

Usage:

    registry = ComponentRegistry.from_yaml("configs/components/components.yaml")
    provider = registry.build("market_data_providers", "yfinance_daily")

The registry is the single place where dotted-path strings turn into
imported classes. Service code never imports adapters directly.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    name: str
    cls_path: str           # e.g. "rl_swing.adapters.data.yfinance_provider.YFinanceProvider"
    params: dict[str, Any]

    def build(self, **overrides: Any) -> Any:
        module_path, _, class_name = self.cls_path.rpartition(".")
        if not module_path:
            raise ValueError(f"Bad class path: {self.cls_path}")
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        kwargs = {**self.params, **overrides}
        return cls(**kwargs)


class ComponentRegistry:
    """A flat ``{category: {name: ComponentSpec}}`` registry."""

    def __init__(self, components: dict[str, dict[str, ComponentSpec]]) -> None:
        self._components = components

    # -- construction ---------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> ComponentRegistry:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComponentRegistry:
        components: dict[str, dict[str, ComponentSpec]] = {}
        for category, entries in (data.get("components") or {}).items():
            components[category] = {}
            for name, spec in (entries or {}).items():
                cls_path = spec.get("class")
                if not cls_path:
                    raise ValueError(f"{category}.{name} missing 'class'")
                params = spec.get("params") or {}
                components[category][name] = ComponentSpec(
                    name=name, cls_path=cls_path, params=params
                )
        return cls(components)

    # -- lookups --------------------------------------------------------
    def categories(self) -> list[str]:
        return sorted(self._components.keys())

    def names(self, category: str) -> list[str]:
        return sorted(self._components.get(category, {}).keys())

    def get_spec(self, category: str, name: str) -> ComponentSpec:
        try:
            return self._components[category][name]
        except KeyError as e:
            available = self.names(category) if category in self._components else []
            raise KeyError(
                f"Component {category}.{name!r} not registered. "
                f"Known {category}: {available}"
            ) from e

    def build(self, category: str, name: str, **overrides: Any) -> Any:
        return self.get_spec(category, name).build(**overrides)
