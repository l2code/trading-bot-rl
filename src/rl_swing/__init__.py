"""rl_swing — RL-assisted equity swing trading bot.

This package follows a ports-and-adapters architecture (see
``docs/adr/0001-use-ports-and-adapters.md``). Top-level layout:

* ``rl_swing.domain``   — frozen dataclasses for handoffs (MarketBar,
                            FeatureFrame, CandidateTrade, PolicyDecision,
                            RiskDecision, OrderIntent, etc.).
* ``rl_swing.ports``    — ``typing.Protocol`` interfaces (data providers,
                            strategies, scorers, risk policies, brokers,
                            repositories).
* ``rl_swing.adapters`` — concrete implementations (yfinance, WRDS,
                            Alpaca paper, SQLite, etc.).
* ``rl_swing.features`` — deterministic feature pipelines.
* ``rl_swing.strategies`` — rule-based candidate generators.
* ``rl_swing.rl``       — Gymnasium environment + RL training/validation.
* ``rl_swing.services`` — orchestration of the decision pipeline steps.
* ``rl_swing.runtime``  — CLI entry point + dependency container.
* ``rl_swing.risk``     — risk-policy stack + kill switch.
* ``rl_swing.reporting``— daily / training / validation reports.

The decision pipeline is identical across runtime modes (research /
shadow / paper / live_guarded). Only the broker adapter and active risk
profile change.
"""

__version__ = "0.1.0"
