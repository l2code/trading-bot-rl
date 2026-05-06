# rl-swing — RL-assisted equity swing trading bot

Reinforcement-learning-assisted equity swing trading research stack.
Rule-based strategies generate candidate trades; an RL `PolicyScorer`
chooses **skip / take_25 / take_50 / take_100**; a composable risk
engine has the final say. The same decision pipeline runs in
**research**, **shadow**, **paper**, and **live_guarded** modes — only
the broker adapter and the active risk profile change.

This is a research-grade implementation of Phases 0A through 5 of
the spec under `docs/spec/rl_swing_trading_bot_spec.md`. Phases 6–10
(WRDS upgrade, shadow-mode runner, Alpaca paper, production hardening,
tiny live experiment) are scaffolded as ports/stubs and ADRs but are
not wired to a real broker yet.

> **This is not financial advice.** The RL policy must not be connected
> to live capital without paper-trading evidence, strict risk limits,
> reconciliation, and manual model promotion. See ADRs 0002 and 0005.

## Quick start

```bash
git clone https://github.com/<USER>/trading-bot-rl.git
cd trading-bot-rl
pip install -e .[dev]

# List the components the registry knows about
rl-swing list-components

# Run one research-mode pipeline cycle on synthetic data
rl-swing run daily --mode research --config configs/runtime/research.yaml

# Smoke-test PPO training (a few thousand steps on synthetic data)
rl-swing train --experiment configs/experiments/ppo_filter_smoke.yaml \
  --total-timesteps 16000

# Walk-forward validate the trained model + baselines
rl-swing validate --experiment configs/experiments/ppo_filter_smoke.yaml \
  --report-dir data/reports
```

The training run writes a model artifact to
`data/models/<experiment>/model.zip` and a per-seed metadata file under
`data/models/<experiment>/seed_<N>/metadata.json`. The validation
command produces `data/reports/walkforward_<experiment>_<dates>.json`.

## Architecture

Ports-and-adapters (see `docs/adr/0001-use-ports-and-adapters.md`):

```
rl_swing/
  domain/        # frozen dataclasses for handoffs (MarketBar, FeatureFrame, ...)
  ports/         # typing.Protocol interfaces for replaceable components
  adapters/      # concrete implementations (yfinance, WRDS, Alpaca, sqlite, ...)
  features/      # deterministic feature pipelines + leakage checks
  strategies/    # rule-based candidate generators
  rl/            # Gymnasium env, agents, training, validation
  risk/          # composable RiskPolicy stack + risk engine
  services/      # decision pipeline orchestration
  runtime/       # CLI, dependency container, component registry, modes
```

Key contract: research, shadow, paper, and live_guarded modes use the
**same** 12-step decision pipeline. The broker adapter, data provider,
and risk profile are config swaps, not code changes.

## Configuration

All wiring lives under `configs/`:

- `configs/components/components.yaml` — registry mapping abstract names
  (`yfinance_daily`, `ppo_filter_v001`, `alpaca_paper`, …) to dotted
  Python classes plus default params.
- `configs/runtime/{research,shadow,paper,live_guarded}.yaml` — runtime
  profiles: which provider, which strategies, which scorer, which broker.
- `configs/experiments/*.yaml` — RL training experiment definitions.
- `configs/risk_profiles/*.yaml` — composable risk-policy stacks.
- `configs/universes/*.yaml` — symbol lists.

## Data sources

| Provider             | Adapter                             | Use                       |
|----------------------|-------------------------------------|---------------------------|
| Synthetic (in-tree)  | `SyntheticProvider`                 | Sanity tests, CI          |
| yfinance             | `YFinanceProvider`                  | Prototyping, Colab        |
| WRDS parquet cache   | `WrdsParquetProvider`               | Research-grade backtests  |
| Alpaca historical    | (Phase 8 stub)                      | Production alignment      |
| Parquet fixtures     | `ParquetProvider`                   | Test fixtures, replay     |

The WRDS adapter reads the parquet cache that `trading-bot2`'s
`scripts/research/wrds_refresh.py` writes — this build does not import
the live `wrds` Python package.

## Training in Google Colab

For runs beyond a smoke test (the spec's recommended 500k–2M steps × 3–5
seeds), use Colab. See `docs/colab_training.md` for the walkthrough.
The notebook entry point is `notebooks/04_colab_training.ipynb` and the
public-facing function is
`rl_swing.rl.training.colab_entrypoint.train(...)`.

## Tests + coverage

```bash
pytest                                          # all 219 tests
pytest --cov=rl_swing --cov-config=.coveragerc  # coverage report
```

CI gate: 85% line coverage on `rl_swing/`. Excluded from the gate:
the Alpaca paper/live broker stubs (Phase 8), the Postgres repository
(future), and the unit-yet-to-be-implemented service layer files.

## Repository layout

```
configs/        # YAML configs (no code)
docs/           # ADRs + ops docs
notebooks/      # Colab + exploration notebooks
scripts/        # Operator scripts (data refresh, etc)
src/rl_swing/   # Package source
tests/          # Unit, integration, contract tests
data/           # Local cache / models / reports (gitignored)
```

## Status

| Phase | Spec section                           | This build           |
|-------|----------------------------------------|----------------------|
| 0A    | Architecture contract foundation       | ✅ Complete          |
| 0     | Project foundation                     | ✅ Complete          |
| 1     | Data pipeline MVP                      | ✅ Complete          |
| 2     | Baseline strategy layer                | ✅ Complete          |
| 2A    | Equity feature intelligence upgrade    | Roadmap (features_v002) |
| 3     | RL environment MVP                     | ✅ Complete          |
| 4     | Initial RL training                    | ✅ Complete          |
| 5     | Walk-forward validation                | ✅ Complete          |
| 5A    | Crisis / cost / robustness validation  | Cost-stress wired; crisis-window roster TBD |
| 6     | WRDS research upgrade                  | Read-side adapter ✅; live `wrds` pulls TBD |
| 7     | Shadow mode                            | NoOp broker ✅; daily runner TBD |
| 8     | Alpaca paper trading                   | Adapter stubs only   |
| 9     | Production hardening                   | TBD                  |
| 10    | Tiny live experiment                   | TBD                  |

## License

MIT. See `LICENSE`.
