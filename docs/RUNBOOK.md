# Runbook

Day-to-day operator commands. Read top-to-bottom on first use; thereafter
use the table of contents to jump.

## Bootstrapping

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pytest                  # 219 tests, ~12s
```

## Routine commands

| Task                                | Command                                                                              |
|-------------------------------------|--------------------------------------------------------------------------------------|
| List registered components          | `rl-swing list-components`                                                           |
| Smoke-test the research pipeline    | `rl-swing run daily --mode research --config configs/runtime/research.yaml`          |
| Train PPO smoke (synthetic data)    | `rl-swing train --experiment configs/experiments/ppo_filter_smoke.yaml`              |
| Train PPO full                       | `rl-swing train --experiment configs/experiments/ppo_filter_v001.yaml`              |
| Train DQN full                       | `rl-swing train --experiment configs/experiments/dqn_filter_v001.yaml`              |
| Walk-forward validate               | `rl-swing validate --experiment configs/experiments/ppo_filter_v001.yaml`           |
| Run tests                           | `pytest`                                                                             |
| Coverage                            | `pytest --cov=rl_swing --cov-config=.coveragerc`                                     |

## Adding a new component

The registry is the single source of truth. To add a new strategy,
data provider, broker, scorer, or risk rule:

1. Implement the class. It must satisfy the corresponding `Protocol`
   in `src/rl_swing/ports/`.
2. Add an entry under the appropriate category in
   `configs/components/components.yaml`:
   ```yaml
   strategies:
     my_new_strategy:
       class: rl_swing.strategies.my_new.MyNewStrategy
       params: {strategy_id: my_new_strategy, ...}
   ```
3. Add it to a runtime config's `strategies:` list.
4. Add a test asserting the new class satisfies its port (see
   `tests/contract/test_port_contracts.py`).

No service / runtime / pipeline code should change.

## Promoting a model

The spec's promotion gate is enforced by the model registry's
`approval_status`. Promotion is **manual**:

1. `rl-swing train --experiment configs/experiments/<exp>.yaml`
   produces a `model.zip` and `training_summary.json`.
2. `rl-swing validate --experiment configs/experiments/<exp>.yaml`
   produces a walk-forward report.
3. Operator reviews the report. Required passes (per spec §12.5.10):
   - Beats random policy
   - Beats at least one rule baseline after costs
   - 3+ of 5 seeds show acceptable behaviour
   - No single year/symbol drives the result
   - Acceptable drawdown / turnover
   - Stays acceptable under doubled slippage (`*_cost2x` rows in the
     report)
4. Operator transitions the registry status `TRAINED → VALIDATED →
   SHADOW_APPROVED → ...` (manual API; future work).
5. **No model with status != `LIVE_APPROVED` may be selected by a live
   broker adapter.** The runtime container's safety check enforces
   that.

## Live trading safeguards

Three independent flags must all be true to actually send a real order:

1. `runtime.allow_live_trading: true` in the runtime config.
2. `runtime.place_orders: true` in the runtime config.
3. `RL_SWING_LIVE_APPROVAL_TOKEN` environment variable set.

The default runtime config for `live_guarded` mode has the first two
set to `false`. The dependency container raises at startup if any
combination is incoherent (e.g. live mode requested with a paper
broker).

## Common issues

| Symptom                                                       | Fix                                                                                  |
|---------------------------------------------------------------|--------------------------------------------------------------------------------------|
| `FileNotFoundError: Model artifact not found`                 | Train first with `rl-swing train --experiment ...` before `validate`.                |
| `RuntimeError: Feature version mismatch`                      | The model was trained on a different feature version. Re-train, or use the matching feature pipeline. |
| `AlpacaPaperBrokerAdapter not implemented`                    | Phase 8 deliverable. This build runs research/shadow only against the simulated/no-op brokers. |
| Coverage gate fails after edits                               | `pytest --cov=rl_swing --cov-config=.coveragerc` — see `Missing` column for uncovered lines. |
| WRDS adapter returns no bars                                  | Make sure `cache_dir` points at the parquet cache populated by `trading-bot2/scripts/research/wrds_refresh.py`. |
