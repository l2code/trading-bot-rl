# Project scorecard

Strengths, weaknesses, and structural debts. Updated periodically.
Last update: 2026-05-06 after applying SDLC lessons (issue #1).

---

## Strengths

- **Pluggable variant architecture.** `TrainingVariant` + component
  registry means new variants are one new file plus one registry
  entry. v1 (filter) and v2 (selector) live side by side cleanly.
- **Test discipline.** 242 tests today. Coverage floor 85%. Every
  variant ships with its own test module.
- **Pure-function reward + cost models.** `RewardModel` and
  `EquityExecutionModel` are stateless dataclasses; they run
  identically in training, validation, and any future live path.
- **Hexagonal core.** Domain dataclasses, Protocol-based ports,
  swappable adapters via the registry. Dependency container with
  three-flag-AND safety check on live mode (which is presently a
  stub).
- **Headless training.** Kaggle Notebooks orchestrated via
  `scripts/kaggle_run.py` — push, poll, download, no human in the
  loop.

## Weaknesses

- **Single-cycle walk-forward.** Spec calls for multi-cycle; we
  run one cycle. NO_GO results may be year-specific.
- **No purge / embargo at the train-test boundary.** Theoretical
  contamination risk. Filed under issue #5 alongside multi-cycle
  WF.
- **Default PPO hyperparams unexplored.** No Optuna sweep yet.
  Current policy collapse on v1 may be entropy-collapse rather
  than architecture-bound. Filed as #8.
- **No champion/challenger promotion gate.** Acceptance gate
  module exists (#6, landed) but no automated "is candidate
  strictly better than current champion."
- **No SHAP / decision attribution.** When v3+ produces a GO,
  we'll want to explain why; not yet built.

## Structural debts (CLAUDE.md §6 mirror)

| # | Debt | Issue | Severity |
|---|------|-------|----------|
| 1 | WRDS not wired into Kaggle runs — every Kaggle run is exploratory tier | [#4](https://github.com/l2code/trading-bot-rl/issues/4) | high (blocks GO verdicts) |
| 2 | No multi-cycle walk-forward — one test year only | [#5](https://github.com/l2code/trading-bot-rl/issues/5) | high (year-specific bias risk) |
| 3 | No purge / embargo at WF boundaries | [#5](https://github.com/l2code/trading-bot-rl/issues/5) (bundled) | medium |
| 4 | No doc-as-code drift script for experiment YAMLs | [#9](https://github.com/l2code/trading-bot-rl/issues/9) | medium |
| 5 | No issue templates pre-populated until #1 closes | [#1](https://github.com/l2code/trading-bot-rl/issues/1) | low (closing) |
| 6 | No champion/challenger promotion gate | not filed yet | medium (becomes urgent at first GO) |
| 7 | Acceptance gate module exists ([#6](https://github.com/l2code/trading-bot-rl/issues/6)) but isn't wired into walk_forward output yet | follow-up filed | low (gate is callable; just not auto-applied) |

## Research state

| Variant | Tier | Verdict | Diary |
|---------|------|---------|-------|
| `filter_v001` (loose) | exploratory (yfinance) | **NO_GO (provisional)** | [`2026-05-06_v001_filter_loose_NO_GO.md`](../research/diary/2026-05-06_v001_filter_loose_NO_GO.md) |
| `selector_v002` | exploratory (yfinance) | **NO_GO (provisional)** | [`2026-05-06_v002_selector_NO_GO.md`](../research/diary/2026-05-06_v002_selector_NO_GO.md) |
| per-strategy training-EV analysis | exploratory (yfinance) | partial-H2 (provisional) | [`2026-05-06_per_strategy_training_ev.md`](../research/diary/2026-05-06_per_strategy_training_ev.md) |

> All entries above are PROVISIONAL pending P1 simulator/evaluation
> fixes [#22](https://github.com/l2code/trading-bot-rl/issues/22),
> [#23](https://github.com/l2code/trading-bot-rl/issues/23),
> [#24](https://github.com/l2code/trading-bot-rl/issues/24).

## Roadmap progress

Tier 1 (cheap and informative):
- [ ] [#5](https://github.com/l2code/trading-bot-rl/issues/5) Multi-cycle walk-forward
- [ ] [#7](https://github.com/l2code/trading-bot-rl/issues/7) Cross-strategy agreement features
- [ ] [#8](https://github.com/l2code/trading-bot-rl/issues/8) Optuna sweep on entropy/LR

Tier 2 (architectural): MaskablePPO, RecurrentPPO, per-symbol embeddings, continuous sizing.

Tier 3 (substantive): distributional RL, portfolio-aware decisions, differential reward, offline RL.

Tier 4 (production): champion/challenger, drift detection, SHAP, replay-buffer logging.

See CLAUDE.md §4 for the full ordered roadmap.
