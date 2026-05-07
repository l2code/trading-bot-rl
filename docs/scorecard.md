# Project scorecard

Strengths, weaknesses, and structural debts. Updated periodically.
Last update: 2026-05-06 after applying SDLC lessons (issue #1).

---

## Strengths

- **Pluggable variant architecture.** `TrainingVariant` + component
  registry means new variants are one new file plus one registry
  entry. v1 (filter) and v2 (selector) live side by side cleanly.
- **Test discipline.** 292 tests today. Coverage floor 85%. Every
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
| `filter_v001` (loose) | exploratory (yfinance) | **NO_GO (superseded by post-Phase-0)** | [`2026-05-06_v001_filter_loose_NO_GO.md`](../research/diary/2026-05-06_v001_filter_loose_NO_GO.md) |
| `selector_v002` (pre-Phase-0) | exploratory (yfinance) | **NO_GO (superseded by post-Phase-0)** | [`2026-05-06_v002_selector_NO_GO.md`](../research/diary/2026-05-06_v002_selector_NO_GO.md) |
| per-strategy training-EV analysis | exploratory (yfinance) | partial-H2 (numbers superseded; ranking stands) | [`2026-05-06_per_strategy_training_ev.md`](../research/diary/2026-05-06_per_strategy_training_ev.md) |
| `filter_v001` (post-Phase-0) | exploratory (yfinance) | **FINAL_NO_GO** | [`2026-05-06_v001_filter_post_phase0_FINAL_NO_GO.md`](../research/diary/2026-05-06_v001_filter_post_phase0_FINAL_NO_GO.md) |
| `selector_v002` (post-Phase-0) | exploratory (yfinance) | **FINAL_NO_GO** | [`2026-05-06_v002_selector_post_phase0_FINAL_NO_GO.md`](../research/diary/2026-05-06_v002_selector_post_phase0_FINAL_NO_GO.md) |
| `selector_v002_masked` (FEAT-29 / Phase 1) | exploratory (yfinance) | ~~SHADOW_ONLY~~ → **NO_GO** (rebaselined on yfinance per FIX-#78) | [`2026-05-06_v002_masked_SHADOW_ONLY.md`](../research/diary/2026-05-06_v002_masked_SHADOW_ONLY.md) |
| `selector_baseline_supervised` (FEAT-30 / Phase 1) | exploratory (yfinance) | **NO_GO** | [`2026-05-06_v002_masked_supervised_ranker_NO_GO.md`](../research/diary/2026-05-06_v002_masked_supervised_ranker_NO_GO.md) |
| Cross-strategy agreement features (FEAT-7 / Phase 1) + ranker re-test | exploratory (yfinance) | **NO_GO** (marginal +0.0038 composite; gap to random halved but not flipped) | [`2026-05-06_v002_feat7_agreement_features_NO_GO.md`](../research/diary/2026-05-06_v002_feat7_agreement_features_NO_GO.md) |
| `selector_v002_masked` retrain on FEAT-7 obs (Phase 1 closure tie-breaker) | exploratory (yfinance) | **NO_GO** (4-of-5 strict criteria fail; bit-identical to first_fired) | [`2026-05-06_v002_masked_feat7_tiebreaker_NO_GO.md`](../research/diary/2026-05-06_v002_masked_feat7_tiebreaker_NO_GO.md) |
| `selector_baseline_set_ranker` (FEAT-34 PR-1, Phase 3) | exploratory (yfinance) | ~~SHADOW_ONLY~~ → **NO_GO** (rebaselined on yfinance per FIX-#78; the synthetic 5-of-5 gate-pass was a contamination artifact) | [`2026-05-06_v002_set_ranker_SHADOW_ONLY.md`](../research/diary/2026-05-06_v002_set_ranker_SHADOW_ONLY.md) |
| `selector_baseline_set_ranker` (FEAT-34 PR-1b stabilized, Phase 3) | exploratory (yfinance) | **NO_GO** (rebaselined on yfinance per FIX-#78). The "lowest-DD-of-any-policy" and "highest-composite" claims were synthetic-only; per_strat distinctness from first_fired survives. | [`2026-05-06_v002_set_ranker_stabilized_NO_GO.md`](../research/diary/2026-05-06_v002_set_ranker_stabilized_NO_GO.md) |
| 2026-05-07 canonical yfinance rebaseline (FIX-#78) | exploratory (yfinance) | **REBASELINE** — every trained selector is NO_GO vs random on yfinance 2022; masked-PPO bit-identical to first_fired holds | [`2026-05-07_d4_canonical_yfinance_rebaseline.md`](../research/diary/2026-05-07_d4_canonical_yfinance_rebaseline.md) |

> **Phase 0 fully closed.** Both post-Phase-0 entries are
> `FINAL_NO_GO` with audit-v2 / phase0-final metrics (daily-P&L
> basis, FIX-#36; 260 trading days). Pre-Phase-0 entries are
> retained as historical record with `SUPERSEDED` banners. v1
> trained PPO is bit-identical to `baseline_always_take_100`
> (material-DD regression caps verdict). v2 unmasked trained PPO is
> bit-identical to `selector_baseline_always_skip` (3-metric
> material regression). Phase 1 step 1 (#29 MaskablePPO) landed
> SHADOW_ONLY — Phase-24 gate output GO (4-of-5 improved, no
> material regressions); but **bit-identical to
> `selector_baseline_first_fired`** (3-line baseline: take the
> lowest-index fired strategy) — see the masked-PPO diary
> addendum. Phase 1 step 2 (#30 supervised ranker) landed NO_GO
> — 3 material regressions vs random; doesn't beat masked-PPO.
> Phase 1 next (revised): #7 cross-strategy agreement features,
> then #8 Optuna with tightened acceptance criterion ("beat
> first_fired absolute, not just random gate-relative").

## Roadmap progress

Tier 1 (cheap and informative):
- [ ] [#5](https://github.com/l2code/trading-bot-rl/issues/5) Multi-cycle walk-forward
- [ ] [#7](https://github.com/l2code/trading-bot-rl/issues/7) Cross-strategy agreement features
- [ ] [#8](https://github.com/l2code/trading-bot-rl/issues/8) Optuna sweep on entropy/LR

Tier 2 (architectural): MaskablePPO, RecurrentPPO, per-symbol embeddings, continuous sizing.

Tier 3 (substantive): distributional RL, portfolio-aware decisions, differential reward, offline RL.

Tier 4 (production): champion/challenger, drift detection, SHAP, replay-buffer logging.

See CLAUDE.md §4 for the full ordered roadmap.
