# Project changelog

Rolling, reverse-chronological log of substantive project events:
findings, decisions, run verdicts, RFC outcomes, infrastructure
changes that affect how we work.

This is **not** the operating brief (`CLAUDE.md`) and **not** a
per-experiment artifact (`research/diary/`). It's the
"what-happened-when" log a future reader (human or Claude
session) reads to catch up without having to re-read every PR.

## Entry format

```
## YYYY-MM-DD — <kind>: <one-line summary>

**Issue/PR:** #N (link)
**Diary:** research/diary/...md (if applicable)

Two to five sentences of context: why this happened, what we
learned, what changed in our model of the project.
```

Kinds: `RESEARCH`, `RFC`, `FEAT`, `FIX`, `OPS`, `STRUCTURAL`.

When a PR closes a research issue OR makes a substantive process /
infra change, the same PR appends an entry here. CONTRIBUTING.md §12
codifies the rule.

---

## 2026-05-06 — OPS: CI switched to workflow_dispatch only (manual-only)

**Issue:** [#45](https://github.com/l2code/trading-bot-rl/issues/45)
**PR:** (this PR)

Operator request: stop CI from auto-running on push/PR. The workflow
ran across 3 Python versions per push, costing GitHub Actions
minutes on every commit including doc-only changes. Local
verification is the merge gate per CONTRIBUTING.md §6 anyway, so
the auto-trigger was duplicative. CI remains available as a
catch-net the operator can run manually from the UI.

## 2026-05-06 — FIX: P1 walk-forward warmup; first ~200 test days no longer use degraded features

**Issue:** [#24](https://github.com/l2code/trading-bot-rl/issues/24)
**PR:** [#44](https://github.com/l2code/trading-bot-rl/pull/44)

Third P1 simulator/eval fix. Both `walk_forward.py` and
`trainer._build_env` now load bars with 1.5 trading years of
warmup before the requested window. Long-lookback features (sma_200,
return_60d, atr_pct_14, etc.) populate before the in-window region
and frames are filtered to the eval window before candidate
generation. Three new unit tests in
`tests/unit/test_warmup_helpers.py`. One P1 (#36 portfolio
equity-curve eval) and one P2 (#26 hindsight skip-CF) remain
before v1/v2 re-runs can replace the PROVISIONAL diary entries.

## 2026-05-06 — FIX: P1 round-trip cost charged 2x per-side (fixed)

**Issue:** [#23](https://github.com/l2code/trading-bot-rl/issues/23)
**PR:** [#43](https://github.com/l2code/trading-bot-rl/pull/43)

Second P1 simulator fix. `cost_model.cost_bps()` documented as
per-side; simulator was subtracting once. Now multiplies by 2 for
round-trip explicitly. Combined with FIX-22 (size scaling), a 10%
sized trade with 50bps per-side cost on a flat asset now correctly
produces -10bps portfolio drag (was -5bps before FIX-23, was -50bps
before FIX-22+23).

## 2026-05-06 — OPS: roadmap restructured around operator's "Next Stage" framework

**Issues filed:** [#36](https://github.com/l2code/trading-bot-rl/issues/36) (P1 portfolio equity-curve eval), [#37](https://github.com/l2code/trading-bot-rl/issues/37) (promotion matrix), [#38](https://github.com/l2code/trading-bot-rl/issues/38) (baseline-dominance gate), [#39](https://github.com/l2code/trading-bot-rl/issues/39) (ablation harness), [#40](https://github.com/l2code/trading-bot-rl/issues/40) (shadow mode), and earlier #29–#35 from the RL-design review.

Operator review reframed the project trajectory as a phased
sequence: `fix correctness → prove baselines → run matrix →
ablate → shadow → paper`. Each phase gates on the previous
phase's evidence; more runs only have value as controlled
experiments inside the matrix harness, not as brute-force
training.

CLAUDE.md §4 rewritten to reflect this. Phase 0 (P1 simulator
fixes + v1/v2 re-runs) now explicitly gates all post-Phase-0
work. Earlier impact-per-effort tier ordering preserved as
"cheap diagnostics" runnable in parallel.

A new P1 surfaced from the same review: portfolio equity-curve
evaluation (#36). The current `validation_composite_score`
sums per-trade returns and computes Sharpe / DD on the trade
sequence — not on a date-ordered daily P&L. Adds to the
PROVISIONAL banner scope.

## 2026-05-06 — FIX: P1 size_pct now scales realized portfolio return

**Issue:** [#22](https://github.com/l2code/trading-bot-rl/issues/22)
**PR:** [#28](https://github.com/l2code/trading-bot-rl/pull/28)

First of the P1 simulator fixes. `ExecutionSimulator` now returns
`return_pct` as the portfolio contribution (sized + cost-net), not
the asset's standalone percent return. New regression test asserts
`take_25` and `take_100` produce proportionally different returns.
Backward-compat `raw_return_pct` alias preserved. Two more P1s
queued (#23 round-trip costs, #24 WF warmup) plus the new #36
portfolio equity-curve evaluation.

## 2026-05-06 — STRUCTURAL: 5 simulator/evaluation bugs identified by code review; current verdicts marked PROVISIONAL

**Issues filed:** [#22](https://github.com/l2code/trading-bot-rl/issues/22) (P1 size-scale), [#23](https://github.com/l2code/trading-bot-rl/issues/23) (P1 round-trip cost), [#24](https://github.com/l2code/trading-bot-rl/issues/24) (P1 WF warmup), [#25](https://github.com/l2code/trading-bot-rl/issues/25) (P2 selector runtime), [#26](https://github.com/l2code/trading-bot-rl/issues/26) (P2 hindsight skip-CF)

Operator code review after the v002 NO_GO and per-strategy EV diary
landed identified five real issues affecting both the simulator
(`return_pct` not scaled by `size_pct`; round-trip cost charged
once despite per-side docstring) and walk-forward evaluation
(no lookback warmup, so first ~200 days of test window run on
degraded long-lookback features). Two additional v2-specific
issues: selector_v002 not wired into the runtime DecisionPipeline,
and the skip-reward counterfactual uses hindsight-best (max-over-
noise bias). All three current diary entries (v001 NO_GO, v002
NO_GO, per-strategy EV) marked PROVISIONAL pending the P1 fixes.
Optuna sweep (#8) paused — running on a broken simulator would
burn compute. Sequence: P1 fixes (#22/#23/#24) → re-run v1 and v2
with corrected metrics → then resume #8 with confidence the gate
output is meaningful.

## 2026-05-06 — OPS: separate rolling findings (CHANGELOG.md) from CLAUDE.md operating brief

**Issue:** [#19](https://github.com/l2code/trading-bot-rl/issues/19)
**PR:** [#20](https://github.com/l2code/trading-bot-rl/pull/20)

CLAUDE.md was bloating with per-experiment narrative. Split: CLAUDE.md
stays a stable operating brief (variant-status table, rules,
debts); this CHANGELOG.md absorbs chronological findings; per-
experiment artifacts stay under `research/diary/`. CONTRIBUTING.md
§11 now codifies the rule (append to changelog on merge).

## 2026-05-06 — FEAT: hyperparam-override CLI plumbing for kaggle sweeps

**Issue:** [#28](https://github.com/l2code/trading-bot-rl/issues/28) (closed)
**PR:** [#21](https://github.com/l2code/trading-bot-rl/pull/21)

`scripts/kaggle_run.py` now accepts `--hyperparam-overrides='{...}'`
which propagates via env var into `kaggle_train.py` and finally
into `train_from_experiment(hyperparam_overrides=)`. Override dict
merges over `cfg.hyperparams` (override wins). Prerequisite for
the Optuna entropy sweep (#8) — but blocked by P1 simulator fixes.

## 2026-05-06 — RESEARCH: per-strategy training-EV analysis (PARTIAL-H2)

**Issue:** [#15](https://github.com/l2code/trading-bot-rl/issues/15)
**PR:** [#18](https://github.com/l2code/trading-bot-rl/pull/18)
**Diary:** [`2026-05-06_per_strategy_training_ev.md`](diary/2026-05-06_per_strategy_training_ev.md)

Pure data analysis (no RL) to test whether v2's "Momentum specialist"
collapse is rational on training data (H2) or pure entropy collapse
(H1). Result: PARTIAL-H2. Momentum has the highest mean risk-
adjusted return on training data (+0.327 vs Breakout +0.279 vs RSI
+0.151), so preferring it is rational. But specializing to *only*
Momentum is irrational — Breakout has 85% of Momentum's EV across
8,411 candidates that the trained model ignored. The collapse has
both a rational direction (H2) and an irrational severity (H1).
This refines the success criterion for #8: not just "trained beats
baseline" but "per_strategy_take_counts shows real diversification."
Filed #17 (take_all_fired baseline) as a parallel diagnostic.

## 2026-05-06 — RESEARCH: v002 selector NO_GO on yfinance starter_equities

**Issue:** [#3](https://github.com/l2code/trading-bot-rl/issues/3)
**PR:** [#16](https://github.com/l2code/trading-bot-rl/pull/16)
**Diary:** [`2026-05-06_v002_selector_NO_GO.md`](diary/2026-05-06_v002_selector_NO_GO.md)

500k×3 Kaggle run completed. Phase-24 gate returns NO_GO twice:
4-of-5-improved-but-material-DD-regression vs strongest selector
baseline (random); 1-of-5-improved-with-3-regressions vs v1 trained.
Notable wrinkle: v2 collapsed *differently* from v1 — to a Momentum
specialist (`per_strategy_take_counts = [323, 0, 0]`) rather than
"always take everything." Random selector beats trained on composite
score (0.7037 vs 0.6665), the textbook entropy-collapse signature.
Both variants now NO_GO under default PPO hyperparams; the framing
(filter vs selector) is not the lever.

## 2026-05-06 — OPS: codify critical self-review pass before merge

**Issue:** [#12](https://github.com/l2code/trading-bot-rl/issues/12)
**PR:** [#14](https://github.com/l2code/trading-bot-rl/pull/14)

CONTRIBUTING.md §7 + CLAUDE.md §3.7. Self-review checklist captured
verbatim from PR #10's live demonstration: ruff on touched files,
test counts in docs match reality, no aspirational tooling claims,
`Closes #N` only for fully-met AC, ambiguous design choices flagged
in code AND filed, no `__pycache__`/cache files in diff. Self-review
captured as a PR comment so the audit trail is visible. PR #10's
self-review caught 6 real issues; this codifies the practice as
permanent.

## 2026-05-06 — OPS: apply trading-bot2 SDLC lessons (foundational docs + acceptance gate)

**Issue:** [#1](https://github.com/l2code/trading-bot-rl/issues/1)
**PR:** [#10](https://github.com/l2code/trading-bot-rl/pull/10)
**Diary:** [`2026-05-06_v001_filter_loose_NO_GO.md`](diary/2026-05-06_v001_filter_loose_NO_GO.md)

Distilled the patterns from `SDLC_LESSONS_FOR_NEW_PROJECT.md` into
this repo: CLAUDE.md operating brief, CONTRIBUTING.md workflow
rules, docs/data_tiers.md, docs/acceptance_gates.md (≥2 of 5
metric improvement gate), docs/scorecard.md, issue templates.
First research diary entry written for the v1 loose run (NO_GO,
0 of 5 metrics improved). Acceptance gate module + 11 tests
landed but not yet wired into walk_forward output (filed as #11).
Self-review during this PR caught and fixed 6 real issues that
would have shipped uncaught (#13, #11 follow-ups filed during
the review).

## 2026-05-06 — STRUCTURAL: pluggable RL variant architecture + v2 multi-strategy selector

**PR:** `af86e06` (pre-issue-first discipline; legacy)

Refactored trainer + walk_forward to dispatch via a TrainingVariant
abstraction registered in the ComponentRegistry. Adding a new
variant is now a single new file in `rl_swing/rl/variants/` plus
a one-line entry in `configs/components/components.yaml`. v1 logic
extracted as `FilterV001Variant`; v2 multi-strategy selector
implemented as `SelectorV002Variant` (per-(symbol, date) decisions,
Discrete(N+1) action, no candidate dedupe so the agent sees the
full slate).

## 2026-05-06 — RESEARCH: v001 filter NO_GO confirmed at three intervention levels

**PR:** `c1ba1ed`, `f907dda`, `0f7c18b` (pre-issue-first discipline; legacy)

Three increasingly aggressive interventions (turnover penalty
0.02 → 0.30, skip mirror 0 → 1.0, candidate threshold loosening to
widen pool 309 → 477) all converged to "always take" — bit-identical
to baseline_always_take_100 across 30 evaluation points × 3 seeds.
Confirms the candidate-set-EV problem: the strategy stage produces
candidates with so much positive expected value that "always take"
is genuinely the EV-optimal policy under any reasonable reward.
The filter framing has no lift on this universe in this regime.
Recorded with verdict in `research/diary/2026-05-06_v001_filter_loose_NO_GO.md`.
