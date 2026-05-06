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
