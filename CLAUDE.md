# CLAUDE.md — operating brief for trading-bot-rl

This file is load-bearing. Claude reads it at the start of every
session. Update it when state changes. Keep it short, concrete, and
brutally honest about where things stand.

The patterns and discipline here are distilled from
`SDLC_LESSONS_FOR_NEW_PROJECT.md` (lessons from `trading-bot2`).
Treat that document as the meta-spec; this file is the project-
specific application.

---

## 1. What this project optimizes for

A reinforcement-learning-assisted equity swing-trading research
platform with a pluggable variant architecture. The optimization
function, in priority order:

1. **Research integrity.** No silent train-test contamination, no
   exploratory results passed off as decision-grade, no soft NO_GO
   verdicts. Every artifact carries its data tier and an explicit
   {GO, SHADOW_ONLY, NO_GO} verdict.
2. **Pluggability.** Variants (filter v1, selector v2, future v3+)
   plug in via the component registry. Adding a variant is one new
   file plus one registry entry. No surgery on trainer or walk-
   forward.
3. **Reproducibility.** Same experiment YAML + same seed = same
   numbers, every time. Kaggle and local must agree to the bit on
   smoke runs.
4. **Test discipline.** 242 tests today; coverage floor is 85%.
   Every new variant ships with its own test module.

What this project explicitly does *not* yet optimize for:
- Live trading (we're research-grade only — no broker connection,
  no order routing, no money at risk).
- Multi-cycle walk-forward (we run one cycle; spec calls for 3-4).
- Decision-grade data (yfinance is exploratory tier; WRDS is the
  intended canonical source but isn't yet wired into experiment
  runs — see §6).

---

## 2. Current strategic state (as of 2026-05-06)

### v1 (filter_v001) — NO_GO on yfinance starter_equities

After three increasingly aggressive interventions (turnover penalty
0.02 → 0.30, skip mirror 0 → 1.0, candidate threshold tightening
relaxed to widen the pool 309 → 477), the trained PPO converges to
"always take" — bit-identical to `baseline_always_take_100`. See
`research/diary/v001_filter_loose_NO_GO.md` for the durable verdict.

The candidate set produced by the strategy stage has so much
positive expected value that "always take" is genuinely the EV-
optimal policy. The filter framing has no lift on this universe in
this regime.

### v2 (selector_v002) — NO_GO on yfinance starter_equities

500k×3 Kaggle run completed. Trained model **NO_GO** vs strongest
selector baseline (4 of 5 improved, 1 material regression on
max_drawdown +12pp) and NO_GO vs v1 trained (1 of 5 improved, 3
material regressions). See
`research/diary/2026-05-06_v002_selector_NO_GO.md`.

Notable: v2 collapsed *differently* from v1 — to a "Momentum
specialist" (`per_strategy_take_counts = [323, 0, 0]`) rather
than "always take everything." Random selector across all 3
strategies beats trained selector on composite score, which is
the textbook entropy-collapse signature. Issue #8 (Optuna sweep
on `ent_coef` and `learning_rate`) is the next experiment.

### Cross-variant summary so far

Both v1 and v2 collapse to degenerate policies under default PPO
hyperparams on this candidate distribution. Different shapes,
same wall. The framing alone (filter vs selector) does not unlock
learning here.

Per-strategy training-EV analysis (#15, completed) showed that
v2's "Momentum specialist" collapse is **partially rational**:
Momentum has the highest mean risk-adjusted return on training
data (+0.327 vs Breakout +0.279 vs RSI +0.151), so preferring
Momentum *first* makes sense. But specializing to *only*
Momentum is irrational — Breakout has 85% of Momentum's EV
across 8,411 candidates, so a portfolio policy that takes
Momentum AND Breakout would dominate. That's the
entropy-collapse component, isolated.

Open follow-ups: #17 (take-all-fired baseline; bounds the EV
the trained model leaves on the table), #8 (Optuna sweep with
refined success criteria — does higher entropy diversify across
strategies?), #5 (multi-cycle WF), #4 (canonical replication on
WRDS).

### Pluggable variant architecture — operational

`rl_swing.rl.variants.base.TrainingVariant` is the contract. v1 and
v2 are registered in `configs/components/components.yaml` under
`rl_variants`. Future variants are one file plus one registry line.

---

## 3. Non-negotiable workflow rules

### 3.1 Issue-first

Every substantive change gets a tracked issue with acceptance
criteria *before* implementation. Use the templates in
`.github/ISSUE_TEMPLATE/`.

Quick fixes (typo, comment, one-liner) skip this. Anything that
touches a variant, a strategy, the reward model, the env, or the
training loop does not skip this.

### 3.2 Branch + commit prefix

- Branch: `kind/N-short-description` where `kind ∈ {feat, fix,
  rfc, docs, refactor}` and N is the issue number. Example:
  `feat/22-walkforward-multi-cycle`.
- Commit prefix: `<KIND>-<N>: short summary`. Example:
  `FEAT-22: multi-cycle walk-forward harness`.
- PR body includes `Closes #N`.

This applies going forward. Pre-existing commits on `main` are
grandfathered.

### 3.3 Default-OFF for anything that affects an active behavior

If a change adds new training/validation/inference behavior, ship
it default-OFF behind an explicit flag. The flag's docstring
documents what happens when False. Operator flips after observation
period (≥30 events, ≥one full Kaggle run with the flag exercised).

This applies even to research code: a new reward shaping mode, a
new candidate aggregation, a new feature should ship behind a flag
so we can A/B against the existing behavior without code surgery
to revert.

### 3.4 Honest verdicts

Every decision-bearing experiment produces a durable artifact under
`research/diary/<exp>_<verdict>.md`. The verdict is one word:
**GO**, **SHADOW_ONLY**, or **NO_GO**. No "promising" or "needs
more analysis" or "directionally encouraging." If the gate isn't
met, the verdict is NO_GO.

The Phase-24-equivalent acceptance gate is defined in
`src/rl_swing/rl/validation/acceptance_gate.py`: a candidate must
improve at least 2 of 5 metrics over the strongest baseline. See
`docs/acceptance_gates.md` for details.

### 3.5 Data-tier labeling

Every research artifact and every Kaggle run names its data
provider AND its tier:

| Tier | Provider | Decision authority |
|------|----------|--------------------|
| **canonical** | WRDS (CRSP) — survivorship-aware, point-in-time | decision-grade |
| **execution-realism** | (future: Databento) | slippage/cost calibration |
| **exploratory** | yfinance, synthetic_* | quick-look only; never decision-grade |

A backtest using yfinance labeled "decision-grade" is a research-
integrity violation. See `docs/data_tiers.md`.

### 3.6 Local verification is the merge gate

Before claiming "done":
1. `pytest tests/ -q` passes (all of it, not just the test you
   wrote).
2. `ruff check src/ tests/` is clean.
3. PR body lists what was run, with counts.

The 85% coverage floor enforces test discipline. Pre-commit hooks
enforcing ruff + doc-drift are *planned* (#9) but not yet wired —
until they are, the discipline is by-convention. CI is configured
but does not gate merges (per the trading-bot2 lesson §2.2).

### 3.7 Critical self-review pass before merge

After local verification (§3.6) passes and the PR is open, run a
deliberate critical pass of your own diff before merging. Treat it
like a stern reviewer's read of the code. **Capture the review as
a PR comment** so a future reader can audit what was checked.

The full checklist lives in `CONTRIBUTING.md` §7. Highlights:

- `ruff` clean on every touched file (not just new ones).
- Test counts in CLAUDE.md / scorecard / README match the new total.
- No claims about tooling that doesn't exist yet (no aspirational
  pre-commit / CI / script statements).
- PR body's `Closes #N` is *only* for issues whose AC is fully met.
  Partial progress is `Refs #N` plus a follow-up issue filed
  *before* merge.
- Ambiguous design choices documented in code AND in a follow-up
  issue.
- No accidental `__pycache__` / per-run caches in the diff.

PR #10's self-review caught 6 real issues that would otherwise have
shipped uncaught. The cost of skipping the pass is real; the cost
of running it is ~5 minutes. Always run it.

---

## 4. Roadmap (current, ordered by impact-per-effort)

Tier 1 (do first, cheap and informative):
1. Multi-cycle walk-forward (3-4 windows, slide one year each).
2. Cross-strategy agreement features in observation.
3. Optuna sweep on entropy_coef and learning_rate.

Tier 2 (architectural, medium effort):
4. MaskablePPO action masking for selector variants.
5. RecurrentPPO (LSTM) policy for multi-day context.
6. Per-symbol embeddings.
7. Continuous sizing head (Beta distribution).

Tier 3 (substantive, larger):
8. Distributional RL (IQN) — uncertainty-aware decisions.
9. Portfolio-aware decisions (positions in observation).
10. Differential reward shaping (reward over baseline).
11. Offline RL pretraining from logged decisions.

Tier 4 (production):
12. Champion/challenger promotion gate.
13. Feature drift detection.
14. SHAP attribution.
15. Replay-buffer logging.

Tier 1 only commences after the v1/v2 comparison is recorded in a
research diary entry.

---

## 5. Repository conventions

- `src/rl_swing/` — package source.
- `src/rl_swing/rl/variants/` — pluggable RL variants. **One file
  per variant**, registered in `configs/components/components.yaml`.
- `tests/` — unit + integration + contract tests; coverage gated.
- `configs/experiments/` — one YAML per experiment, named after
  the variant + version.
- `configs/components/` — registry of swappable adapters.
- `configs/universes/` — symbol lists.
- `kaggle/` — kaggle-script kernel + metadata template.
- `scripts/` — operational scripts (kaggle_run, future doc-drift
  check, future param-table audit).
- `research/diary/` — durable verdict artifacts. One per decision-
  bearing experiment cycle. Required sections per template.
- `data/` — local cache and per-run kaggle dirs (gitignored where
  appropriate).
- `docs/` — design docs, acceptance gates, data tiers.

---

## 6. Open structural debts

Items that are NOT yet right and need attention:

1. **WRDS not wired into Kaggle runs.** Canonical-data results are
   blocked. yfinance is fine for v1/v2 architectural validation
   but any final decision-grade run needs WRDS.
2. **No multi-cycle walk-forward.** One-cycle WF means our v1 NO_GO
   is on a single test year. Could be year-specific.
3. **No purge / embargo** in the WF harness. Train-test boundary
   contamination is theoretically possible.
4. **No doc-drift script yet.** `configs/experiments/*.yaml` and
   the dataclass fields can drift silently.
5. **No issue templates pre-populated.** The templates exist but
   ad-hoc filings still get used.
6. **No champion/challenger promotion gate.** The Phase-24
   equivalent gate is defined, but no code automates "is this
   candidate strictly better than the current champion."

These are tracked in `docs/scorecard.md`.

---

## 7. Anti-patterns to actively defend against

(Subset of `SDLC_LESSONS_FOR_NEW_PROJECT.md` §7 most relevant here.)

- **Telemetry field mistaken for gating field.** When adding to
  domain types: name reference-only fields with `_logged` or
  `_reference` suffix; document gating fields explicitly.
- **Synthetic test data persisting in "production" logs.**
  Eval results from `synthetic_*` providers must never be
  recorded as decision-grade. The data_provider field surfaces in
  every research diary entry; the diary template requires the
  tier label.
- **"Promising" hiding NO_GO.** Verdict line is required and is
  one word. If I'm tempted to write "promising," I write NO_GO and
  add a follow-up issue with the specific question that would
  change my mind.
- **Doc/config drift.** `configs/experiments/*.yaml` and the
  `_ExperimentCfg` dataclass need to stay in sync. Future:
  `scripts/check_param_doc_drift.py`.
- **Tiny noisy commits.** Bundle related changes into one
  reviewable commit. The `data/kaggle/__pycache__` accident on
  this branch is the cautionary tale — `git add -A` without
  scrutiny.

---

## 8. The single sentence

> Make it safer, more honest, and more observable *before* making
> it more active.
