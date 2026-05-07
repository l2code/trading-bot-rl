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
4. **Test discipline.** 276 tests today; coverage floor is 85%.
   Every new variant ships with its own test module.

What this project explicitly does *not* yet optimize for:
- Live trading (we're research-grade only — no broker connection,
  no order routing, no money at risk).
- Multi-cycle walk-forward (we run one cycle; spec calls for 3-4).
- Decision-grade data (yfinance is exploratory tier; WRDS is the
  intended canonical source but isn't yet wired into experiment
  runs — see §6).

---

## 2. Current strategic state

This section is a **stable snapshot** — one row per active variant.
For findings, run results, RFC outcomes, and decisions, see
`research/CHANGELOG.md`. For per-experiment artifacts see
`research/diary/`.

### Variant status (one line each)

| Variant | Tier | Verdict | Latest |
|---------|------|---------|--------|
| `filter_v001` | exploratory (yfinance) | **FINAL_NO_GO (post-Phase-0)** | 2026-05-06 |
| `selector_v002` | exploratory (yfinance) | **FINAL_NO_GO (post-Phase-0)** | 2026-05-06 |
| `selector_v002_masked` (FEAT-29) | exploratory (yfinance) | **SHADOW_ONLY (Phase 1)** | 2026-05-06 |

> **Phase 0 fully closed** — all 16 P1+P2 simulator/evaluation
> fixes plus FIX-AUDIT-V2 (#56–#59) and FIX-AUDIT-V3 (#61, #62)
> merged. Audit-v2 / phase0-final Kaggle runs landed; both
> unmasked variants are FINAL_NO_GO with corrected daily-P&L
> metrics. v1 trained PPO is bit-identical to `baseline_always_take_100`
> (material-DD regression caps verdict). v2 unmasked trained PPO is
> bit-identical to `selector_baseline_always_skip` (3-metric
> material regression). See diary entries for details.

> **Phase 1 step 1 landed (#29 MaskablePPO for v2): SHADOW_ONLY.**
> Phase-24 gate output is GO (4-of-5 improved, no material
> regressions); per_strategy_take_counts=[1423, 79, 278]
> diversifies across all three strategies (vs unmasked [0, 0, 0]).
> Verdict capped at SHADOW_ONLY by (a) exploratory-tier
> yfinance (CLAUDE.md §3.5 — promotion to GO requires WRDS, #4)
> and (b) only 1-of-3 seeds found a productive policy and even
> that one was transient (seed 11 escaped at step 300k then
> collapsed back to all-skip; seed 22 stayed all-skip; seed 33
> broke into take-everything). Masking is necessary but not
> sufficient — default ent_coef=0.01 looks too low.

> **Phase 1 step 2 landed (#30 supervised ranker baseline): NO_GO.**
> sklearn HistGB on slate features + per-slot fields → realized
> risk-adjusted return; argmax with skip-at-0. NO_GO vs random (3
> material regressions on return / sharpe / PF). Also DOES NOT beat
> masked-PPO. **Important refinement to the masked-PPO finding:**
> the trained masked-PPO is *bit-identical* (every metric to 6
> decimals) to `selector_baseline_first_fired` — a 3-line "take
> the lowest-index fired strategy" rule. The masked-PPO didn't
> learn anything beyond first_fired. SHADOW_ONLY still stands
> (gate output + tier rules unchanged) but the "PPO learned
> something" reading is wrong.

> **Phase 1 next** (revised after the supervised NO_GO + bit-
> identity finding, ordered by EV):
> 1. **#7 cross-strategy agreement features** — highest-leverage.
>    Current obs gives per-slot features but not "do strategies
>    agree on this (symbol, date)." Cheaper than architectural
>    work and could shift both PPO and ranker verdicts.
> 2. **#8 Optuna sweep on masked-PPO** with a tightened
>    acceptance criterion: must clear the gate vs random AND beat
>    `first_fired` on absolute composite score.
> 3. **#34 set/attention or #32 chronological v3** only if (1)
>    and (2) both fail.
>
> v1 PPO, v2 unmasked PPO, and the v0 supervised ranker are all
> closed for further compute at default hyperparams.

Diary entries linked from `docs/scorecard.md`. Narrative findings
live in `research/CHANGELOG.md` and the per-entry diary files —
not here.

### Pluggable variant architecture

`rl_swing.rl.variants.base.TrainingVariant` is the contract. v1 and
v2 are registered in `configs/components/components.yaml` under
`rl_variants`. Future variants are one file plus one registry line.

### What's next (queued)

- `#15` (closed): per-strategy training-EV analysis → PARTIAL-H2
- `#17`: take_all_fired selector baseline (bounds residual EV)
- `#8`: Optuna sweep on `ent_coef` + `lr` — refined success
  criterion is "trained model's `per_strategy_take_counts` shows
  diversification," not just "trained > baseline"

The full roadmap and tracked structural debts live in §4 / §6 below
(stable, not changelog material).

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

## 4. Roadmap

The order matters. Skipping ahead = scaling complexity on
foundations that may still be wrong. Each phase gates on the
previous phase's evidence. From the operator's 2026-05-06 review.

### Phase 0 — fix the foundations, then re-run v1 and v2

P1 simulator/evaluation correctness:
- [#22](https://github.com/l2code/trading-bot-rl/issues/22)
  size_pct now scales return — ✅ merged
- [#23](https://github.com/l2code/trading-bot-rl/issues/23)
  round-trip cost charged once despite per-side docstring
- [#24](https://github.com/l2code/trading-bot-rl/issues/24)
  walk-forward lacks lookback warmup
- [#36](https://github.com/l2code/trading-bot-rl/issues/36)
  portfolio equity-curve evaluation (sums-of-returns isn't real)

P2:
- [#26](https://github.com/l2code/trading-bot-rl/issues/26)
  v2 hindsight-best skip counterfactual
- [#25](https://github.com/l2code/trading-bot-rl/issues/25)
  v2 not wired into runtime DecisionPipeline (deferred until Phase 2)

Then re-run v1 and v2 on Kaggle with corrected metrics. Update
diaries. **Phase 0 closes when both verdicts are no longer
PROVISIONAL.**

**Until Phase 0 is done, no new training experiments and no new
RL machinery.** Better algorithms on a broken simulator just learn
a cleaner version of the wrong economics.

### Phase 1 — Next Stage (post-Phase-0)

Per the operator's "Next Stage" sequence:
`fix correctness → prove baselines → run matrix → ablate → shadow → paper`

1. **Prove baselines** — [#30](https://github.com/l2code/trading-bot-rl/issues/30)
   add supervised / contextual-bandit selector baseline. If PPO
   can't beat a simple ranker, RL machinery isn't earning its
   complexity yet. [#17](https://github.com/l2code/trading-bot-rl/issues/17)
   `take_all_fired` baseline lands here too.
2. **MaskablePPO** — [#29](https://github.com/l2code/trading-bot-rl/issues/29)
   formal action masking for v2 selector. Replaces illegal-action
   penalty with hard mask.
3. **Promotion matrix** — [#37](https://github.com/l2code/trading-bot-rl/issues/37)
   run all variants (v1 PPO, v1 DQN, v2 highest-signal, v2 masked
   PPO, v2 supervised) through the same battery (synthetic /
   yfinance / WRDS × multi-seed × multi-year WF × cost stress ×
   crisis windows × universes).
4. **Baseline-dominance gate** — [#38](https://github.com/l2code/trading-bot-rl/issues/38)
   v2 must beat *every* declared baseline (not just the strongest)
   to earn GO.
5. **Ablations** — [#39](https://github.com/l2code/trading-bot-rl/issues/39)
   feature-subset sweep tells us what's load-bearing vs noise
   memorization.

### Phase 2 — runtime + shadow mode

6. **Wire v2 into runtime** — [#25](https://github.com/l2code/trading-bot-rl/issues/25).
7. **Shadow mode** — [#40](https://github.com/l2code/trading-bot-rl/issues/40)
   record selections, skipped alternatives, counterfactuals, drift.
   No trading. The bridge between "passes validation" and "touches
   real money."

Only after a model clears Phase 1 + sustained Phase 2 does paper
trading make sense.

### Phase 3 — size and architecture refinements

- [#31](https://github.com/l2code/trading-bot-rl/issues/31)
  Size-aware v2 action space (recovers v1's sizing dimension).
- [#34](https://github.com/l2code/trading-bot-rl/issues/34)
  Set/attention slate encoder (kills slot-index overfitting).
- [#32](https://github.com/l2code/trading-bot-rl/issues/32)
  Portfolio-aware chronological v3 — full sequential RL.

### Phase 4 — risk-awareness, OOD safety, production rails

- [#33](https://github.com/l2code/trading-bot-rl/issues/33)
  Distributional / quantile selector head.
- [#35](https://github.com/l2code/trading-bot-rl/issues/35)
  Conservative offline RL (CQL).
- Champion/challenger gate (not yet filed).
- Feature drift, SHAP, replay-buffer logging (not yet filed).

### Cheap diagnostics — runnable in parallel, don't gate phases

- [#5](https://github.com/l2code/trading-bot-rl/issues/5) Multi-cycle WF.
- [#7](https://github.com/l2code/trading-bot-rl/issues/7) Cross-strategy agreement features.
- [#8](https://github.com/l2code/trading-bot-rl/issues/8) Optuna sweep on entropy/lr (deferred until Phase 0 lands).
- [#9](https://github.com/l2code/trading-bot-rl/issues/9) Doc-drift script.
- [#11](https://github.com/l2code/trading-bot-rl/issues/11) Wire acceptance_gate into walk_forward.
- [#13](https://github.com/l2code/trading-bot-rl/issues/13) Pre-existing ruff cleanup.

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
- `research/CHANGELOG.md` — rolling project log: dated entries
  for findings, RFC outcomes, run verdicts, infra changes.
  Append on merge per CONTRIBUTING.md §11. **This file, not
  CLAUDE.md, is where narrative findings live.**
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
