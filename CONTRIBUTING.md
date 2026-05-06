# Contributing to trading-bot-rl

The non-negotiable rules. Distilled from
`SDLC_LESSONS_FOR_NEW_PROJECT.md` and codified in `CLAUDE.md`.
This file is the surface every contributor (human or agent) reads
before opening a PR.

---

## 1. Issue-first, always

Every substantive change requires a tracked GitHub issue with
explicit acceptance criteria *before* implementation. Quick fixes
(typo, comment-only, single-line obvious) are exempt; anything
that touches a variant, strategy, reward model, env, training
loop, or evaluation surface is not.

Use the templates in `.github/ISSUE_TEMPLATE/`:
- **RFC** — substantive design decisions (new variant, new gate,
  new architecture).
- **FEAT** — implementing an approved design.
- **FIX** — bug fix with diagnosis and repro.
- **RESEARCH** — decision-bearing experimental run.
- **OPS** — process / infra / tooling.

The issue forces you to articulate AC up front. AC up front catches
scope drift, clarifies edge cases, and gives reviewers a check
against the implementation. PRs without issues get merged with
implicit AC that nobody can re-derive six months later.

## 2. Branch + commit conventions

- **Branch:** `kind/N-short-description` where `kind ∈ {rfc, feat,
  fix, ops, research, refactor, docs}` and `N` is the issue number.
  - Good: `feat/5-multi-cycle-walkforward`
  - Good: `fix/12-kaggle-spawn-fork-bug`
  - Bad: `my-changes`, `update-things`
- **Commit prefix:** `<KIND>-<N>: short summary`
  - Good: `FEAT-5: multi-cycle walk-forward harness, 3-cycle default`
  - Good: `FIX-12: SubprocVecEnv start_method spawn -> fork on Linux`
- **PR body** must contain `Closes #N` (or `Refs #N` for partial
  progress) so the merge auto-closes the issue.

## 3. Default-OFF for active-behavior changes

Anything that changes how the system *behaves* in training,
validation, or inference ships behind a flag that defaults to
**False**. The flag flips after at least one full Kaggle run with
the flag exercised in shadow mode.

The pattern, baked into the comment style on the flag:

```python
new_reward_mode_active: bool = False  # When False: legacy reward path runs.
                                       # When True: enables FEAT-N (RFC #M).
                                       # Flip after observation period.
```

Default-OFF is *not* a placeholder. It is the deliberately-shipped
state of the feature for the duration of the observation phase.
The flag itself is the contract.

## 4. Honest verdicts

Every decision-bearing experiment produces a durable artifact at
`research/diary/<YYYY-MM-DD>_<exp>_<verdict>.md` with one of three
verdict words:

- **GO** — passes the acceptance gate (≥2 of 5 metrics improved
  over baseline, no material regression). Eligible for next
  promotion stage.
- **SHADOW_ONLY** — partial improvement (1 of 5) without
  regression. Worth shadow observation but not GO.
- **NO_GO** — gate not met. Do not invest further effort in this
  framing without a fresh question.

"Promising," "directionally encouraging," "needs more analysis" are
banned. If the gate isn't met, the verdict is NO_GO. If you want
to re-investigate, file a new RESEARCH issue with a specific
question.

The acceptance gate is implemented in
`src/rl_swing/rl/validation/acceptance_gate.py` and documented in
`docs/acceptance_gates.md`.

## 5. Data-tier labeling

Every experiment names its data provider AND its tier
(`docs/data_tiers.md`):

| Tier | Provider | Decision authority |
|------|----------|--------------------|
| canonical | WRDS / CRSP | decision-grade |
| execution-realism | (future: Databento) | calibration only |
| exploratory | yfinance, synthetic_* | quick-look only |

A research artifact run on exploratory tier *cannot* earn a GO
verdict. At best it produces SHADOW_ONLY pending canonical
replication. A backtest using yfinance labeled "decision-grade"
is a research-integrity violation, not a methodology error.

## 6. Local verification is the merge gate

Before claiming "ready":

```bash
python3 -m pytest tests/ -q
python3 -m ruff check src/ tests/
```

Both must be clean. The PR body lists what was run, with counts:

```
Local verification:
  - pytest tests/ -q: 230 passed
  - ruff check src/ tests/: clean
  - smoke train (configs/experiments/ppo_filter_smoke.yaml, 4k steps): score 0.4250
```

CI is configured but does not gate merges (per the trading-bot2
lesson §2.2): local verification is faster and more reliable for
a small team. CI is the catch-net for things you forgot to run
locally, not the primary gate.

> **Note:** pre-commit hooks for ruff and the doc-drift check are
> *planned* (#9) but not yet wired in. Until they are, "ran
> ruff/pytest locally" is enforced by convention, not tooling.

## 7. Critical self-review pass before merge

After local verification passes and the PR is open, run a **deliberate
critical pass** of your own diff before merging. Treat the diff as
if a stern reviewer wrote it. The goal is to catch the issues that
look fine to the author but stick out to a reader.

This is not optional and is not a formality. PR #10's self-review
caught **6 real issues** — ruff errors, stale numbers in docs,
infrastructure claims that weren't yet true, an over-claimed
\`Closes #N\`, an ambiguous design choice — every one of which would
have shipped uncaught without the pass.

**Capture the self-review as a PR comment** so a future reader can
audit what was checked. A merged PR with no visible self-review is
a smell; the discipline lives or dies on visibility.

### Checklist

Run through every item. Pass or document a deliberate exception.

**Code quality**
- [ ] \`ruff check\` clean on every file the diff touched (not just
      the new files — touched files too).
- [ ] \`pytest tests/\` passes with the new total count, not just
      "still passes."

**Doc accuracy**
- [ ] Every count or number in CLAUDE.md / scorecard / README that
      could go stale matches reality (test count, variant count,
      issue references).
- [ ] No claim about tooling, hooks, scripts, or infrastructure
      that doesn't actually exist yet. If it's planned, say
      "planned (#N)" — not present-tense.
- [ ] Cross-references resolve — every \`#N\` and \`docs/...\` link
      points somewhere real.

**Issue / PR hygiene**
- [ ] PR body's \`Closes #N\` lines are *only* for issues whose AC
      is fully met by this PR. Partial progress is \`Refs #N\`, not
      \`Closes #N\`.
- [ ] Loose ends (work that was in scope but slipped) are filed as
      follow-up issues *before* merging — not after.
- [ ] Commit messages start with \`<KIND>-<N>:\` prefix.

**Design**
- [ ] Any new public surface (Protocol, dataclass field, YAML key)
      has a corresponding test or example.
- [ ] Ambiguous design choices (e.g., a metric whose "right
      direction" is debatable, a default value that may need tuning)
      are flagged in code comments AND in a follow-up issue.
- [ ] Default-OFF discipline (§3) honored for any active-behavior
      change.

**Repo hygiene**
- [ ] No accidentally-tracked \`__pycache__\`, per-run caches, or
      temporary files. Audit \`git status\` before \`git add\`; never
      \`git add -A\` without scrutiny.
- [ ] Any large generated artifact (e.g., kaggle output dirs) is
      gitignored.

### When to skip

Trivial doc-only changes (typo, comment grammar) and the kinds of
fixes you'd land in seconds may skip the full checklist. Anything
that touches code, tests, configs, workflow, or claims about the
project's state goes through the full pass.

### Anti-pattern

"I'll do the self-review after merging" is how the catches get
missed. The pass happens *before* the merge button. The PR comment
that captures the review is part of the merge artifact.

## 8. Coherent batches over noisy fragments

Group related changes into one issue-sized commit. Five PRs of 10
lines each that ship one feature is worse than one PR of 50 lines
that ships the same feature. Reviewer context-switching cost
dominates line count.

Counter-example exemption: when a fix and an unrelated cleanup
both happen to be in flight, they should be separate PRs.

## 9. Diagnose before fix; verify after deploy

Two operator-facing updates around any FIX:

1. **Diagnosis update.** "Here's what I think is wrong, here's the
   root cause, here's the proposed fix shape." *Before* writing
   code. Lets the operator correct your theory before you've burned
   time on the wrong fix.
2. **Verification update.** "Fix landed at SHA X, verified Y, alert
   quiesced at Z." *After* deploy. Creates a paper trail of what
   was actually verified.

This is a discipline-of-communication pattern, not tooling.

## 10. Doc-as-code drift prevention

`configs/experiments/*.yaml` and the `_ExperimentCfg` dataclass
must stay in sync. Adding a new config field without adding it to
the YAML schema or documentation is forbidden.

A `scripts/check_param_doc_drift.py` script is filed (#9) to
enforce this in pre-commit. **Until both #9 and pre-commit setup
land, the discipline is manual:** when you add a field to
`_ExperimentCfg`, you also update every existing experiment YAML.
(`docs/experiment_schema.md` is itself filed as a follow-up — it
doesn't yet exist.)

## 11. Anti-patterns to actively defend against

(See CLAUDE.md §7 for the full list distilled from the trading-bot2
lessons. The most relevant on this repo today:)

- **Telemetry mistaken for gating** — name reference-only fields
  with `_logged` or `_reference` suffix; document gating fields
  explicitly.
- **Synthetic data persisting as decision-grade** — every diary
  entry surfaces `data_provider` and `tier`. Synthetic results
  cannot be GO.
- **"Promising" hiding NO_GO** — see §4.
- **Doc/config drift** — see §10.
- **`git add -A` without scrutiny** — was the cause of an
  accidental `__pycache__` commit on this repo. Always use targeted
  `git add <paths>` or audit `git status` before committing.

---

## TL;DR

1. File an issue.
2. Branch off `main` with `kind/N-name`.
3. Implement, test locally, commit with `KIND-N:` prefix.
4. PR with `Closes #N`.
5. After merge, watch the verification (next Kaggle run, next test
   pass, next monitor cycle).
