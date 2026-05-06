---
name: RESEARCH — experimental run with verdict
about: A decision-bearing experiment that produces a research/diary/ artifact
title: "RESEARCH: <variant> on <data tier> — <question>"
labels: research
assignees: ''
---

## Question

The single concrete question this experiment answers.

## Data tier

- [ ] canonical (WRDS / CRSP)
- [ ] execution-realism (Databento) — N/A yet
- [ ] exploratory (yfinance / synthetic_*)

A run on exploratory tier CANNOT produce a GO verdict. At best it
produces SHADOW_ONLY pending canonical replication.

## Methodology

- Experiment YAML: `configs/experiments/...`
- Variant: `filter_v001 | selector_v002 | ...`
- Data window: train / val / test
- Seeds: ...
- Hyperparams: link to YAML
- Cost layer: yes / no

## Acceptance gate

- [ ] Phase-24-equivalent: ≥2 of 5 metrics improved over strongest
      baseline (default)
- [ ] Other (specify):

## Verdict (filled at completion)

- [ ] GO
- [ ] SHADOW_ONLY
- [ ] NO_GO

Diary artifact: `research/diary/<YYYY-MM-DD>_<exp>_<verdict>.md`

## Cross-references
