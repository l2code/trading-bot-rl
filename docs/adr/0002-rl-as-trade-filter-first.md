# ADR 0002 — RL as a trade filter / position sizer first

Status: Accepted (Phase 0A)
Date: 2026-05-06

## Context

Letting an RL agent learn end-to-end equity trading from raw price
data is a harder problem than this team can credibly solve in a
research timeline. The literature (FinRL, deep-RL stock-trading
ensembles, position-sizing RL studies) and the reference bot's
experience both point to RL as a layer over rule-based candidates
rather than a free-form trader.

## Decision

The MVP role for RL is:

* Rule-based strategies (momentum, mean reversion, breakout, trend
  following) generate `CandidateTrade` objects.
* The RL `PolicyScorer` chooses one of `{skip, take_25, take_50,
  take_100}` for each candidate.
* The risk engine has the final say and may scale size further down.

Long-only in the MVP. Shorting stays an interface capability disabled
by config.

## Consequences

* The action space is tiny and discrete, which keeps PPO/DQN tractable.
* Baselines (random, always-take, momentum-rule, buy-and-hold SPY/QQQ)
  are easy to compare against — they are policies in the same harness.
* Future modes (`exit_policy`, `position_sizer`, `portfolio_allocator`)
  are reachable by swapping the environment's `ActionMapper` /
  `RewardModel` / `EpisodeSampler` without touching the rest of the
  pipeline.
* If RL cannot beat simple baselines on this problem, we don't move
  to paper trading.
