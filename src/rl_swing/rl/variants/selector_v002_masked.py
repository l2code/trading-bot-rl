"""SelectorV002MaskedVariant — v2 selector with formal action masking
(FEAT-29).

Same env, observation, action space, reward, and per-pack semantics as
``selector_v002``. The only differences are:

  - Training algorithm: ``sb3-contrib.MaskablePPO`` (the trainer
    routes via ``cfg.algorithm == "MaskablePPO"``).
  - Inference scorer: ``MaskablePpoSelectorScorer`` — passes
    ``env.action_masks()`` to ``model.predict()`` so the policy
    cannot select non-fired strategy slots at all.
  - ``model_id`` for the trained policy is ``masked_ppo_selector_v002``
    so train/eval artifacts (kernel slugs, kaggle dirs, summary rows)
    clearly distinguish masked vs unmasked.

Replaces the illegal-action penalty pathway in the unmasked variant
with a hard mask. The unmasked ``selector_v002`` is unchanged and
remains available for A/B comparison; this variant is opt-in via the
component registry / experiment YAML.

Decision criteria (per operator scope, 2026-05-06): the trained
policy must beat ``selector_baseline_random`` (audit-v2 score
0.7186) on the Phase-24 gate, not merely escape all-skip. Escape
from the all-skip attractor is necessary but not sufficient.
"""
from __future__ import annotations

import logging

from rl_swing.rl.agents.selector_scorers import (
    AlwaysFirstFiredSelectorScorer,
    AlwaysSkipSelectorScorer,
    HighestSignalSelectorScorer,
    MaskablePpoSelectorScorer,
    RandomSelectorScorer,
    SelectorScorer,
)
from rl_swing.rl.variants.base import EvaluationContext, PolicyResult
from rl_swing.rl.variants.selector_v002 import SelectorV002Variant

_log = logging.getLogger(__name__)


class SelectorV002MaskedVariant(SelectorV002Variant):
    """Masked counterpart of ``SelectorV002Variant``. Inherits env
    construction (the env's ``action_masks()`` is harmless to vanilla
    PPO and required by MaskablePPO) and shares the per-scorer
    evaluation pipeline. Only the trained-PPO scorer is swapped to
    ``MaskablePpoSelectorScorer`` so eval-time inference uses the
    same mask the trainer used."""

    name: str = "selector_v002_masked"

    def evaluate(self, ctx: EvaluationContext) -> list[PolicyResult]:
        # Mirror of SelectorV002Variant.evaluate but swaps in the
        # MaskablePpoSelectorScorer. Kept as a separate override
        # rather than parameterizing the base because the algorithm
        # selection is a structural variant property, not a runtime
        # toggle.
        from datetime import datetime

        from rl_swing.domain import PortfolioState

        portfolio = PortfolioState(
            as_of=datetime(ctx.test_end.year, ctx.test_end.month, ctx.test_end.day),
            cash=100_000.0, equity=100_000.0,
        )
        # Same packing as the unmasked variant — same fired sets,
        # same n_slots — only the inference scorer differs.
        packs, n_slots = self._pack_for_eval(ctx, portfolio)

        scorers: list[SelectorScorer] = []
        if "random" in ctx.include_baselines:
            scorers.append(RandomSelectorScorer(seed=42))
        if "always_skip" in ctx.include_baselines or "never_take" in ctx.include_baselines:
            scorers.append(AlwaysSkipSelectorScorer())
        if "first_fired" in ctx.include_baselines:
            scorers.append(AlwaysFirstFiredSelectorScorer())
        if "highest_signal" in ctx.include_baselines:
            scorers.append(HighestSignalSelectorScorer())

        rl_added = False
        if ctx.artifact_path is not None and ctx.artifact_path.exists():
            scorers.append(MaskablePpoSelectorScorer(
                model_id=ctx.model_id,
                artifact_path=str(ctx.artifact_path),
                n_strategies=n_slots,
            ))
            rl_added = True

        results: list[PolicyResult] = []
        for s in scorers:
            res = self._evaluate_scorer(
                s, packs, ctx, n_slots, cost_stress_multiplier=1.0,
            )
            results.append(res)
            if ctx.include_cost_stress:
                res2 = self._evaluate_scorer(
                    s, packs, ctx, n_slots, cost_stress_multiplier=2.0,
                )
                results.append(PolicyResult(
                    **{**res2.to_dict(), "model_id": res2.model_id + "_cost2x"}
                ))

        for r in results:
            r.extras.setdefault("rl_model_present", rl_added)
            r.extras.setdefault("masking", "sb3_contrib_maskable_ppo")
        return results

    def _pack_for_eval(self, ctx: EvaluationContext, portfolio):
        # Thin wrapper so a future v3+ subclass can override the
        # packing without copy-pasting evaluate(). Today this is
        # identical to selector_v002's _pack_candidates.
        from rl_swing.rl.variants.selector_v002 import _pack_candidates
        return _pack_candidates(ctx.frames, portfolio)
