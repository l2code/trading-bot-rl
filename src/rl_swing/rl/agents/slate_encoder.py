"""DeepSets-style slate encoder for v002 (FEAT-34, Phase 3 step 1).

After PR #73 closed Phase 1 with the finding that masked-PPO is
bit-identical to ``selector_baseline_first_fired`` even with FEAT-7
agreement features, the architecture-level shortcut needed a real
fix: an MlpPolicy on a flat per-slot vector trivially encodes
"always pick slot 0," and that's exactly the local optimum it lands
in.

This module implements a permutation-equivariant encoder over the
strategy slate:

  per-slot phi(x_k)             — shared small MLP, slot-invariant
  aggregate {phi(x_k)}_k         — sum + max + mean pooling
  rho_slot(phi(x_k), agg, ctx)   — per-slot scoring head
  rho_skip(agg, ctx)             — separate skip-action head

The slot weights are shared across slots, and the aggregate is
order-invariant. A model with this inductive bias **cannot**
trivially encode "always pick slot 0" — the per-slot logit depends
only on (this slot's features, the order-invariant slate summary,
the pack-level context), not on slot index.

This is supervised-only at v0: torch nn.Module + a regression
target. PR-2 wires the same module as a sb3 features extractor
for MaskablePPO; that's gated on this module clearing a cheap
offline diagnostic against the FEAT-7 HistGB ranker.

Lazy torch import via the trainer / scorer modules — importing
this file does require torch (it's installed in pyproject.toml's
top-level deps already).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class SlateEncoderConfig:
    """Hyperparameters for the v0 DeepSets-style slate encoder.

    Defaults are deliberately small — Phase 3 step 1 is supposed
    to test the *inductive bias*, not chase model-size gains. The
    architecture is:

        phi : (slot_feature_dim,) -> (embed_dim,)
        rho_slot : (embed_dim + 3*embed_dim + ctx_dim,) -> 1
        rho_skip : (3*embed_dim + ctx_dim,) -> 1

    where the 3*embed_dim is the concat of [sum_phi, max_phi, mean_phi]
    pools across slots. ``ctx_dim`` is the pack-level + portfolio
    feature width (frame features + agreement features + portfolio).
    """
    slot_feature_dim: int
    ctx_feature_dim: int
    n_slots: int
    embed_dim: int = 32
    phi_hidden: tuple[int, ...] = (64, 32)
    rho_slot_hidden: tuple[int, ...] = (32, 16)
    rho_skip_hidden: tuple[int, ...] = (32, 16)
    activation: str = "relu"
    dropout: float = 0.0
    extra: dict = field(default_factory=dict)


def _make_mlp(
    in_dim: int, hidden: tuple[int, ...], out_dim: int,
    *, activation: str, dropout: float,
) -> nn.Sequential:
    act_cls = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}[activation.lower()]
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class SlateEncoder(nn.Module):
    """Permutation-equivariant slate encoder.

    Inputs (forward):
      slot_features : Tensor of shape (B, N, slot_feature_dim)
      slot_mask     : Tensor of shape (B, N), 1 for fired slots, 0 for unfired.
                      Unfired slots are zeroed in the aggregate (no contribution
                      to sum/mean; ignored in max via -inf masking).
      ctx           : Tensor of shape (B, ctx_feature_dim) — pack-level + portfolio.

    Outputs (forward):
      slot_logits : Tensor of shape (B, N) — per-slot score (use mask at decision time).
      skip_logit  : Tensor of shape (B, 1) — separate skip score.
      logits      : Tensor of shape (B, N+1) — concat [skip_logit, *slot_logits],
                    the standard order matching ``MultiStrategySwingTradingEnv``'s
                    ``action_space = Discrete(1 + n_slots)``.
    """

    def __init__(self, config: SlateEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.phi = _make_mlp(
            config.slot_feature_dim, config.phi_hidden, config.embed_dim,
            activation=config.activation, dropout=config.dropout,
        )
        # rho_slot input: phi(x_k) + [sum, max, mean] aggregates + ctx
        rho_slot_in = config.embed_dim + 3 * config.embed_dim + config.ctx_feature_dim
        self.rho_slot = _make_mlp(
            rho_slot_in, config.rho_slot_hidden, 1,
            activation=config.activation, dropout=config.dropout,
        )
        rho_skip_in = 3 * config.embed_dim + config.ctx_feature_dim
        self.rho_skip = _make_mlp(
            rho_skip_in, config.rho_skip_hidden, 1,
            activation=config.activation, dropout=config.dropout,
        )

    def encode_slots(
        self,
        slot_features: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply phi to each slot; produce (slot_embeds, aggregate)."""
        # slot_features: (B, N, slot_feature_dim)
        # apply phi to every slot (shared weights) -> (B, N, embed_dim)
        b, n, _ = slot_features.shape
        flat = slot_features.reshape(b * n, -1)
        embeds = self.phi(flat).reshape(b, n, -1)
        # mask unfired slots: zero embed contribution
        mask3 = slot_mask.unsqueeze(-1).float()
        embeds_masked = embeds * mask3

        # Order-invariant aggregates over slots:
        sum_pool = embeds_masked.sum(dim=1)               # (B, embed_dim)
        # mean over fired slots only; avoid div-by-zero on empty packs.
        n_fired = slot_mask.sum(dim=1, keepdim=True).clamp(min=1).float()  # (B, 1)
        mean_pool = sum_pool / n_fired
        # max over fired slots; mask unfired with -inf so they never win.
        very_neg = torch.finfo(embeds.dtype).min
        max_input = embeds.masked_fill(~slot_mask.bool().unsqueeze(-1), very_neg)
        max_pool, _ = max_input.max(dim=1)                # (B, embed_dim)
        # If no fired slots at all, max_input is all -inf; replace with zeros.
        all_unfired = (slot_mask.sum(dim=1) == 0).unsqueeze(-1)
        max_pool = torch.where(all_unfired, torch.zeros_like(max_pool), max_pool)

        agg = torch.cat([sum_pool, max_pool, mean_pool], dim=-1)  # (B, 3*embed_dim)
        return embeds, agg

    def forward(
        self,
        slot_features: torch.Tensor,
        slot_mask: torch.Tensor,
        ctx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        embeds, agg = self.encode_slots(slot_features, slot_mask)
        b, n, e = embeds.shape

        # Per-slot logit input: concat (phi_k, agg, ctx). Broadcast agg/ctx
        # to per-slot shape so rho_slot is applied per (b, k).
        agg_b = agg.unsqueeze(1).expand(b, n, agg.shape[-1])
        ctx_b = ctx.unsqueeze(1).expand(b, n, ctx.shape[-1])
        slot_in = torch.cat([embeds, agg_b, ctx_b], dim=-1)
        slot_logits = self.rho_slot(slot_in.reshape(b * n, -1)).reshape(b, n)

        # Skip logit input: agg + ctx.
        skip_in = torch.cat([agg, ctx], dim=-1)
        skip_logit = self.rho_skip(skip_in)  # (B, 1)

        # logits ordered to match Discrete(N+1) action: [skip, slot_0, ..., slot_{N-1}].
        logits = torch.cat([skip_logit, slot_logits], dim=-1)
        return {
            "slot_logits": slot_logits,
            "skip_logit": skip_logit,
            "logits": logits,
        }
