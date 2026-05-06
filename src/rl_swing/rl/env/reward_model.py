"""Reward model.

Symmetric trade-filter reward: skip and take are evaluated on the same
risk-adjusted scale, so the agent has a meaningful choice on every
candidate rather than a degenerate "always take" optimum.

    if action == take:
        reward = clipped(net_return / target_risk)
                 - drawdown_penalty * peak_drawdown
                 - holding_period_penalty * (holding_days / max_holding_days - 1).clip(0)
                 - turnover_penalty

    if action == skip and we have a counterfactual:
        reward = -clipped(counterfactual_return / target_risk) * skip_scale
        (so skipping a +5% winner costs you the +2.5 you missed,
         and skipping a -5% loser earns you the +2.5 you avoided.)

    if action == skip and counterfactual is unknown (data ran out):
        reward = 0.0   # neutral

This design — earlier, the SKIP rewards were hard-coded to ±0.05/±0.10
while TAKE rewards ranged ±5.0 (clipped). That made "always take"
optimal for any candidate with positive expected return, even tiny ones,
because the agent was paying a 50x bigger penalty for skipping than it
ever earned by skipping. Mirroring the scales fixes that.
"""
from __future__ import annotations

from dataclasses import dataclass

from rl_swing.rl.env.execution_simulator import TradeOutcome


@dataclass
class RewardModel:
    """Reward model for the trade-filter / selector envs.

    Defaults are calibrated for **portfolio-scale, sized returns**
    post-FIX-22/23/#50. With size_pct=0.10, atr_stop=8% asset, a
    typical winner has risk_adj ≈ +0.4-0.8 (portfolio scale).
    Penalties are tuned to be small relative to that range so the
    discriminating signal isn't crowded out — see FIX-#49 for the
    calibration discussion.

    A previous calibration (turnover=0.30, holding=0.05) was tuned
    for the OLD unsized return scale and made every winner net-
    negative under sized returns; the agent collapsed to all-skip.
    Defaults updated to FIX-#58 to match the YAML weights, so any
    direct ``RewardModel()`` construction (smoke tests, default
    env path) gets the calibrated values out of the box.
    """
    target_risk_pct: float = 0.02
    drawdown_penalty_weight: float = 0.10
    turnover_penalty_weight: float = 0.05         # FIX-#58: was 0.30
    holding_period_penalty_weight: float = 0.02   # FIX-#58: was 0.05
    skip_counterfactual_scale: float = 1.0
    reward_clip: float = 5.0

    def reward_for_take(
        self,
        outcome: TradeOutcome,
        max_holding_days: int,
    ) -> float:
        risk_adj = outcome.return_pct / self.target_risk_pct
        risk_adj = max(-self.reward_clip, min(self.reward_clip, risk_adj))
        dd_pen = self.drawdown_penalty_weight * outcome.peak_drawdown_pct / max(self.target_risk_pct, 1e-4)
        holding_excess = max(0.0, outcome.holding_days / max(1, max_holding_days) - 1.0)
        time_pen = self.holding_period_penalty_weight * holding_excess
        turnover_pen = self.turnover_penalty_weight    # one trade = one turnover unit
        return float(risk_adj - dd_pen - time_pen - turnover_pen)

    def reward_for_skip(
        self,
        counterfactual: TradeOutcome | None,
    ) -> float:
        # No counterfactual (trade ran past the data we have) → neutral.
        if counterfactual is None:
            return 0.0
        # Mirror the take reward on the same risk-adjusted scale.
        # Skipping a winner of +R%/target costs you +R; skipping a loser
        # of -R%/target earns you +R. Drawdown / holding-period penalties
        # are NOT mirrored because they only accrue if you actually
        # held the position.
        risk_adj = counterfactual.return_pct / self.target_risk_pct
        risk_adj = max(-self.reward_clip, min(self.reward_clip, risk_adj))
        return float(-risk_adj * self.skip_counterfactual_scale)
