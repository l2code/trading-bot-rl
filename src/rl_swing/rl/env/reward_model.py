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
    target_risk_pct: float = 0.02
    drawdown_penalty_weight: float = 0.10
    # 0.30 risk-adjusted units per trade (was 0.02 — way too small to
    # affect choice). With reward_clip=5, a marginal +0.5 risk-adj
    # winner now nets 0.20, while skipping it under full-mirror costs
    # 0.50; so taking weak winners is still right, but a near-zero EV
    # trade is clearly skip-worthy.
    turnover_penalty_weight: float = 0.30
    holding_period_penalty_weight: float = 0.05
    # Scale for the mirrored counterfactual on skip. With the
    # turnover_penalty_weight bump above, the take-side is paying a
    # 0.30 floor cost per trade. Full mirror (1.0) is now appropriate:
    # both "always take" and "always skip" are strictly dominated by
    # a discriminating policy, and the trade-cost floor keeps the
    # take side from dominating purely on positive base rate.
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
