"""Reward model.

Implements the spec's recommended trade-filter reward:

    if action == skip:
        reward = opportunity_score
            (mildly penalize missed winners; reward avoided losers)

    if action == take:
        reward =
              clipped(net_return / target_risk)
            - drawdown_penalty * peak_drawdown
            - holding_period_penalty * (holding_days / max_holding_days - 1).clip(0)
            - turnover_penalty
"""
from __future__ import annotations

from dataclasses import dataclass

from rl_swing.rl.env.execution_simulator import TradeOutcome


@dataclass
class RewardModel:
    target_risk_pct: float = 0.02
    drawdown_penalty_weight: float = 0.10
    turnover_penalty_weight: float = 0.02
    holding_period_penalty_weight: float = 0.05
    skip_winner_penalty: float = 0.10        # missing a winner is mildly bad
    skip_loser_reward: float = 0.05          # avoiding a loser is mildly good
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
        # If we don't know the counterfactual (e.g. the trade ran past
        # the data we have), give a small zero-ish reward.
        if counterfactual is None:
            return 0.0
        if counterfactual.return_pct > 0:
            return -self.skip_winner_penalty
        return self.skip_loser_reward
