"""Policy scorers (baselines + sb3 wrapper) and validation utilities."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from rl_swing.domain import CandidateTrade, FeatureFrame, MarketBar, PortfolioState
from rl_swing.features.pipelines import ALL_FEATURE_NAMES
from rl_swing.rl.agents.baseline_scorers import (
    AlwaysTakePolicyScorer,
    NeverTakePolicyScorer,
    RandomPolicyScorer,
)
from rl_swing.rl.agents.dqn_scorer import DqnPolicyScorer
from rl_swing.rl.agents.ppo_scorer import PpoPolicyScorer
from rl_swing.rl.validation.baselines import buy_and_hold_return
from rl_swing.rl.validation.metrics import (
    annualized_sharpe,
    max_drawdown_from_returns,
    profit_factor,
    total_return,
    turnover_metric,
    validation_composite_score,
)
from rl_swing.rl.validation.walk_forward import (
    evaluate_policy,
    validate_from_experiment,
)


def _frame(symbol: str = "AAPL") -> FeatureFrame:
    full = {n: 0.0 for n in ALL_FEATURE_NAMES}
    return FeatureFrame(
        as_of=datetime(2024, 1, 2),
        symbol=symbol, feature_version="features_v001_core_daily",
        values=full, feature_names=ALL_FEATURE_NAMES,
        source_snapshot_id="s",
    )


def _candidate() -> CandidateTrade:
    return CandidateTrade(
        candidate_id="c1", as_of=datetime(2024, 1, 2), symbol="AAPL",
        strategy_id="momentum_20_60", direction="long",
        entry_timing="next_open", base_size_pct=0.10, max_holding_days=10,
        stop_rule_id=None, exit_rule_id="x", signal_strength=0.5,
    )


def _portfolio() -> PortfolioState:
    return PortfolioState(as_of=datetime(2024, 1, 2),
                          cash=100_000, equity=100_000)


# --- baseline scorers ----------------------------------------------------
def test_random_scorer_uses_seed_for_determinism():
    s1 = RandomPolicyScorer(seed=7)
    s2 = RandomPolicyScorer(seed=7)
    actions1 = [s1.score(_candidate(), _frame(), _portfolio()).action for _ in range(20)]
    actions2 = [s2.score(_candidate(), _frame(), _portfolio()).action for _ in range(20)]
    assert actions1 == actions2


def test_always_take_emits_configured_action():
    s = AlwaysTakePolicyScorer(action="take_50")
    d = s.score(_candidate(), _frame(), _portfolio())
    assert d.action == "take_50"
    assert d.target_size_pct == pytest.approx(0.05)


def test_never_take_emits_skip():
    s = NeverTakePolicyScorer()
    d = s.score(_candidate(), _frame(), _portfolio())
    assert d.action == "skip"
    assert d.target_size_pct == 0.0


# --- sb3 scorer happy/sad paths -----------------------------------------
def test_ppo_scorer_missing_artifact_raises(tmp_path: Path):
    s = PpoPolicyScorer(model_id="m", artifact_path=str(tmp_path / "absent.zip"))
    with pytest.raises(FileNotFoundError):
        s.score(_candidate(), _frame(), _portfolio())


def test_ppo_scorer_feature_version_mismatch_raises(tmp_path: Path):
    # Construct a frame with wrong feature version.
    bad_frame = FeatureFrame(
        as_of=datetime(2024, 1, 2),
        symbol="AAPL", feature_version="features_v999",
        values={n: 0.0 for n in ALL_FEATURE_NAMES},
        feature_names=ALL_FEATURE_NAMES,
        source_snapshot_id="s",
    )
    # We do NOT need an artifact to exercise this path: the version check
    # runs before _load(). Use a path that doesn't exist.
    s = PpoPolicyScorer(
        model_id="m", artifact_path=str(tmp_path / "x.zip"),
        feature_version="features_v001_core_daily",
    )
    with pytest.raises(RuntimeError):
        s.score(_candidate(), bad_frame, _portfolio())


def test_ppo_and_dqn_have_correct_algorithm_strings(tmp_path: Path):
    p = PpoPolicyScorer(model_id="m", artifact_path=str(tmp_path / "x.zip"))
    d = DqnPolicyScorer(model_id="m", artifact_path=str(tmp_path / "y.zip"))
    assert p.algorithm == "PPO"
    assert d.algorithm == "DQN"


def test_sb3_scorer_unknown_algorithm_raises(tmp_path: Path):
    from rl_swing.rl.agents.sb3_scorer import _Sb3Scorer
    artifact = tmp_path / "fake.zip"
    artifact.write_bytes(b"not a real model")
    s = _Sb3Scorer(model_id="m", artifact_path=str(artifact),
                   algorithm="BANDIT")
    with pytest.raises(ValueError):
        s._load()


# --- validation metrics --------------------------------------------------
def test_total_return_compounds():
    assert total_return([0.10, 0.10]) == pytest.approx(0.21)


def test_total_return_empty_returns_zero():
    assert total_return([]) == 0.0


def test_annualized_sharpe_with_zero_std_is_zero():
    assert annualized_sharpe([0.01, 0.01, 0.01], [10, 10, 10]) == 0.0


def test_annualized_sharpe_positive_for_winners():
    s = annualized_sharpe([0.02, 0.01, 0.02, 0.01], [10, 10, 10, 10])
    assert s > 0


def test_profit_factor_no_losses():
    assert profit_factor([0.05, 0.10]) > 0
    assert profit_factor([]) == 1.0


def test_profit_factor_balanced_winners_losers():
    pf = profit_factor([0.05, -0.05])
    assert pf == pytest.approx(1.0)


def test_max_drawdown_zero_for_monotone_returns():
    assert max_drawdown_from_returns([0.01, 0.02, 0.01]) == pytest.approx(0.0)


def test_max_drawdown_positive_for_decline():
    dd = max_drawdown_from_returns([0.10, -0.20, -0.10])
    assert dd > 0


def test_max_drawdown_empty():
    assert max_drawdown_from_returns([]) == 0.0


def test_turnover_metric_basic():
    assert turnover_metric([]) == 0.0
    assert turnover_metric(["take", "skip", "take"]) == pytest.approx(2 / 3)


def test_validation_composite_score_returns_breakdown():
    score, br = validation_composite_score(
        net_returns=[0.02, -0.01, 0.03],
        cost_bps=[5, 5, 5],
        holding_days=[5, 4, 6],
        rewards=[0.5, -0.2, 0.7],
        actions=["take", "take", "take"],
    )
    assert "n_total_return" in br["components"]
    assert isinstance(score, float)
    assert br["n_trades"] == 3


def test_validation_composite_score_no_trades_doesnt_crash():
    score, br = validation_composite_score(
        net_returns=[], cost_bps=[], holding_days=[],
        rewards=[], actions=[],
    )
    assert isinstance(score, float)
    assert br["n_trades"] == 0


# --- buy and hold --------------------------------------------------------
def test_buy_and_hold_return_basic():
    bars = [
        MarketBar(
            symbol="X", timestamp=datetime(2024, 1, 1) + (datetime(2024, 1, 2) - datetime(2024, 1, 1)) * i,
            timeframe="1d", open=100 * (1 + 0.01 * i), high=101, low=99,
            close=100 * (1 + 0.01 * i),
            volume=1e6, adjusted_close=100 * (1 + 0.01 * i), source="t",
        )
        for i in range(10)
    ]
    r = buy_and_hold_return(bars, "X", date(2024, 1, 1), date(2024, 1, 12))
    assert r > 0


def test_buy_and_hold_empty_returns_zero():
    assert buy_and_hold_return([], "X", date(2024, 1, 1), date(2024, 1, 31)) == 0.0


def test_buy_and_hold_zero_start_price_returns_zero():
    bars = [
        MarketBar(symbol="X", timestamp=datetime(2024, 1, 2), timeframe="1d",
                  open=0, high=0, low=0, close=0, volume=0,
                  adjusted_close=0, source="t"),
        MarketBar(symbol="X", timestamp=datetime(2024, 1, 3), timeframe="1d",
                  open=10, high=10, low=10, close=10, volume=1,
                  adjusted_close=10, source="t"),
    ]
    assert buy_and_hold_return(bars, "X", date(2024, 1, 1), date(2024, 1, 5)) == 0.0


# --- walk-forward harness -----------------------------------------------
def test_evaluate_policy_with_baseline_returns_summary(synthetic_bars,
                                                       feature_frames,
                                                       portfolio_state,
                                                       candidates):
    from rl_swing.rl.env.cost_model import EquityExecutionModel
    from rl_swing.rl.env.reward_model import RewardModel
    out = evaluate_policy(
        AlwaysTakePolicyScorer(model_id="bt", action="take_25"),
        synthetic_bars, candidates, feature_frames,
        cost_model=EquityExecutionModel(),
        reward_model=RewardModel(),
    )
    assert "validation_composite_score" in out
    assert out["model_id"] == "bt"


def test_evaluate_policy_handles_missing_model_artifact(synthetic_bars,
                                                       feature_frames,
                                                       candidates,
                                                       tmp_path):
    from rl_swing.rl.env.cost_model import EquityExecutionModel
    from rl_swing.rl.env.reward_model import RewardModel
    s = PpoPolicyScorer(model_id="m", artifact_path=str(tmp_path / "absent.zip"))
    out = evaluate_policy(
        s, synthetic_bars, candidates, feature_frames,
        cost_model=EquityExecutionModel(),
        reward_model=RewardModel(),
    )
    assert "error" in out


def test_validate_from_experiment_writes_report(tmp_path: Path):
    # A tiny synthetic universe + a tiny experiment file.
    universes_dir = Path("configs/universes")
    universes_dir.mkdir(parents=True, exist_ok=True)

    exp = tmp_path / "exp.yaml"
    exp.write_text(
        "experiment:\n"
        "  name: walkfwd_test\n"
        "  algorithm: PPO\n"
        "  policy: MlpPolicy\n"
        "  feature_pipeline: equities_features_v001\n"
        "  universe: synthetic\n"
        "  reward_config_version: r\n"
        "  train_start: '2018-01-01'\n  train_end: '2019-01-01'\n"
        "  validation_start: '2019-01-02'\n  validation_end: '2019-06-30'\n"
        "  test_start: '2019-07-01'\n  test_end: '2020-06-30'\n"
        "  total_timesteps_initial: 1000\n  total_timesteps_max: 1000\n"
        "  eval_interval_timesteps: 1000\n"
        "  early_stopping_patience_evaluations: 1\n"
        "  min_validation_delta: 0.0\n"
        "  save_best_only: true\n"
        "  seeds: [1]\n"
        "  hyperparams: {}\n"
        "  cost_model: {}\n"
        "  reward: {}\n"
        "  data_provider: synthetic_momentum\n"
        "  artifact_root: " + str(tmp_path) + "\n"
    )
    summary = validate_from_experiment(
        exp, report_dir=tmp_path / "reports", include_cost_stress=False,
    )
    assert summary["experiment"] == "walkfwd_test"
    assert (tmp_path / "reports").exists()


# ---------------------------------------------------------------------
# FIX-#78: guardrail against silent synthetic_momentum fallback for
# selector-class variants. Selectors are decision-grade; an unset
# data_provider must not silently produce synthetic numbers.
def test_fix78_guardrail_fires_for_selector_without_provider(tmp_path: Path):
    """selector_v002 YAML missing data_provider AND no caller override
    AND no allow_synthetic_validation=True -> RuntimeError."""
    import pytest

    exp = tmp_path / "exp.yaml"
    exp.write_text(
        "experiment:\n"
        "  name: guardrail_test\n"
        "  rl_variant: selector_v002\n"
        "  algorithm: PPO\n"
        "  policy: MlpPolicy\n"
        "  feature_pipeline: equities_features_v001\n"
        "  universe: synthetic\n"
        "  reward_config_version: r\n"
        "  train_start: '2018-01-01'\n  train_end: '2019-01-01'\n"
        "  validation_start: '2019-01-02'\n  validation_end: '2019-06-30'\n"
        "  test_start: '2019-07-01'\n  test_end: '2020-06-30'\n"
        "  total_timesteps_initial: 1000\n  total_timesteps_max: 1000\n"
        "  eval_interval_timesteps: 1000\n"
        "  early_stopping_patience_evaluations: 1\n"
        "  min_validation_delta: 0.0\n"
        "  save_best_only: true\n"
        "  seeds: [1]\n"
        "  hyperparams: {}\n"
        "  cost_model: {}\n"
        "  reward: {}\n"
        "  artifact_root: " + str(tmp_path) + "\n"
    )
    # No data_provider in YAML, no override -> guardrail must fire.
    with pytest.raises(RuntimeError, match="FIX-#78 guardrail"):
        validate_from_experiment(
            exp, report_dir=tmp_path / "reports", include_cost_stress=False,
        )


def test_fix78_guardrail_lets_filter_variant_default_to_synthetic(tmp_path: Path):
    """Filter (v1) variants are unaffected by the guardrail — their
    smoke tests use synthetic providers heavily and they're not
    decision-grade in the same way the selectors are."""
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        "experiment:\n"
        "  name: filter_synthetic_ok\n"
        "  rl_variant: filter_v001\n"  # filter, not selector
        "  algorithm: PPO\n"
        "  policy: MlpPolicy\n"
        "  feature_pipeline: equities_features_v001\n"
        "  universe: synthetic\n"
        "  reward_config_version: r\n"
        "  train_start: '2018-01-01'\n  train_end: '2019-01-01'\n"
        "  validation_start: '2019-01-02'\n  validation_end: '2019-06-30'\n"
        "  test_start: '2019-07-01'\n  test_end: '2020-06-30'\n"
        "  total_timesteps_initial: 1000\n  total_timesteps_max: 1000\n"
        "  eval_interval_timesteps: 1000\n"
        "  early_stopping_patience_evaluations: 1\n"
        "  min_validation_delta: 0.0\n"
        "  save_best_only: true\n"
        "  seeds: [1]\n"
        "  hyperparams: {}\n"
        "  cost_model: {}\n"
        "  reward: {}\n"
        "  artifact_root: " + str(tmp_path) + "\n"
        # NOTE: deliberately NO data_provider; should default to
        # synthetic without firing the guardrail (filter, not selector).
    )
    summary = validate_from_experiment(
        exp, report_dir=tmp_path / "reports", include_cost_stress=False,
    )
    assert summary["experiment"] == "filter_synthetic_ok"


def test_fix78_guardrail_allows_explicit_synthetic_via_flag(tmp_path: Path):
    """Selector variant + allow_synthetic_validation=True -> proceeds
    (with a logged warning, not tested here). Smoke tests / plumbing
    checks need this opt-in escape hatch."""
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        "experiment:\n"
        "  name: selector_smoke\n"
        "  rl_variant: selector_v002\n"
        "  algorithm: PPO\n"
        "  policy: MlpPolicy\n"
        "  feature_pipeline: equities_features_v001\n"
        "  universe: synthetic\n"
        "  reward_config_version: r\n"
        "  train_start: '2018-01-01'\n  train_end: '2019-01-01'\n"
        "  validation_start: '2019-01-02'\n  validation_end: '2019-06-30'\n"
        "  test_start: '2019-07-01'\n  test_end: '2020-06-30'\n"
        "  total_timesteps_initial: 1000\n  total_timesteps_max: 1000\n"
        "  eval_interval_timesteps: 1000\n"
        "  early_stopping_patience_evaluations: 1\n"
        "  min_validation_delta: 0.0\n"
        "  save_best_only: true\n"
        "  seeds: [1]\n"
        "  hyperparams: {}\n"
        "  cost_model: {}\n"
        "  reward: {}\n"
        "  artifact_root: " + str(tmp_path) + "\n"
    )
    summary = validate_from_experiment(
        exp, report_dir=tmp_path / "reports", include_cost_stress=False,
        allow_synthetic_validation=True,
    )
    assert summary["experiment"] == "selector_smoke"
