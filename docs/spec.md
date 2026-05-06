# RL Swing Trading Bot Specification

**Project:** Reinforcement-learning-assisted equity swing trading bot  
**Target execution venue:** Alpaca paper trading first, optional small live allocation later  
**Primary data sources:** WRDS, yfinance, Alpaca Market Data  
**Training style:** Offline reinforcement learning / simulated environment training, not live self-learning  
**Recommended initial role for RL:** Meta-policy for trade filtering and position sizing over rule-based candidate strategies  
**Status:** Draft implementation specification, refactored for extensibility with ports/adapters, domain contracts, component registry, and unified runtime modes

> **Important:** This is a technical design for research, simulation, and paper trading. It is not financial advice. Do not connect an RL policy directly to live capital without paper-trading evidence, strict risk limits, reconciliation, and manual model promotion.

---

## 1. Executive Summary

The goal is to build a swing-trading bot where traditional strategies such as momentum, mean reversion, breakout, trend-following, and regime filters generate candidate trade setups. A reinforcement learning policy then learns whether to:

1. take the trade,
2. skip the trade,
3. reduce or increase position size,
4. exit early,
5. stay flat.

This is safer and more explainable than letting an RL agent freely invent an end-to-end trading strategy from raw price data. The RL model should be treated as a controlled decision layer inside a broader trading system.

The bot should progress through these stages:

```text
Historical research
  -> backtest simulation
  -> RL training environment
  -> walk-forward validation
  -> shadow-mode signal generation
  -> Alpaca paper trading
  -> tiny live experiment only after manual approval
```

---

## 2. Goals and Non-Goals

### 2.1 Goals

- Build a research-grade RL swing-trading environment.
- Use WRDS as the preferred clean historical source when available.
- Use yfinance for fast prototyping and sanity checks.
- Use Alpaca Market Data for production-aligned daily signal generation.
- Use Alpaca Trading API for paper trading and later tightly controlled live execution.
- Maintain full auditability of:
  - features,
  - signals,
  - model versions,
  - proposed trades,
  - approved trades,
  - orders,
  - fills,
  - positions,
  - reconciliation breaks,
  - risk decisions.
- Prevent the RL model from bypassing risk controls.
- Compare RL decisions against simple rule-based baselines.
- Run in shadow mode before any paper/live execution.

### 2.2 Non-Goals

- No live self-learning loop.
- No unmanaged autonomous trading.
- No scalping or high-frequency execution.
- No automatic deployment of newly trained models.
- No live trading until the model has survived walk-forward testing and paper trading.
- No options, crypto, leveraged ETFs, penny stocks, or illiquid names in the MVP.
- No short selling in the first version.

---

## 3. Core Design Philosophy

The safest practical design is:

```text
Rule-based strategies generate candidates
        ↓
RL model evaluates candidate trade context
        ↓
Risk engine approves, rejects, or scales the order
        ↓
Execution engine submits orders through Alpaca
        ↓
Reconciliation engine verifies broker state
        ↓
Monitoring/reporting records results
```

The RL policy should initially learn the **when** and **how much**, not the full **what, when, how, and why**.

Recommended starting behavior:

```text
Candidate trade: Buy MU from momentum strategy
RL action: take at 50% size, skip, or take at full allowed size
Risk engine: final authority
Broker API: execution only
```

### 3.1 External Reference Bot Learnings Adapted for This Project

The public `zero-was-here/tradingbot` project is a deep-reinforcement-learning XAUUSD/gold trading bot using MetaTrader/MetaAPI, PPO/Dreamer, multi-timeframe features, macro/event inputs, realistic execution modeling, and risk overlays. The useful ideas should be adapted selectively for this equities/Alpaca goal.

Transferable ideas:

```text
1. Modular architecture: data -> features -> environment -> training -> evaluation -> execution -> monitoring.
2. Multi-timeframe context: slower strategic context plus faster timing context.
3. Feature families: technical, macro, calendar/event, microstructure/execution, and portfolio state.
4. PPO as the first production-style RL algorithm; Dreamer/world-model RL only after the PPO/DQN baseline is proven.
5. Conservative realistic-execution model that penalizes spread, slippage, volatility, liquidity, market impact, and event windows.
6. Explicit risk supervisor above the model: daily loss limits, drawdown guardrails, concentration limits, and dynamic position sizing.
7. Crisis validation: test the model on stress regimes instead of only average market periods.
8. Checkpointed Colab training workflow with saved artifacts, metrics, and reproducibility metadata.
9. Monitoring and production reports as first-class components, not afterthoughts.
```

Do **not** transfer directly:

```text
1. MetaTrader 5 or MetaAPI execution; replace with Alpaca paper/live APIs.
2. XAUUSD-specific feature assumptions; replace with equity, ETF, sector, macro, earnings, and liquidity features.
3. Minute-level base timeframe for the MVP; use daily bars first, then optionally add hourly context.
4. Aggressive return targets; use conservative validation gates and risk-adjusted outperformance versus baselines.
5. Full autonomy; keep RL as a trade filter / sizing layer until paper trading proves otherwise.
```

### 3.2 Research Learnings and Design Implications

The external literature supports the conservative architecture selected for this project: a modular DRL research pipeline, realistic execution costs, classical baselines, walk-forward validation, and RL as a filter/sizing layer before full autonomy.

| Research / Source | Applicable learning | Design implication for this project |
|---|---|---|
| FinRL full-stack DRL framework | A trading RL system benefits from clear separation of data, environment, agent, backtesting, and deployment layers, with reproducibility as a core principle. | Keep `data/`, `features/`, `strategies/`, `env/`, `training/`, `backtest/`, `execution/`, `monitoring/`, and `model_registry/` separate. Every training run must be reproducible from config, data snapshot, feature version, seed, and commit SHA. |
| Deep RL automated stock-trading ensemble work | PPO, A2C, and DDPG-style actor-critic agents can be compared and ensembled, but the ensemble idea should come after a single-agent baseline is understood. | Start with PPO and DQN. Add A2C/DDPG/SAC/TD3 or ensemble voting only after the PPO/DQN MVP beats baselines after costs. |
| Empirical DRL stock-trading analysis | Algorithm choice matters, and results should be compared across multiple markets, assets, and methods instead of relying on one favorable backtest. | Require multi-seed experiments, multiple symbols, multiple walk-forward years, and baseline comparisons before model promotion. |
| Reproducible RL-vs-classical-baselines trading study | PPO/DQN should be evaluated against momentum and mean-reversion baselines with transaction costs, slippage, market impact, experiment logs, and reproducible benchmarks. | The validation report must include RL vs. random policy, always-take, momentum, mean reversion, breakout, trend-following, buy-and-hold SPY/QQQ, and the current rule-based bot. |
| Stock evaluation / position-sizing RL work | RL is well suited to learning stock weighting or position sizing where supervised labels are ambiguous. | MVP action space should be `skip`, `25% size`, `50% size`, `100% size`, not unconstrained buy/sell/short. |
| Portfolio-management RL framework | Portfolio state and previous allocation memory matter when the agent controls weights across assets. | Add portfolio-vector state later: current positions, target weights, cash, exposure, days held, unrealized P&L, realized P&L, and drawdown. Do not start with full portfolio allocation in Phase 1. |
| Realistic market-impact / cost-model research | Fixed or negligible costs can make RL agents learn unrealistic trading behavior; nonlinear cost models can materially change algorithm rankings and turnover. | The simulator must include spread, slippage, liquidity, volatility, turnover, event-window, and market-impact penalties. Promotion must fail if performance collapses under stricter cost assumptions. |
| Reference `zero-was-here/tradingbot` architecture | Multi-timeframe features, macro/event context, risk overlays, crisis validation, Colab checkpoints, and live monitoring are useful concepts. | Adapt these ideas to equities and Alpaca: daily base timeframe first, weekly/hourly context later, macro/sector/earnings features, Alpaca-only execution, paper trading first, crisis validation required. |

#### 3.2.1 Practical Research-Backed Design Rules

```text
1. Treat RL as a decision layer, not a magic strategy generator.
2. Use rule-based strategies to create candidate trades first.
3. Let RL learn trade filtering and position sizing before giving it entry/exit autonomy.
4. Compare every RL model against simple classical baselines.
5. Penalize turnover, drawdown, and cost-sensitive behavior in the reward.
6. Run synthetic sanity tests before real-market training.
7. Run multi-seed walk-forward validation before paper trading.
8. Fail promotion if the edge disappears under realistic slippage/spread assumptions.
9. Keep execution, reconciliation, and risk controls outside the model.
10. Do not allow live self-learning.
```

#### 3.2.2 Research-Informed MVP Choice

The most defensible first experiment is not a full autonomous RL trader. It is:

```text
Rule-based candidate strategies
  -> RL trade filter / position-size selector
  -> strict cost-aware simulator
  -> walk-forward validation against classical baselines
  -> shadow mode
  -> Alpaca paper trading
```

Recommended MVP action space:

```text
0 = skip candidate trade
1 = take at 25% of allowed size
2 = take at 50% of allowed size
3 = take at 100% of allowed size
```

Recommended MVP baseline set:

```text
Random policy
Always-take candidate policy
Buy-and-hold SPY
Buy-and-hold QQQ
Momentum rule
Mean-reversion rule
Breakout rule
Trend-following rule
Existing rule-based bot strategy
```

Recommended MVP promotion principle:

```text
Promote only if RL improves validation and walk-forward risk-adjusted results after realistic costs.
Do not promote based only on training reward or one lucky backtest window.
```

---

## 4. Architecture Review and Refactored Target Architecture

### 4.1 What Needs to Be Torn Apart

The current design contains the right ingredients, but it is still too **implementation-shaped** instead of **contract-shaped**. The spec lists loaders, feature builders, strategies, an RL environment, Alpaca execution, risk, storage, and reconciliation, but it does not yet make the boundaries strict enough.

Main architectural weaknesses to correct:

```text
1. Data providers are too coupled to downstream feature/training code.
2. Strategy generation and RL scoring are conceptually separate, but the contract between them is not explicit enough.
3. Feature definitions, model inputs, and training artifacts need stronger versioning boundaries.
4. The execution stack is Alpaca-specific too early in the design.
5. Backtest, shadow, paper, and live modes need to use the same domain objects and decision pipeline.
6. The RL environment should not know about broker APIs, database schemas, or live runtime details.
7. Risk rules should be composable policies, not one giant `risk_engine.py`.
8. Reconciliation should be treated as a first-class state machine, not a report after execution.
9. Monitoring should consume domain events rather than scrape tables after the fact.
10. Configuration should select plug-ins/adapters rather than require code changes.
```

The refactor goal is:

```text
Make every major piece replaceable:
  yfinance -> WRDS -> Alpaca data
  PPO -> DQN -> SAC -> ensemble
  rule-based signals -> ML candidate generator
  Alpaca -> another broker
  SQLite -> Postgres
  daily bars -> hourly/daily hybrid
  trade filter -> position sizer -> portfolio allocator
```

### 4.2 Refactored Design Principle

Use a **ports-and-adapters** architecture:

```text
Core domain model and service contracts stay stable.
Adapters change when data vendors, brokers, algorithms, or storage engines change.
```

The core system should be organized around these stable concepts:

```text
MarketDataProvider
FeaturePipeline
CandidateStrategy
CandidateTrade
PolicyScorer
RiskPolicy
PortfolioTarget
BrokerAdapter
OrderManager
FillEvent
ReconciliationService
ExperimentTracker
ModelRegistry
EventBus
```

No component should call another component's internal implementation. Components communicate through typed domain objects and interfaces.

---

### 4.3 Target Runtime Architecture

```text
+-----------------------------------------------------------------------+
|                            Configuration                              |
| universe.yaml | features.yaml | strategies.yaml | policy.yaml         |
| risk.yaml     | broker.yaml   | storage.yaml    | runtime.yaml        |
+-----------------------------------------------------------------------+
                                  |
                                  v
+-----------------------------------------------------------------------+
|                            Domain Core                                |
| MarketBar | FeatureFrame | CandidateTrade | PolicyDecision        |
| RiskDecision | PortfolioTarget | OrderIntent | BrokerOrder          |
| FillEvent | PositionSnapshot | ReconciliationBreak | AuditEvent          |
+-----------------------------------------------------------------------+
                                  |
          +-----------------------+-----------------------+
          |                       |                       |
          v                       v                       v
+-------------------+   +-------------------+   +-----------------------+
| Research Runtime  |   | Shadow Runtime    |   | Paper/Live Runtime    |
+-------------------+   +-------------------+   +-----------------------+
| historical data   |   | latest data       |   | latest data           |
| feature build     |   | feature build     |   | feature build         |
| candidate replay  |   | candidate scoring |   | candidate scoring     |
| RL environment    |   | no orders         |   | risk + order routing  |
| validation        |   | outcome tracking  |   | reconciliation        |
+-------------------+   +-------------------+   +-----------------------+
          |                       |                       |
          +-----------------------+-----------------------+
                                  |
                                  v
+-----------------------------------------------------------------------+
|                      Storage / Registry / Observability               |
| bars | features | candidates | decisions | orders | fills | positions |
| models | experiments | reports | audit_events | reconciliation_breaks    |
+-----------------------------------------------------------------------+
```

Key design rule:

```text
Research, shadow, paper, and live modes must run through the same decision pipeline.
Only the broker adapter changes behavior:
  research = simulated broker
  shadow = no-op broker
  paper = Alpaca paper broker
  live = Alpaca live broker, gated by manual approval
```

---

### 4.4 Core Domain Objects

The system should define stable dataclasses or Pydantic models for all handoffs. This is more important than the first RL algorithm.

#### 4.4.1 MarketBar

```python
@dataclass(frozen=True)
class MarketBar:
    symbol: str
    timestamp: datetime
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: float | None = None
    source: str = "unknown"
    quality_flags: tuple[str, ...] = ()
```

#### 4.4.2 FeatureFrame

```python
@dataclass(frozen=True)
class FeatureFrame:
    as_of: datetime
    symbol: str
    feature_version: str
    values: dict[str, float]
    feature_names: tuple[str, ...]
    source_snapshot_id: str
```

Important rule:

```text
A FeatureFrame must only contain information available as of `as_of`.
No future adjusted data, future event labels, or post-close leakage can enter the observation.
```

#### 4.4.3 CandidateTrade

```python
@dataclass(frozen=True)
class CandidateTrade:
    candidate_id: str
    as_of: datetime
    symbol: str
    strategy_id: str
    direction: Literal["long", "short"]
    entry_timing: Literal["next_open", "next_close", "limit"]
    base_size_pct: float
    max_holding_days: int
    stop_rule_id: str | None
    exit_rule_id: str
    signal_strength: float
    metadata: dict[str, Any]
```

MVP rule:

```text
Only `direction = long` is enabled in the first paper-trading version.
Shorting remains an interface capability but is disabled by risk config.
```

#### 4.4.4 PolicyDecision

```python
@dataclass(frozen=True)
class PolicyDecision:
    decision_id: str
    candidate_id: str
    as_of: datetime
    model_id: str
    action: Literal["skip", "take_25", "take_50", "take_100"]
    confidence: float | None
    target_size_pct: float
    raw_action: int | float
    observation_hash: str
    explanation: dict[str, Any]
```

Explanation can be shallow at first:

```text
top_features_by_sensitivity
strategy_id
regime_bucket
risk_bucket
policy_logits/probabilities
```

#### 4.4.5 RiskDecision

```python
@dataclass(frozen=True)
class RiskDecision:
    decision_id: str
    candidate_id: str
    policy_decision_id: str
    approved: bool
    final_size_pct: float
    blocked_reasons: tuple[str, ...]
    applied_rules: tuple[str, ...]
```

Risk is the final authority:

```text
PolicyDecision proposes.
RiskDecision disposes.
```

#### 4.4.6 OrderIntent

```python
@dataclass(frozen=True)
class OrderIntent:
    intent_id: str
    as_of: datetime
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    order_type: Literal["market", "limit", "stop", "stop_limit"]
    time_in_force: str
    limit_price: float | None
    source_decision_id: str
    environment: Literal["backtest", "shadow", "paper", "live"]
```

#### 4.4.7 PositionSnapshot

```python
@dataclass(frozen=True)
class PositionSnapshot:
    as_of: datetime
    source: Literal["simulated", "alpaca", "internal"]
    symbol: str
    quantity: float
    market_value: float
    avg_entry_price: float | None
    unrealized_pnl: float | None
```

---

### 4.5 Stable Interface Contracts

Define contracts before building the implementation. The interfaces below are intentionally small so components remain swappable.

#### 4.5.1 MarketDataProvider

```python
class MarketDataProvider(Protocol):
    provider_id: str

    def get_bars(
        self,
        symbols: list[str],
        start: date,
        end: date,
        timeframe: str,
        adjusted: bool,
    ) -> Iterable[MarketBar]: ...

    def get_snapshot_id(self) -> str: ...
```

Adapters:

```text
yfinance provider
WRDS provider
Alpaca historical provider
CSV/parquet provider
mock/synthetic provider
```

#### 4.5.2 FeaturePipeline

```python
class FeaturePipeline(Protocol):
    feature_version: str

    def build(
        self,
        bars: Iterable[MarketBar],
        context: dict[str, Any],
    ) -> Iterable[FeatureFrame]: ...
```

Rules:

```text
Feature pipeline must be deterministic.
Feature version must be part of every model artifact.
Feature scaling must be fit on training windows only.
```

#### 4.5.3 CandidateStrategy

```python
class CandidateStrategy(Protocol):
    strategy_id: str

    def generate(
        self,
        features: Iterable[FeatureFrame],
        portfolio_state: "PortfolioState",
    ) -> Iterable[CandidateTrade]: ...
```

Adapters:

```text
momentum strategy
mean-reversion strategy
breakout strategy
trend-following strategy
volatility contraction strategy
future ML candidate generator
```

#### 4.5.4 PolicyScorer

```python
class PolicyScorer(Protocol):
    model_id: str

    def score(
        self,
        candidate: CandidateTrade,
        features: FeatureFrame,
        portfolio_state: "PortfolioState",
    ) -> PolicyDecision: ...
```

Adapters:

```text
random policy
always-take policy
rule policy
PPO policy
DQN policy
ensemble policy
manual override policy
```

This makes baseline comparison cheap. Baselines become policies, not separate backtest paths.

#### 4.5.5 RiskPolicy

```python
class RiskPolicy(Protocol):
    rule_id: str

    def evaluate(
        self,
        candidate: CandidateTrade,
        policy_decision: PolicyDecision,
        portfolio_state: "PortfolioState",
        market_state: "MarketState",
    ) -> "RiskRuleResult": ...
```

A `RiskEngine` simply composes many `RiskPolicy` rules:

```text
MaxSinglePositionPolicy
MaxPortfolioExposurePolicy
MaxDailyLossPolicy
EarningsBlackoutPolicy
LiquidityPolicy
VolatilityPolicy
DuplicateOrderPolicy
KillSwitchPolicy
LiveTradingApprovalPolicy
```

#### 4.5.6 BrokerAdapter

```python
class BrokerAdapter(Protocol):
    broker_id: str
    environment: Literal["backtest", "shadow", "paper", "live"]

    def submit_order(self, intent: OrderIntent) -> "BrokerOrder": ...
    def cancel_order(self, broker_order_id: str) -> None: ...
    def list_open_orders(self) -> list["BrokerOrder"]: ...
    def list_positions(self) -> list[PositionSnapshot]: ...
    def get_account_snapshot(self) -> "AccountSnapshot": ...
```

Adapters:

```text
SimulatedBrokerAdapter
NoOpShadowBrokerAdapter
AlpacaPaperBrokerAdapter
AlpacaLiveBrokerAdapter
```

No training or feature code should import Alpaca.

#### 4.5.7 ReconciliationService

```python
class ReconciliationService(Protocol):
    def reconcile(
        self,
        internal_positions: list[PositionSnapshot],
        broker_positions: list[PositionSnapshot],
        open_orders: list["BrokerOrder"],
    ) -> list["ReconciliationBreak"]: ...
```

Reconciliation should be runnable in all modes:

```text
backtest: simulated state vs expected ledger
shadow: internal no-order state only
paper/live: internal ledger vs Alpaca broker state
```

---

### 4.6 Decision Pipeline Contract

Every runtime mode should follow the same pipeline:

```text
1. Load runtime config.
2. Resolve universe.
3. Load market data through MarketDataProvider.
4. Build FeatureFrame objects through FeaturePipeline.
5. Generate CandidateTrade objects through CandidateStrategy plug-ins.
6. Score each candidate through PolicyScorer.
7. Pass policy decisions through composed RiskPolicy rules.
8. Convert approved risk decisions into OrderIntent objects.
9. Submit OrderIntent objects to BrokerAdapter.
10. Persist all domain events.
11. Reconcile positions/orders/cash.
12. Emit monitoring alerts and daily report.
```

This is the most important extensibility requirement:

```text
Backtest, shadow, paper, and live must not have separate decision logic.
They differ only by data window, broker adapter, and enabled risk policies.
```

---

### 4.7 Runtime Modes

| Mode | Data | Broker adapter | Orders? | Purpose |
|---|---|---|---:|---|
| `research` | historical | simulated | simulated | train, validate, backtest |
| `shadow` | latest historical/live-aligned | no-op | no | generate signals and measure outcomes |
| `paper` | Alpaca/yfinance-aligned | Alpaca paper | yes, fake money | execution and reconciliation testing |
| `live_guarded` | Alpaca | Alpaca live | yes, tiny size | tightly capped live experiment |

Mode-specific changes must be config-driven:

```yaml
runtime:
  mode: paper
  data_provider: alpaca_historical
  broker_adapter: alpaca_paper
  policy_scorer: ppo_model
  risk_profile: conservative_paper
```

---

### 4.8 Event-Driven Audit Trail

The system should emit immutable domain events. This makes replay, debugging, reporting, and monitoring much easier.

Core events:

```text
MarketDataLoaded
FeaturesBuilt
CandidateGenerated
PolicyScored
RiskEvaluated
OrderIntentCreated
OrderSubmitted
OrderFilled
OrderRejected
PositionUpdated
ReconciliationCompleted
ReconciliationBreakDetected
RiskLimitBreached
ModelPromoted
KillSwitchActivated
```

Event schema:

```python
@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    event_type: str
    timestamp: datetime
    correlation_id: str
    payload: dict[str, Any]
    run_id: str
    environment: str
```

Use `correlation_id` to tie the whole chain together:

```text
candidate_id -> policy_decision_id -> risk_decision_id -> order_intent_id -> broker_order_id -> fill_id
```

---

### 4.9 Extensible Feature Architecture

Feature groups should be plug-ins with declared dependencies, timeframes, and leakage constraints.

```yaml
feature_groups:
  technical_daily:
    enabled: true
    timeframe: 1d
    depends_on: [daily_bars]
    leakage_safe: true
  weekly_context:
    enabled: false
    timeframe: 1w
    depends_on: [daily_bars]
    leakage_safe: true
  macro_regime:
    enabled: true
    depends_on: [vix, yields, dxy, spy_context]
    publication_lag_days: 1
  earnings_events:
    enabled: true
    depends_on: [earnings_calendar]
    use_known_before_asof_only: true
  sentiment:
    enabled: false
    research_only: true
```

Feature rules:

```text
1. Every feature group has a version.
2. Every feature group declares source data and lag assumptions.
3. Every model stores the exact feature group list used at training time.
4. Feature ablation should run by toggling feature groups, not editing code.
5. A feature can be promoted from research-only to production only after leakage tests pass.
```

---

### 4.10 Extensible RL Environment Architecture

The RL environment should be decomposed into replaceable pieces:

```text
EpisodeSampler
ObservationBuilder
ActionMapper
RewardModel
CostModel
ExecutionSimulator
PortfolioSimulator
TerminationRule
```

Instead of one large `SwingTradingEnv`, build it as:

```python
class SwingTradingEnv(gym.Env):
    def __init__(
        self,
        episode_sampler: EpisodeSampler,
        observation_builder: ObservationBuilder,
        action_mapper: ActionMapper,
        reward_model: RewardModel,
        execution_simulator: ExecutionSimulator,
        portfolio_simulator: PortfolioSimulator,
        termination_rules: list[TerminationRule],
    ):
        ...
```

This allows the same environment shell to support:

```text
trade filter mode
position sizing mode
exit timing mode
portfolio allocation mode
long-only mode
long/short mode
single-symbol mode
multi-symbol mode
```

MVP environment mode:

```text
mode = trade_filter
input = CandidateTrade + FeatureFrame + PortfolioState
action = skip / take_25 / take_50 / take_100
reward = net trade outcome after cost/risk penalties
```

Later environment modes:

```text
mode = exit_policy
mode = position_sizer
mode = strategy_allocator
mode = portfolio_allocator
```

---

### 4.11 Extensible Strategy Architecture

Do not make momentum, mean reversion, and breakout special cases. They are all `CandidateStrategy` plug-ins.

```yaml
strategies:
  momentum_20_60:
    class: rl_swing.strategies.momentum.MomentumStrategy
    enabled: true
    params:
      lookback_short: 20
      lookback_long: 60
      min_relative_strength: 0.65
  mean_reversion_rsi:
    class: rl_swing.strategies.mean_reversion.RsiMeanReversionStrategy
    enabled: true
    params:
      rsi_window: 5
      rsi_threshold: 25
  breakout_20d:
    class: rl_swing.strategies.breakout.BreakoutStrategy
    enabled: true
    params:
      breakout_lookback: 20
```

Each strategy should expose:

```text
strategy_id
feature_dependencies
candidate_generation_config
exit_rule
baseline_policy
```

This lets the validation engine report:

```text
RL improvement by source strategy
RL skipped-trade quality by strategy
RL sizing behavior by strategy
strategy contribution to drawdown
```

---

### 4.12 Extensible Risk Architecture

Risk should be a policy stack, not a monolith.

```yaml
risk_profile:
  name: conservative_paper_v001
  policies:
    - class: MaxSinglePositionPolicy
      params: {max_pct: 0.10}
    - class: MaxPortfolioExposurePolicy
      params: {max_pct: 0.50}
    - class: MaxDailyLossPolicy
      params: {max_pct: 0.01}
    - class: EarningsBlackoutPolicy
      params: {days_before: 1, days_after: 1}
    - class: LiquidityPolicy
      params: {min_avg_dollar_volume: 50000000}
    - class: DuplicateOrderPolicy
      params: {}
    - class: KillSwitchPolicy
      params: {}
```

Risk output should include explainability:

```text
approved = false
blocked_reasons = ["earnings_blackout", "max_daily_new_positions"]
applied_rules = ["MaxSinglePositionPolicy", "EarningsBlackoutPolicy"]
```

---

### 4.13 Extensible Storage Architecture

Storage should use repository interfaces so SQLite, DuckDB, Postgres, and parquet can be swapped.

```python
class CandidateRepository(Protocol):
    def save_candidates(self, candidates: list[CandidateTrade]) -> None: ...
    def load_candidates(self, run_id: str) -> list[CandidateTrade]: ...

class DecisionRepository(Protocol):
    def save_policy_decisions(self, decisions: list[PolicyDecision]) -> None: ...
    def load_policy_decisions(self, run_id: str) -> list[PolicyDecision]: ...
```

Recommended storage split:

```text
Columnar/parquet: raw/staged bars, feature matrices, backtest outputs
Relational DB: candidates, decisions, orders, fills, positions, reconciliation, audit events
Model registry: model artifacts, configs, scalers, reports
```

Do not store giant feature matrices only in the relational DB. Keep large numeric data columnar and use hashes/snapshot IDs in relational tables.

---

### 4.14 Refactored Deployment Architecture

Deployment should not assume one long-running script that does everything.

Recommended jobs:

```text
data_ingest_job
feature_build_job
candidate_generation_job
policy_scoring_job
risk_and_order_intent_job
broker_execution_job
fill_monitor_job
reconciliation_job
daily_report_job
```

MVP can run these in one CLI command, but internally they should be separable.

```bash
rl-swing run daily --mode shadow --config configs/runtime_shadow.yaml
rl-swing run daily --mode paper --config configs/runtime_paper.yaml
rl-swing train --config configs/experiments/ppo_filter_v001.yaml
rl-swing validate --model-id ppo_filter_v001
rl-swing reconcile --mode paper
```

---

## 5. Refactored Repository Structure

The repository should be organized around domain contracts and adapters, not around individual scripts.

```text
trading-bot2/
  rl_swing/
    README.md
    pyproject.toml

    configs/
      runtime/
        research.yaml
        shadow.yaml
        paper.yaml
        live_guarded.yaml
      components/
        data_providers.yaml
        feature_groups.yaml
        strategies.yaml
        policies.yaml
        risk_profiles.yaml
        broker_adapters.yaml
        storage.yaml
      experiments/
        ppo_filter_v001.yaml
        dqn_filter_v001.yaml
        ablation_features_v001.yaml
      universes/
        starter_equities.yaml
        liquid_etfs.yaml
        semis_watchlist.yaml

    src/rl_swing/
      __init__.py

      domain/
        market.py              # MarketBar, MarketSnapshot
        features.py            # FeatureFrame, FeatureSnapshot
        candidates.py          # CandidateTrade
        decisions.py           # PolicyDecision, RiskDecision
        orders.py              # OrderIntent, BrokerOrder, FillEvent
        portfolio.py           # PortfolioState, PositionSnapshot, AccountSnapshot
        reconciliation.py      # ReconciliationBreak
        events.py              # AuditEvent, EventType

      ports/
        market_data.py         # MarketDataProvider Protocol
        feature_pipeline.py    # FeaturePipeline Protocol
        strategy.py            # CandidateStrategy Protocol
        policy_scorer.py       # PolicyScorer Protocol
        risk_policy.py         # RiskPolicy Protocol
        broker.py              # BrokerAdapter Protocol
        storage.py             # Repository Protocols
        event_bus.py           # EventBus Protocol
        model_registry.py      # ModelRegistry Protocol

      adapters/
        data/
          yfinance_provider.py
          wrds_provider.py
          alpaca_data_provider.py
          parquet_provider.py
          synthetic_provider.py
        broker/
          simulated_broker.py
          noop_shadow_broker.py
          alpaca_paper_broker.py
          alpaca_live_broker.py
        storage/
          sqlite_repositories.py
          postgres_repositories.py
          parquet_feature_store.py
        notifications/
          discord_notifier.py
          email_notifier.py

      features/
        pipelines.py
        technical.py
        regime.py
        macro.py
        calendar_events.py
        earnings.py
        liquidity.py
        normalization.py
        leakage_checks.py

      strategies/
        momentum.py
        mean_reversion.py
        breakout.py
        trend_following.py
        volatility_contraction.py
        aggregator.py

      rl/
        env/
          swing_env.py
          episode_sampler.py
          observation_builder.py
          action_mapper.py
          reward_model.py
          cost_model.py
          execution_simulator.py
          portfolio_simulator.py
          termination.py
        agents/
          ppo_scorer.py
          dqn_scorer.py
          ensemble_scorer.py
          baseline_scorers.py
        training/
          trainer.py
          callbacks.py
          colab_entrypoint.py
          seed_runner.py
          hyperparams.py
        validation/
          walk_forward.py
          baselines.py
          crisis_validation.py
          ablation.py
          metrics.py

      services/
        data_ingest_service.py
        feature_build_service.py
        candidate_service.py
        policy_service.py
        risk_service.py
        order_intent_service.py
        execution_service.py
        reconciliation_service.py
        reporting_service.py
        model_promotion_service.py

      runtime/
        pipeline.py
        dependency_container.py
        scheduler.py
        cli.py
        modes.py

      risk/
        policies.py
        profiles.py
        kill_switch.py
        live_approval.py

      reporting/
        daily_report.py
        training_report.py
        validation_report.py
        paper_trading_report.py

    notebooks/
      01_data_exploration.ipynb
      02_baseline_strategies.ipynb
      03_environment_debug.ipynb
      04_colab_training.ipynb
      05_walk_forward_review.ipynb

    tests/
      unit/
        test_domain_models.py
        test_feature_leakage.py
        test_candidate_generation.py
        test_reward_model.py
        test_risk_policies.py
      integration/
        test_research_pipeline.py
        test_shadow_pipeline.py
        test_paper_pipeline_with_mock_alpaca.py
        test_reconciliation.py
      fixtures/
        synthetic_momentum_market.parquet
        synthetic_mean_reversion_market.parquet
        synthetic_random_market.parquet
```

### 5.1 Why This Structure Is More Adaptable

| Change needed | Old-style impact | Refactored impact |
|---|---|---|
| Add WRDS | Risk of touching training/data code | Add `WrdsProvider` adapter and config entry |
| Replace Alpaca | Execution rewrite | Add new `BrokerAdapter` |
| Try DQN instead of PPO | Training-specific changes | Add `DqnPolicyScorer`, reuse validation pipeline |
| Add hourly context | Feature/env coupling risk | Add feature group and observation config |
| Add new strategy | May alter signal aggregator | Add `CandidateStrategy` plug-in |
| Add new risk rule | Modify monolith | Add `RiskPolicy` class and config entry |
| Add paper baseline account | Runtime special case | Add broker config/profile |
| Move SQLite to Postgres | SQL churn everywhere | Swap repository adapter |

### 5.2 Architecture Decision Records

Add lightweight ADRs for major choices:

```text
docs/adr/
  0001-use-ports-and-adapters.md
  0002-rl-as-trade-filter-first.md
  0003-alpaca-as-broker-adapter.md
  0004-parquet-plus-relational-storage.md
  0005-no-live-self-learning.md
  0006-research-shadow-paper-live-same-pipeline.md
```

Every significant architectural pivot should create or update an ADR.

---

## 6. Data Layer

### 6.1 Data Sources

#### WRDS

Primary use:

- Research-grade historical daily data.
- Survivorship-aware universe construction when using CRSP-like data.
- Corporate action adjusted returns.
- Fundamentals later, if desired.
- Index membership and delisting information, depending on available subscriptions.

Expected access method:

```python
import wrds

conn = wrds.Connection()
libraries = conn.list_libraries()
```

WRDS should be preferred for serious backtesting because it can reduce survivorship and corporate-action bias when the correct datasets are available.

#### yfinance

Primary use:

- Rapid prototyping.
- Early feature experiments.
- Sanity checks.
- Debugging the RL environment before WRDS integration is complete.

Caution:

- Do not treat yfinance as the final source of truth for production-grade research.
- Use it to validate mechanics, not to approve a strategy for live trading.

#### Alpaca Market Data

Primary use:

- Production-aligned daily signal generation.
- Paper/live data feed alignment.
- Historical bars when WRDS is unavailable or when matching Alpaca execution context.

Alpaca historical bars should be used for final paper-trading alignment because the same vendor context will drive live/paper decisions.

---

### 6.2 Data Frequency

MVP:

```text
Daily adjusted OHLCV bars
```

Later:

```text
Hourly bars
Daily + hourly multi-timeframe features
Intraday risk monitoring
```

Do not start with minute-level trading. This is a swing bot.

---

### 6.3 Data Universe

MVP universe:

```text
SPY
QQQ
IWM
XLK
SMH
SOXX
GLD
NVDA
AMD
MU
MSFT
AAPL
AMZN
AVGO
ANET
GEV
VRT
```

Rules:

- Highly liquid only.
- No penny stocks.
- No illiquid stocks.
- No leveraged ETFs initially.
- No shorting initially.
- No stocks with insufficient history.
- Avoid trading around earnings in early phases.

Later universe expansion:

```text
Top 100–500 liquid U.S. equities
Sector ETFs
Quality-filtered large/mid-cap universe
```

---

### 6.4 Required Data Fields

For each symbol/date:

```text
symbol
date
open
high
low
close
adjusted_close
volume
vwap, optional
dividend, optional
split_factor, optional
source
ingested_at
data_quality_status
```

Derived return fields:

```text
return_1d
return_3d
return_5d
return_10d
return_20d
return_60d
gap_return
intraday_return
overnight_return
```

---

### 6.5 Corporate Actions and Adjustments

The system must explicitly track whether data is:

```text
raw
split-adjusted
dividend-adjusted
total-return adjusted
```

Rules:

- Backtests should use adjusted prices for signal and return calculations unless testing a raw-price execution-specific feature.
- Execution must use current real prices from Alpaca.
- Model training features must be generated consistently across all periods.
- Corporate actions must be logged as data events.
- Any restatement of historical prices should invalidate affected feature/model runs.

---

### 6.6 Data Quality Checks

Checks:

```text
Missing bars
Zero or negative prices
Extreme one-day returns
Volume equal to zero
Duplicate symbol/date rows
Unexpected holidays
Adjusted close discontinuities
Bad split adjustment
Timestamp mismatches
Source disagreement between WRDS/yfinance/Alpaca
```

Quality statuses:

```text
PASS
WARN
FAIL
MANUAL_REVIEW
```

A failed data-quality check must block training and paper/live signal generation for affected symbols.

---

## 7. Feature Engineering

### 7.1 Technical Features

Momentum:

```text
return_5d
return_10d
return_20d
return_60d
return_120d
relative_strength_vs_spy
relative_strength_vs_sector
```

Trend:

```text
close_vs_sma_10
close_vs_sma_20
close_vs_sma_50
close_vs_sma_200
sma_20_vs_sma_50
sma_50_vs_sma_200
```

Mean reversion:

```text
rsi_2
rsi_5
rsi_14
zscore_close_20
distance_from_20d_high
distance_from_20d_low
```

Volatility:

```text
atr_14
atr_pct
realized_vol_10
realized_vol_20
volatility_percentile_60
```

Volume:

```text
relative_volume_20
volume_zscore_20
dollar_volume
```

Market regime:

```text
spy_above_sma_50
spy_above_sma_200
qqq_above_sma_50
market_return_20d
market_volatility_20d
sector_relative_strength
```

Portfolio/risk state:

```text
current_position
days_in_position
unrealized_pnl_pct
realized_pnl_20d
current_drawdown
portfolio_exposure
available_risk_budget
```

---

### 7.2 Strategy Signal Features

Each rule-based strategy should produce numeric and categorical outputs.

Example:

```text
momentum_signal: -1, 0, 1
momentum_score: 0.0 to 1.0
mean_reversion_signal: -1, 0, 1
mean_reversion_score: 0.0 to 1.0
breakout_signal: -1, 0, 1
breakout_score: 0.0 to 1.0
trend_signal: -1, 0, 1
trend_score: 0.0 to 1.0
```

The RL agent then learns when those signals are trustworthy.

---

### 7.3 Feature Normalization

Rules:

- Fit scalers on training data only.
- Save scalers with model artifacts.
- Apply the exact same scaler in backtest, paper, and live modes.
- Never use future data in feature normalization.
- Use rolling z-scores where practical.

Artifacts:

```text
feature_config_v001.yaml
scaler_v001.pkl
feature_schema_v001.json
```

### 7.4 Repo-Inspired Feature Stack for Equities

The reference bot uses a large multi-source feature set. For this project, the same architectural idea should be used, but with an equity-appropriate and phased feature plan. Do not jump straight to 150+ features in the first version; feature bloat can make the RL model appear smart while actually overfitting.

Recommended feature tiers:

```text
features_v001_core_daily:
  40-60 features
  daily bars only
  technical + strategy-signal + portfolio state

features_v002_regime_event:
  70-100 features
  add macro regime, sector ETF context, VIX/yield/DXY proxies, earnings/calendar flags

features_v003_multitimeframe:
  100-150 features
  add weekly context and optional hourly context for entry timing

features_v004_research_only:
  150+ features
  only if v001-v003 prove stable under walk-forward validation
```

Equity adaptation of the reference bot's feature families:

| Reference idea | Equity/Alpaca adaptation | MVP priority |
|---|---|---:|
| M5/M15/H1/H4/D1 multi-timeframe features | Daily base features; weekly context; optional hourly timing later | Medium |
| Gold macro features: DXY, US10Y, VIX, oil, Bitcoin, silver, GLD | Equity regime features: SPY/QQQ/IWM trend, sector ETF strength, VIX, 10Y yield, DXY, GLD/oil as optional macro context | High |
| Economic calendar events: NFP, CPI, FOMC, GDP | Macro event windows plus equity-specific earnings blackout/proximity flags | High |
| Market microstructure | Spread/liquidity/slippage proxies using dollar volume, ATR, bid/ask where available, and Alpaca quotes later | Medium |
| Session-based patterns | Not required for daily MVP; optional for hourly extension | Low |
| Optional sentiment | Keep separate from MVP; add later as a separate signal source | Low |

### 7.5 Macro, Calendar, and Event Features

Add a dedicated event-feature builder. These features should be known as of the decision date and must not leak future event outcomes.

Macro/regime candidates:

```text
spy_return_20d
qqq_return_20d
iwm_return_20d
sector_etf_return_20d
sector_relative_strength_vs_spy
spy_above_sma_50
spy_above_sma_200
qqq_above_sma_50
vix_level
vix_percentile_60
10y_yield_change_20d
dxy_change_20d
risk_on_score
risk_off_score
```

Event candidates:

```text
days_to_fomc
days_since_fomc
is_fomc_week
is_cpi_week
is_nfp_week
days_to_earnings
days_since_earnings
earnings_within_5_trading_days
earnings_blackout_flag
high_impact_macro_event_window
```

Rules:

```text
No event outcome leakage.
No future earnings surprise information.
No forward-filled macro value unless it was actually known by the decision timestamp.
For MVP, use event flags mostly for risk reduction and trade skipping, not alpha generation.
```

### 7.6 Feature Explosion Guardrails

The reference bot emphasizes a very large feature set. This project should treat that as a roadmap, not a starting point.

Feature acceptance rules:

```text
Every feature must have a name, definition, source, lookback, timestamp rule, and leakage check.
Every new feature version must produce a feature importance / ablation report.
A feature group cannot be promoted unless it improves validation, not only training reward.
If validation gets worse after adding a feature group, revert the group.
Feature count should grow only after the environment and reward pass synthetic tests.
```

---

## 8. Strategy Signal Layer

The strategy layer generates candidate trades. RL should not initially choose symbols from the entire universe without structure.

### 8.1 Candidate Strategies

#### Momentum

Candidate long when:

```text
20d/60d return strong
price above SMA 50/200
relative strength vs SPY positive
volume confirmation present
```

#### Mean Reversion

Candidate long when:

```text
RSI low
price stretched below short-term average
longer-term trend still intact
market regime not hostile
```

#### Breakout

Candidate long when:

```text
price breaks above 20d/55d high
volume confirms
market/sector regime supportive
```

#### Trend Following

Candidate long when:

```text
price above SMA 50/200
pullback resolves upward
volatility acceptable
```

---

### 8.2 Candidate Trade Object

```json
{
  "candidate_id": "2026-05-06_MU_momentum_v001",
  "as_of_date": "2026-05-06",
  "symbol": "MU",
  "strategy_name": "momentum",
  "direction": "long",
  "raw_signal": 1,
  "signal_score": 0.83,
  "suggested_entry": "next_open",
  "default_holding_period_days": 10,
  "default_stop_atr": 2.0,
  "default_target_atr": 4.0,
  "feature_vector_id": "abc123",
  "created_at": "2026-05-06T18:00:00"
}
```

---

## 9. Reinforcement Learning Environment

### 9.1 Environment Type

Use a custom Gymnasium environment.

Recommended class:

```python
class SwingTradingEnv(gymnasium.Env):
    def reset(self, seed=None, options=None):
        ...

    def step(self, action):
        ...
```

The environment should simulate daily trading decisions using historical data.

---

### 9.2 Environment Modes

#### Mode A: Trade Filter MVP

Each episode presents candidate trades.

Observation:

```text
candidate trade features
strategy signal features
market regime
portfolio state
recent strategy performance
```

Action:

```text
0 = skip
1 = take small
2 = take normal
3 = take large, still risk-capped
```

This is the recommended first implementation.

#### Mode B: Position Manager

The agent manages an existing position.

Action:

```text
0 = exit
1 = hold
2 = add small
3 = reduce
```

#### Mode C: Portfolio Allocator

The agent chooses target weights.

Action:

```text
target weight per candidate symbol
cash allocation
```

This should come later.

---

### 9.3 Initial Action Space

For MVP:

```text
0 = skip trade
1 = take 25% of default size
2 = take 50% of default size
3 = take 100% of default size
```

The risk engine can still reduce these sizes further.

---

### 9.4 Observation Space

MVP observation vector:

```text
symbol technical features
strategy signal features
market regime features
portfolio risk state
recent realized strategy performance
candidate default stop/target context
days since earnings, optional
```

Example observation schema:

```yaml
observation_schema:
  technical:
    - return_5d
    - return_20d
    - return_60d
    - close_vs_sma_20
    - close_vs_sma_50
    - close_vs_sma_200
    - rsi_14
    - atr_pct
    - relative_volume_20
  strategy:
    - momentum_score
    - mean_reversion_score
    - breakout_score
    - trend_score
  market:
    - spy_return_20d
    - spy_above_sma_50
    - qqq_above_sma_50
    - sector_relative_strength
    - market_volatility_20d
  portfolio:
    - current_exposure
    - current_drawdown
    - available_risk_budget
    - open_positions_count
  candidate:
    - default_holding_period_days
    - default_stop_atr
    - default_target_atr
```

---

### 9.5 Reward Function

The reward should be based on net outcome after realistic costs.

Recommended MVP reward:

```text
reward =
    realized_trade_return
  - transaction_cost_penalty
  - slippage_penalty
  - drawdown_penalty
  - turnover_penalty
  - oversized_position_penalty
```

For a trade-filter environment:

```text
If skipped:
  reward = opportunity_score_adjusted
           where missed winners are mildly penalized
           and avoided losers are rewarded

If taken:
  reward = net_trade_pnl
           adjusted for drawdown and risk used
```

Important:

- Avoid a reward function that pushes the agent into constant trading.
- Penalize poor risk-adjusted returns, not just losses.
- Reward skipping bad trades.
- Avoid rewarding “luck” from one large trade too heavily.

Possible formula:

```text
trade_reward =
    clipped(net_return / target_risk)
  - 0.10 * max_drawdown_during_trade
  - 0.05 * holding_period_excess
  - 0.02 * turnover_cost
```

---

### 9.6 Episode Design

MVP episode options:

```text
One episode = one trading year
One episode = one rolling 3-month period
One episode = randomized candidate-trade sequence from training window
```

Recommended:

```text
Use randomized rolling windows during training.
Use chronological walk-forward order during validation/test.
```

---

### 9.7 Cost, Slippage, and Fill Model

For swing trading, use a conservative daily-bar fill model.

Entry assumptions:

```text
next_open fill
or next_open plus slippage
```

Exit assumptions:

```text
next_open after exit signal
stop/target simulated using high/low path approximation
```

Cost assumptions:

```text
spread/slippage bps per trade
SEC/TAF/regulatory fees where applicable
borrow cost if shorting later
```

MVP slippage model:

```text
slippage_bps = base_bps + volatility_bps + liquidity_penalty_bps
```

Example:

```text
base_bps = 5
volatility_bps = min(20, atr_pct * 100)
liquidity_penalty_bps = 0 for highly liquid symbols
```

### 9.8 Repo-Inspired Realistic Execution Model

The reference bot includes a separate realistic-execution concept that explicitly models spread, slippage, market impact, volatility expansion, event windows, adverse selection, and partial fills. For this equity swing bot, create a dedicated module such as:

```text
rl_swing/env/equity_execution_model.py
```

Required cost components:

```text
spread_cost_bps:
  Based on actual bid/ask if available; otherwise conservative symbol-level proxy.

slippage_bps:
  Base bps plus volatility multiplier plus liquidity penalty.

volatility_multiplier:
  Higher costs when ATR percentile or realized volatility is elevated.

liquidity_penalty_bps:
  Higher costs for lower average dollar volume or wider spread.

market_impact_bps:
  Usually small for MVP position sizes, but still modeled as order_notional / avg_dollar_volume.

event_window_multiplier:
  Higher cost or forced skip around earnings, FOMC, CPI, NFP, or abnormal volatility.

adverse_selection_penalty_bps:
  Small penalty for assuming the fill occurs after the signal is known.

partial_fill_probability:
  Used later for paper/live reconciliation tests, especially limit orders.
```

MVP default should be intentionally conservative:

```yaml
equity_execution_model:
  base_spread_bps: 3
  base_slippage_bps: 5
  high_volatility_slippage_multiplier: 2.0
  event_window_slippage_multiplier: 2.0
  market_impact_coef: 0.10
  adverse_selection_bps: 2
  doubled_cost_stress_test: true
```

Training, backtesting, and validation must all use the same execution-cost module. If the agent is trained without realistic costs, it may learn excessive trading that disappears in paper trading.

---

### 9.9 Leakage Prevention

Hard rules:

- No future data in features.
- Signals for date `T` can only use information known by close of `T`.
- If executing next open, use next open as fill price.
- Scaling/normalization must be fit only on training data.
- Candidate generation must be recreated separately for train/validation/test.
- No model selection based on test period.

---

## 10. RL Algorithms

### 10.1 MVP Algorithms

Use:

```text
PPO
DQN
```

Recommended starting point:

```text
PPO for position-size action selection
DQN for simpler discrete take/skip decisions
```

### 10.2 Later Algorithms

Possible later additions:

```text
Recurrent PPO
SAC for continuous target weights
TD3 for continuous allocation
Offline RL methods
Contextual bandits
```

### 10.3 Why Start With PPO/DQN

PPO is a stable default for policy optimization. DQN is simple and suitable for discrete action spaces. The first objective is not to build the most advanced RL agent; it is to prove the environment, reward, validation, and deployment pipeline.

---

## 11. Model Registry

Every trained model must be registered with:

```text
model_id
model_type
algorithm
training_data_start
training_data_end
validation_period
test_period
universe_version
feature_config_version
strategy_config_version
reward_config_version
hyperparameters
scaler_artifact
training_code_commit
created_at
approval_status
approved_by
approved_at
paper_trade_start
paper_trade_end
live_enabled
```

Model promotion statuses:

```text
TRAINED
VALIDATED
SHADOW_APPROVED
PAPER_APPROVED
LIVE_CANDIDATE
LIVE_APPROVED
REJECTED
RETIRED
```

No model should trade live unless status is `LIVE_APPROVED`.

---

## 12. Backtesting and Validation

### 12.1 Baselines

Compare RL against:

```text
Buy and hold SPY
Buy and hold QQQ
Equal-weight universe
Momentum rule only
Mean-reversion rule only
Breakout rule only
Trend-following rule only
Random policy
Current existing bot strategy
```

### 12.2 Walk-Forward Validation

Example:

```text
Train:    2014-2019
Validate: 2020
Test:     2021

Train:    2015-2020
Validate: 2021
Test:     2022

Train:    2016-2021
Validate: 2022
Test:     2023

Train:    2017-2022
Validate: 2023
Test:     2024

Train:    2018-2023
Validate: 2024
Test:     2025
```

### 12.3 Required Metrics

Performance:

```text
CAGR
total return
annualized return
annualized volatility
Sharpe ratio
Sortino ratio
Calmar ratio
max drawdown
profit factor
win rate
average win
average loss
expectancy
exposure-adjusted return
```

Trading behavior:

```text
trade count
turnover
average holding period
median holding period
average position size
largest position
max simultaneous positions
longest flat period
```

Risk:

```text
daily VaR approximation
worst day
worst week
drawdown duration
recovery time
concentration by symbol
concentration by sector
```

Robustness:

```text
performance by year
performance by symbol
performance by strategy source
performance by market regime
performance excluding top 5 trades
performance under doubled slippage
performance under delayed entry
```

### 12.4 Crisis and Stress Validation

Adopt the reference bot's idea of explicit crisis validation, but adapt the periods to equities and the selected universe. This should be a required validation report, not an optional notebook.

Stress windows to include:

```text
2020 COVID crash and rebound
2022 inflation/rate-hike bear market
2023 regional banking stress
2024-2025 AI/semiconductor concentration periods, if relevant to the universe
Large single-symbol earnings gaps for names such as NVDA, AMD, MU, AAPL, MSFT, and AMZN
High-VIX periods
Strong bull-trend periods
Sideways/choppy periods
```

Required stress outputs:

```text
return by stress window
max drawdown by stress window
trade count by stress window
exposure by stress window
skip/take behavior by event window
position-size behavior during volatility spikes
comparison vs simple baseline strategies
result under normal costs, doubled costs, and delayed entry
```

Pass/fail principle:

```text
A good model should reduce exposure, skip marginal trades, or shrink size during hostile regimes.
A bad model keeps trading normally, increases turnover, or concentrates risk into volatility spikes.
```

---

## 12.5 Learning Loop and Validation Cycles

The learning system must be organized as two separate loops:

```text
Inner loop: RL training inside the simulated environment
Outer loop: validation, walk-forward testing, model promotion, and rejection
```

The model should never update itself during paper or live trading. A trained model is frozen, evaluated, registered, and manually promoted before it can be used by the signal engine.

### 12.5.1 End-to-End Learning Loop

```text
1. Load historical data for the selected training window.
2. Run data quality checks.
3. Build features using the approved feature configuration.
4. Generate rule-based candidate trades.
5. Convert each candidate trade into an RL observation.
6. RL agent chooses skip / take 25% / take 50% / take 100%.
7. Trading simulator applies the action using the configured fill, cost, slippage, stop, target, and holding-period assumptions.
8. Environment calculates reward.
9. RL algorithm updates the policy from simulated experience.
10. Evaluation callback tests the current policy on the validation window.
11. Best validation model is checkpointed.
12. Early stopping halts training if validation performance stops improving.
13. Final model is evaluated on untouched test data.
14. Model is either rejected or entered into the model registry for shadow-mode review.
```

The training loop should optimize for **validation performance**, not just training reward. A model with rising training reward and flat or declining validation results should be treated as overfit.

### 12.5.2 Step, Episode, and Batch Definitions

For the MVP trade-filter environment:

```text
One step:
  One candidate trade decision.

One action:
  skip, take 25%, take 50%, or take 100% of the default risk-capped size.

One reward:
  The net simulated trade outcome after costs, slippage, drawdown penalty, and turnover penalty.

One episode:
  A randomized sequence of candidate trades from a training period, such as one symbol-year, one rolling quarter, or a fixed number of candidate events.
```

Recommended MVP episode design:

```text
Training episodes:
  Randomized rolling windows from the training period.

Validation episodes:
  Chronological candidate sequence from the validation period.

Test episodes:
  Chronological candidate sequence from the untouched test period.
```

This keeps training diverse while keeping validation and testing realistic.

### 12.5.3 Inner RL Training Loop

For PPO, the inner loop should look like this:

```text
initialize policy network
initialize training environment
initialize validation environment

for total_timesteps:
    collect rollout batch from training environment
    compute advantages and rewards
    update policy for N epochs

    every eval_interval timesteps:
        run deterministic evaluation on validation period
        calculate validation metrics
        save checkpoint if validation score improves
        increment early-stopping counter if validation score does not improve
        stop if patience limit is reached

load best validation checkpoint
run final evaluation on untouched test period
register or reject model
```

For DQN, the equivalent loop is:

```text
initialize Q-network
initialize target network
initialize replay buffer

for total_timesteps:
    choose action with exploration schedule
    step environment
    store transition in replay buffer
    sample mini-batches
    update Q-network
    periodically update target network

    every eval_interval timesteps:
        run deterministic evaluation on validation period
        save checkpoint if validation score improves
        stop if patience limit is reached

load best validation checkpoint
run final evaluation on untouched test period
register or reject model
```

### 12.5.4 Recommended Training Step Counts

Initial expectations for a daily swing-trading trade-filter environment:

```text
Synthetic sanity tests:              10k–50k steps
Single-symbol toy environment:       50k–200k steps
Small multi-symbol trade filter:     300k–1M steps
Full MVP trade-filter model:         500k–2M steps
Position-sizing extension:           1M–5M steps
Full portfolio allocator:            5M+ steps, not recommended for MVP
```

A practical first run should be:

```text
total_timesteps: 500,000
evaluation_interval: 25,000 or 50,000
max_timesteps_before_redesign: 2,000,000
random_seeds: 5
```

Expected interpretation:

```text
Training improvement by 100k–300k steps:
  Good sign that the environment and reward are learnable.

Validation improvement by 500k–2M steps:
  Required before considering shadow mode.

No validation improvement by 2M steps:
  Stop and redesign features, reward, action space, cost model, or candidate generation.
```

More training steps should not be treated as automatically better. In trading, additional training often increases overfitting if the environment is flawed or the reward is too easy to exploit.

### 12.5.5 Evaluation Callback and Early Stopping

The training script should include an evaluation callback.

Recommended settings:

```yaml
validation:
  eval_interval_timesteps: 50000
  deterministic_policy: true
  min_validation_episodes: 1
  primary_score: validation_composite_score
  patience_evaluations: 10
  min_delta: 0.01
  save_best_only: true
  stop_on_no_improvement: true
```

The early-stopping rule:

```text
If validation composite score does not improve by at least min_delta
for patience_evaluations consecutive evaluations,
stop training and keep the best checkpoint.
```

Example:

```text
Max timesteps: 2,000,000
Eval interval: 50,000
Patience: 10 evaluations

If no improvement for 500,000 timesteps, stop early.
```

### 12.5.6 Validation Composite Score

The model should not be selected by return alone.

Recommended validation score:

```text
validation_composite_score =
    0.35 * normalized_total_return
  + 0.25 * normalized_sharpe_or_sortino
  + 0.20 * normalized_profit_factor
  - 0.15 * normalized_max_drawdown
  - 0.05 * normalized_turnover
```

The exact weights can be adjusted, but the score must include:

```text
return
risk-adjusted return
drawdown
profit factor
turnover/cost discipline
```

The final report should also show every component separately so a high score cannot hide an unacceptable drawdown or excessive turnover.

### 12.5.7 Walk-Forward Cycle

Each model family should be evaluated across multiple rolling windows.

Example walk-forward cycle:

```text
Cycle 1:
  Train:      2014-2019
  Validate:   2020
  Test:       2021

Cycle 2:
  Train:      2015-2020
  Validate:   2021
  Test:       2022

Cycle 3:
  Train:      2016-2021
  Validate:   2022
  Test:       2023

Cycle 4:
  Train:      2017-2022
  Validate:   2023
  Test:       2024

Cycle 5:
  Train:      2018-2023
  Validate:   2024
  Test:       2025
```

A model should not be promoted based on one good cycle. Promotion should require consistent behavior across multiple cycles.

Minimum walk-forward promotion criteria:

```text
Positive net performance in most test windows.
No single year or symbol explains most of the return.
Outperforms random policy in all or nearly all cycles.
Outperforms at least one relevant simple baseline after costs.
Does not suffer unacceptable drawdown.
Turnover remains within expected range.
Performance remains acceptable under higher slippage assumptions.
```

### 12.5.8 Random Seeds and Repeatability

Each experiment should run multiple random seeds.

Recommended MVP:

```text
seeds: 5
minimum acceptable passing seeds: 3 out of 5
```

A model family is more credible if multiple seeds produce similar validation behavior. If only one seed works and the rest fail, assume the result may be unstable.

Every training run must record:

```text
random seed
code commit
training data version
feature config version
reward config version
hyperparameters
model artifact path
validation metrics
test metrics
```

### 12.5.9 Sanity Checks Before Real Training

Before training on real market data, the RL environment must pass synthetic tests.

#### Synthetic momentum test

```text
Create fake data where momentum reliably works.
Expected result: RL learns to take momentum candidates.
Expected improvement: 10k–50k steps.
```

#### Synthetic mean-reversion test

```text
Create fake data where buying pullbacks reliably works.
Expected result: RL learns to take mean-reversion candidates.
Expected improvement: 10k–50k steps.
```

#### Random-market test

```text
Create random-walk data with no durable edge.
Expected result: RL does not find stable out-of-sample edge.
```

If the agent finds strong profits in the random-market test, assume there is leakage, reward exploitation, or a simulator bug.

### 12.5.10 Model Promotion Gate

A trained model can move forward only through explicit gates.

```text
TRAINED
  -> VALIDATED
  -> SHADOW_APPROVED
  -> PAPER_APPROVED
  -> LIVE_CANDIDATE
  -> LIVE_APPROVED
```

Promotion from `TRAINED` to `VALIDATED` requires:

```text
walk-forward report generated
validation/test metrics stored
baseline comparison completed
slippage sensitivity completed
drawdown acceptable
turnover acceptable
model artifact reproducible
```

Promotion from `VALIDATED` to `SHADOW_APPROVED` requires:

```text
manual review of walk-forward report
no known leakage issue
no unresolved data-quality issue
model registry entry complete
risk configuration mapped to model outputs
```

Promotion from `SHADOW_APPROVED` to `PAPER_APPROVED` requires:

```text
4–8 weeks of stable shadow-mode decisions
no signal generation failures
no reconciliation blockers
shadow outcomes broadly consistent with expected behavior
```

Promotion from `PAPER_APPROVED` to `LIVE_CANDIDATE` requires:

```text
at least 3 months paper trading
orders and positions reconcile reliably
no duplicate-order incidents
no unresolved kill-switch events
paper drawdown within limits
manual review completed
```

### 12.5.11 When to Redesign Instead of Keep Training

Stop adding more training steps and redesign if:

```text
training reward improves but validation declines
validation remains flat after 2M steps
model trades constantly despite turnover penalties
model refuses nearly all trades without improving risk-adjusted return
performance depends on one symbol or one year
small cost/slippage changes destroy performance
random-market sanity test shows strong profits
walk-forward cycles are inconsistent across seeds
```

The likely redesign targets are:

```text
reward function
feature set
action space
candidate trade generation
cost/slippage model
episode sampler
risk penalties
normalization and leakage controls
```

### 12.5.12 Google Colab Training-Time Estimates

For this project, Google Colab should be used for offline research training and model evaluation, not for live trading. The first useful RL swing-trading model should be trainable in hours to a few days if the MVP stays focused on daily equity bars and a trade-filter / position-sizing action space.

Assumptions for the estimates below:

```text
Universe: 20-40 liquid equities/ETFs
Timeframe: daily bars
Feature count: 50-150 features
RL role: trade filter / position sizer
Algorithms: PPO first, DQN second
Training range: 500k-2M steps
Seeds: 3-5
Runtime: Google Colab GPU/CPU session
```

Important caveat:

```text
Daily-bar swing trading is often CPU/environment-speed limited, not GPU-limited.
The slow part may be the Gymnasium environment, feature lookup, reward calculation,
logging, and validation rather than the neural-network update itself.
```

Expected training durations:

| Training Run | Purpose | Approximate Colab Time |
|---|---:|---:|
| 50k-100k steps | Smoke test / environment sanity | 10-45 minutes |
| 500k steps, 1 seed | First serious PPO run | 1-4 hours |
| 1M steps, 1 seed | Better baseline run | 2-8 hours |
| 2M steps, 1 seed | Maximum before redesign | 4-16 hours |
| 5 seeds x 500k | First credible experiment | 6-20 total hours |
| 5 seeds x 2M | Stronger validation run | 1-3 days total, depending on Colab resources/session limits |

Recommended first training sequence:

```text
Run 1:
  100k steps
  1 seed
  Goal: confirm environment, reward, action space, and logging work.

Run 2:
  500k steps
  3 seeds
  Goal: check whether validation performance begins to improve.

Run 3:
  1M steps
  5 seeds
  Goal: test repeatability of the best configuration.

Run 4:
  2M steps
  Best configuration only
  Goal: continue only if validation is still improving.
```

Expected improvement timing:

| Improvement Type | Expected Timing |
|---|---:|
| Synthetic momentum/mean-reversion sanity tests improve | 10k-50k steps |
| Training reward begins improving on real data | 100k-300k steps |
| Validation improvement begins appearing | 500k-1M steps |
| Stronger confidence across seeds | 1M-2M steps |
| Stop/redesign threshold if no validation improvement | ~2M steps |

Practical interpretation:

```text
First useful answer:
  500k steps x 3 seeds
  Usually same day or overnight in Colab.

First credible research result:
  1M steps x 5 seeds
  Usually 1-2 days of Colab usage.

Stronger validation package:
  5 seeds x 2M steps plus walk-forward reports
  Usually 1-3 days of Colab usage, assuming available sessions.
```

Training time is not the full project timeline. A realistic end-to-end build estimate is:

| Workstream | Practical Estimate |
|---|---:|
| Data pipeline | 3-7 days |
| Gymnasium/RL environment | 1-2 weeks |
| First Colab training loop | 1-3 days |
| Validation/backtest framework | ~1 week |
| Alpaca paper execution | 3-7 days |
| Reconciliation/risk/monitoring | 1-2 weeks |
| Paper-trading observation | 1-3 months |

Project-level expectation:

```text
A first trained model can be produced within a few days once the environment exists.
A trustworthy paper-trading system is more likely a 4-8 week build/test effort,
followed by at least 1-3 months of paper trading before any tiny live allocation.
```


---

## 13. Broker API: Alpaca

### 13.1 Alpaca Roles

Use Alpaca for:

```text
paper trading
order submission
position retrieval
account/buying power checks
historical/recent bars for daily production
trade/order updates
eventual live trading
```

### 13.2 MetaTrader-to-Alpaca Adaptation

The reference bot's live execution concept is useful, but the execution venue changes completely. Replace the MT5/MetaAPI modules with Alpaca-specific components.

| Reference bot execution concept | Alpaca/equity equivalent |
|---|---|
| MT5 live price feed | Alpaca Market Data bars/quotes/trade updates |
| MetaAPI cloud trading | Alpaca paper/live REST + streaming APIs |
| XAUUSD lot sizing | Equity share/notional sizing with max account exposure |
| Forex spread points/pips | Equity bid/ask spread and bps slippage |
| Session patterns: Asia/London/NY | U.S. equity market calendar, open/close, earnings, macro events |
| Stop-loss placement | Alpaca order manager plus internal stop/exit logic; bracket orders later only after testing |
| Demo account | Alpaca paper account |
| MT5 reconnect handling | Alpaca retry/idempotency handling with `client_order_id` |

Important adaptation rule:

```text
The RL model never calls Alpaca directly.
It emits an intended action.
The risk engine converts that action to an approved target.
The order manager converts the target to idempotent Alpaca orders.
The reconciliation engine verifies the broker state.
```

### 13.3 Environment Separation

Separate configs:

```text
alpaca_paper.yaml
alpaca_live.yaml
```

Never share credentials between paper and live configs.

Example:

```yaml
alpaca:
  environment: paper
  base_url: https://paper-api.alpaca.markets
  data_feed: iex
  allow_live_trading: false
  max_notional_per_order: 500
```

### 13.4 Required Alpaca Functions

Account:

```text
get account status
get buying power
get cash
get portfolio value
get margin flags
```

Positions:

```text
get all open positions
get position by symbol
compare broker positions to internal positions
```

Orders:

```text
submit order
cancel order
replace order, later
get order by broker order id
get order by client order id
list open orders
list recent closed orders
```

Market data:

```text
get latest bars
get historical daily bars
get prior close
get recent quote, optional
```

### 13.5 Order Types

MVP:

```text
market-on-open style approximation, if supported by workflow
market order near open, paper first
limit order with conservative price guard
```

Recommended production behavior:

```text
Use limit orders with price collars.
Avoid unlimited market orders for live trading.
Use client_order_id for idempotency.
Log every order request before submission.
```

Example client order ID:

```text
rlswing_20260506_MU_buy_modelv003_001
```

---

## 14. Execution Engine

### 14.1 Daily Signal Timing

Recommended daily flow:

```text
After market close:
  ingest latest data
  run data quality checks
  build features
  generate candidate trades
  run RL policy
  run risk pre-check
  save target portfolio for next session
  generate daily plan report

Before market open:
  refresh account and positions
  refresh latest prices
  rerun risk check
  compare target vs actual
  stage orders

At/after market open:
  submit approved orders
  monitor fills
  log order states
  update positions
  alert on exceptions

After market close:
  reconcile account, positions, orders, fills
  produce daily report
```

### 14.2 Target Portfolio Object

```json
{
  "target_id": "target_2026-05-06_modelv003",
  "as_of_date": "2026-05-06",
  "effective_date": "2026-05-07",
  "model_id": "rl_ppo_swing_v003",
  "targets": [
    {
      "symbol": "MU",
      "target_weight": 0.05,
      "target_notional": 2500,
      "reason": "momentum candidate approved by RL at 50% size"
    }
  ],
  "cash_target_weight": 0.85,
  "risk_status": "APPROVED_PRETRADE"
}
```

### 14.3 Order Lifecycle

States:

```text
PROPOSED
RISK_APPROVED
SUBMITTED
ACCEPTED
PARTIALLY_FILLED
FILLED
CANCELED
REJECTED
EXPIRED
RECONCILED
```

The internal order state should never rely only on local assumptions. Broker status must be queried/streamed and reconciled.

---

## 15. Risk Engine

### 15.1 Hard Limits

Initial paper/live limits:

```yaml
risk_limits:
  max_account_exposure_pct: 0.50
  max_single_position_pct: 0.10
  max_positions: 5
  max_daily_new_positions: 2
  max_daily_loss_pct: 0.01
  max_weekly_loss_pct: 0.03
  max_total_drawdown_pause_pct: 0.05
  max_order_notional_paper: 2500
  max_order_notional_live_initial: 250
  allow_shorting: false
  allow_leveraged_etfs: false
  allow_penny_stocks: false
  allow_live_retraining: false
```

### 15.2 Risk Checks

Pre-trade:

```text
symbol allowed?
model approved?
market open/expected?
data quality passed?
earnings filter passed?
position size within limits?
portfolio exposure within limits?
cash/buying power sufficient?
daily loss limit not breached?
kill switch off?
```

Post-trade:

```text
fill price reasonable?
position matches target?
unexpected open orders?
cash movement reasonable?
new exposure within limits?
```

### 15.3 Kill Switch

Kill switch triggers:

```text
daily loss limit breached
weekly loss limit breached
drawdown pause threshold breached
broker positions mismatch internal state
duplicate orders detected
data quality failure
model artifact missing/inconsistent
unexpected live mode enabled
order rejection rate too high
unhandled exception in execution engine
```

Kill switch actions:

```text
block new orders
cancel open orders, configurable
send alert
write incident record
require manual reset
```

---

## 16. Storage Design

### 16.1 Recommended Storage Stack

Development:

```text
DuckDB + Parquet files
SQLite for simple local order/trade journal
```

Production/paper runtime:

```text
PostgreSQL
Parquet feature snapshots
Object/file storage for model artifacts
```

Given the importance of auditability, PostgreSQL is recommended once moving beyond notebook research.

---

### 16.2 Core Tables

#### assets

```sql
CREATE TABLE assets (
  symbol TEXT PRIMARY KEY,
  name TEXT,
  asset_type TEXT,
  exchange TEXT,
  sector TEXT,
  is_active BOOLEAN,
  is_tradeable BOOLEAN,
  first_seen_date DATE,
  last_seen_date DATE
);
```

#### bars_daily

```sql
CREATE TABLE bars_daily (
  symbol TEXT,
  bar_date DATE,
  open NUMERIC,
  high NUMERIC,
  low NUMERIC,
  close NUMERIC,
  adjusted_close NUMERIC,
  volume BIGINT,
  source TEXT,
  adjustment_type TEXT,
  ingested_at TIMESTAMP,
  quality_status TEXT,
  PRIMARY KEY (symbol, bar_date, source)
);
```

#### features_daily

```sql
CREATE TABLE features_daily (
  feature_id TEXT PRIMARY KEY,
  symbol TEXT,
  as_of_date DATE,
  feature_config_version TEXT,
  features_json JSONB,
  created_at TIMESTAMP
);
```

#### strategy_signals

```sql
CREATE TABLE strategy_signals (
  signal_id TEXT PRIMARY KEY,
  symbol TEXT,
  as_of_date DATE,
  strategy_name TEXT,
  strategy_version TEXT,
  signal_direction TEXT,
  signal_score NUMERIC,
  metadata_json JSONB,
  created_at TIMESTAMP
);
```

#### candidate_trades

```sql
CREATE TABLE candidate_trades (
  candidate_id TEXT PRIMARY KEY,
  symbol TEXT,
  as_of_date DATE,
  strategy_name TEXT,
  direction TEXT,
  signal_score NUMERIC,
  feature_id TEXT,
  default_size_pct NUMERIC,
  default_stop_atr NUMERIC,
  default_target_atr NUMERIC,
  status TEXT,
  created_at TIMESTAMP
);
```

#### model_runs

```sql
CREATE TABLE model_runs (
  model_id TEXT PRIMARY KEY,
  algorithm TEXT,
  model_version TEXT,
  training_start DATE,
  training_end DATE,
  validation_start DATE,
  validation_end DATE,
  test_start DATE,
  test_end DATE,
  universe_version TEXT,
  feature_config_version TEXT,
  reward_config_version TEXT,
  hyperparams_json JSONB,
  artifact_path TEXT,
  approval_status TEXT,
  metrics_json JSONB,
  created_at TIMESTAMP
);
```

#### rl_decisions

```sql
CREATE TABLE rl_decisions (
  decision_id TEXT PRIMARY KEY,
  candidate_id TEXT,
  model_id TEXT,
  as_of_date DATE,
  action TEXT,
  action_size_multiplier NUMERIC,
  confidence_score NUMERIC,
  observation_hash TEXT,
  decision_metadata_json JSONB,
  created_at TIMESTAMP
);
```

#### target_positions

```sql
CREATE TABLE target_positions (
  target_id TEXT PRIMARY KEY,
  effective_date DATE,
  model_id TEXT,
  symbol TEXT,
  target_weight NUMERIC,
  target_notional NUMERIC,
  target_quantity NUMERIC,
  reason TEXT,
  risk_status TEXT,
  created_at TIMESTAMP
);
```

#### broker_orders

```sql
CREATE TABLE broker_orders (
  internal_order_id TEXT PRIMARY KEY,
  broker_order_id TEXT,
  client_order_id TEXT UNIQUE,
  environment TEXT,
  symbol TEXT,
  side TEXT,
  order_type TEXT,
  time_in_force TEXT,
  requested_qty NUMERIC,
  requested_notional NUMERIC,
  limit_price NUMERIC,
  status TEXT,
  submitted_at TIMESTAMP,
  updated_at TIMESTAMP,
  raw_request_json JSONB,
  raw_response_json JSONB
);
```

#### broker_fills

```sql
CREATE TABLE broker_fills (
  fill_id TEXT PRIMARY KEY,
  internal_order_id TEXT,
  broker_order_id TEXT,
  symbol TEXT,
  side TEXT,
  filled_qty NUMERIC,
  filled_avg_price NUMERIC,
  filled_at TIMESTAMP,
  raw_fill_json JSONB
);
```

#### broker_positions

```sql
CREATE TABLE broker_positions (
  snapshot_id TEXT,
  snapshot_at TIMESTAMP,
  environment TEXT,
  symbol TEXT,
  qty NUMERIC,
  market_value NUMERIC,
  cost_basis NUMERIC,
  unrealized_pl NUMERIC,
  raw_position_json JSONB,
  PRIMARY KEY (snapshot_id, symbol)
);
```

#### account_snapshots

```sql
CREATE TABLE account_snapshots (
  snapshot_id TEXT PRIMARY KEY,
  snapshot_at TIMESTAMP,
  environment TEXT,
  cash NUMERIC,
  buying_power NUMERIC,
  equity NUMERIC,
  portfolio_value NUMERIC,
  raw_account_json JSONB
);
```

#### reconciliation_events

```sql
CREATE TABLE reconciliation_events (
  recon_id TEXT PRIMARY KEY,
  recon_at TIMESTAMP,
  environment TEXT,
  recon_type TEXT,
  severity TEXT,
  status TEXT,
  description TEXT,
  expected_json JSONB,
  actual_json JSONB,
  resolved_at TIMESTAMP
);
```

#### risk_events

```sql
CREATE TABLE risk_events (
  risk_event_id TEXT PRIMARY KEY,
  event_at TIMESTAMP,
  severity TEXT,
  trigger_name TEXT,
  symbol TEXT,
  decision_id TEXT,
  action_taken TEXT,
  details_json JSONB
);
```

---

## 17. Reconciliation

### 17.1 Purpose

Reconciliation ensures that the bot’s internal state matches the broker’s state. It protects against duplicate orders, stale assumptions, partial fills, rejected orders, unexpected positions, and cash mismatches.

### 17.2 Reconciliation Types

Order reconciliation:

```text
internal submitted orders vs Alpaca order status
open orders expected vs open orders actual
filled quantities expected vs filled quantities actual
```

Position reconciliation:

```text
target positions vs broker positions
internal positions vs broker positions
unexpected broker positions
missing positions
quantity mismatch
market value mismatch
```

Cash/account reconciliation:

```text
expected cash vs broker cash
buying power check
portfolio value check
margin/account status check
```

Model/signal reconciliation:

```text
daily signal generated?
approved model loaded?
feature config matches model config?
latest feature date available?
```

### 17.3 Reconciliation Schedule

```text
Before market open:
  account + positions + open orders

Immediately after order submission:
  submitted order status

During execution window:
  open orders + fills

After execution window:
  orders + fills + positions

After market close:
  full account, positions, orders, fills, target comparison
```

### 17.4 Break Severity

```text
INFO: harmless mismatch, log only
WARN: needs review, no immediate trading halt
ERROR: block new orders for affected symbol
CRITICAL: activate kill switch
```

Examples:

```text
Unexpected live position: CRITICAL
Duplicate open buy orders: CRITICAL
Small market value drift due to price movement: INFO
Partial fill still inside tolerance: WARN
Feature/model version mismatch: ERROR
```

---

## 18. Monitoring and Reporting

### 18.1 Daily Report

Generate after close:

```text
Date
Model version
Universe
Signals generated
Trades proposed
Trades approved
Trades rejected by risk engine
Orders submitted
Fills
Current positions
Daily P&L
Open P&L
Realized P&L
Exposure
Drawdown
Reconciliation status
Risk events
Data quality events
```

### 18.2 Alerts

Use Discord or email for:

```text
bot started/stopped
daily signal generated
orders submitted
orders rejected
fills received
position mismatch
risk limit breached
kill switch activated
paper/live environment mismatch
model artifact mismatch
```

### 18.3 Dashboard

Recommended dashboard sections:

```text
Account summary
Current positions
Target vs actual positions
Daily trade blotter
RL decisions
Strategy attribution
Model performance
Risk dashboard
Reconciliation exceptions
Data quality status
```

---

## 19. Scheduler and Runtime

### 19.1 Suggested Jobs

```text
data_ingest_daily
data_quality_daily
feature_build_daily
strategy_signal_daily
rl_decision_daily
risk_precheck_daily
target_portfolio_daily
alpaca_order_submit
intraday_order_monitor
post_trade_reconciliation
daily_report
weekly_model_evaluation
monthly_retraining_candidate
```

### 19.2 Runtime Environment

Research/training:

```text
Google Colab
local workstation
GPU optional
```

Paper/live runtime:

```text
Raspberry Pi, desktop, or VPS
systemd service or Docker container
cron/APScheduler/Prefect
PostgreSQL
secure secrets management
```

Do not use Colab as the live trading runtime.

---

## 20. Configuration Files

### 20.1 universe.yaml

```yaml
universe:
  version: universe_v001
  symbols:
    - SPY
    - QQQ
    - IWM
    - XLK
    - SMH
    - SOXX
    - GLD
    - NVDA
    - AMD
    - MU
    - MSFT
    - AAPL
    - AMZN
  filters:
    min_price: 10
    min_avg_dollar_volume: 50000000
    allow_leveraged_etfs: false
    allow_shorting: false
```

### 20.2 features.yaml

```yaml
features:
  version: features_v001
  lookbacks:
    returns: [1, 3, 5, 10, 20, 60, 120]
    sma: [10, 20, 50, 200]
    rsi: [2, 5, 14]
    volatility: [10, 20, 60]
  normalize:
    method: rolling_zscore
    fit_on_train_only: true
```

### 20.3 rl_env.yaml

```yaml
rl_env:
  version: rl_env_v001
  mode: trade_filter
  execution_assumption: next_open
  max_holding_days: 20
  action_space:
    0: skip
    1: take_25_pct
    2: take_50_pct
    3: take_100_pct
  reward:
    type: net_trade_return_with_risk_penalty
    transaction_cost_bps: 5
    slippage_bps: 5
    drawdown_penalty_weight: 0.10
    turnover_penalty_weight: 0.02
```

### 20.4 training.yaml

```yaml
training:
  algorithm: PPO
  total_timesteps_initial: 500000
  total_timesteps_max: 2000000
  eval_interval_timesteps: 50000
  early_stopping_patience_evaluations: 10
  min_validation_delta: 0.01
  save_best_only: true
  seeds: [11, 22, 33, 44, 55]
  train_start: 2014-01-01
  train_end: 2020-12-31
  validation_start: 2021-01-01
  validation_end: 2021-12-31
  test_start: 2022-01-01
  test_end: 2022-12-31
  primary_validation_metric: validation_composite_score
  save_path: data/models/
```

### 20.5 risk_limits.yaml

```yaml
risk:
  max_account_exposure_pct: 0.50
  max_single_position_pct: 0.10
  max_positions: 5
  max_daily_new_positions: 2
  max_daily_loss_pct: 0.01
  max_weekly_loss_pct: 0.03
  max_drawdown_pause_pct: 0.05
  allow_shorting: false
  allow_live_retraining: false
```


### 20.6 component_registry.yaml

The component registry is what makes the system adaptable. It maps abstract interfaces to concrete implementations.

```yaml
components:
  market_data_providers:
    yfinance_daily:
      class: rl_swing.adapters.data.yfinance_provider.YFinanceProvider
      params:
        auto_adjust: true
    wrds_daily:
      class: rl_swing.adapters.data.wrds_provider.WrdsProvider
      params:
        library: crsp
        table: dsf
    alpaca_historical:
      class: rl_swing.adapters.data.alpaca_data_provider.AlpacaHistoricalProvider
      params:
        feed: iex

  broker_adapters:
    simulated:
      class: rl_swing.adapters.broker.simulated_broker.SimulatedBrokerAdapter
    shadow:
      class: rl_swing.adapters.broker.noop_shadow_broker.NoOpShadowBrokerAdapter
    alpaca_paper:
      class: rl_swing.adapters.broker.alpaca_paper_broker.AlpacaPaperBrokerAdapter
    alpaca_live:
      class: rl_swing.adapters.broker.alpaca_live_broker.AlpacaLiveBrokerAdapter

  policy_scorers:
    random:
      class: rl_swing.rl.agents.baseline_scorers.RandomPolicyScorer
    always_take:
      class: rl_swing.rl.agents.baseline_scorers.AlwaysTakePolicyScorer
    ppo_filter:
      class: rl_swing.rl.agents.ppo_scorer.PpoPolicyScorer
    dqn_filter:
      class: rl_swing.rl.agents.dqn_scorer.DqnPolicyScorer
```

### 20.7 runtime_shadow.yaml

```yaml
runtime:
  mode: shadow
  run_id_prefix: shadow_daily
  universe: starter_equities
  data_provider: alpaca_historical
  feature_pipeline: equities_features_v001
  strategies:
    - momentum_20_60
    - mean_reversion_rsi
    - breakout_20d
  policy_scorer: ppo_filter_v001
  risk_profile: conservative_shadow_v001
  broker_adapter: shadow
  storage_profile: local_sqlite_plus_parquet
  emit_events: true
  place_orders: false
```

### 20.8 runtime_paper.yaml

```yaml
runtime:
  mode: paper
  run_id_prefix: paper_daily
  universe: starter_equities
  data_provider: alpaca_historical
  feature_pipeline: equities_features_v001
  strategies:
    - momentum_20_60
    - mean_reversion_rsi
    - breakout_20d
  policy_scorer: ppo_filter_v001
  risk_profile: conservative_paper_v001
  broker_adapter: alpaca_paper
  storage_profile: postgres_plus_parquet
  emit_events: true
  place_orders: true
  require_reconciliation_before_new_orders: true
```

### 20.9 live_guarded.yaml

```yaml
runtime:
  mode: live_guarded
  run_id_prefix: live_guarded_daily
  universe: starter_equities
  data_provider: alpaca_historical
  feature_pipeline: equities_features_v001
  policy_scorer: approved_model_only
  risk_profile: tiny_live_v001
  broker_adapter: alpaca_live
  storage_profile: postgres_plus_parquet
  place_orders: false             # default must remain false
  allow_live_trading: false        # second gate
  manual_approval_token_required: true
  max_live_capital_pct: 0.02
```

### 20.10 Extension Rules

New functionality should be added by plug-in/config whenever possible.

```text
New data source       -> implement MarketDataProvider + registry entry
New broker            -> implement BrokerAdapter + registry entry
New strategy          -> implement CandidateStrategy + config entry
New RL algorithm      -> implement PolicyScorer + training adapter
New cost model        -> implement CostModel + rl_env config entry
New risk rule         -> implement RiskPolicy + risk profile entry
New storage engine    -> implement repository adapter + storage profile
New runtime mode      -> compose existing components through runtime config
```

Code changes should not be required for common experimentation such as:

```text
turning feature groups on/off
switching PPO to DQN
changing universe
changing risk profile
switching shadow to paper mode
changing cost assumptions
running a feature ablation
```

---

## 21. Security and Secrets

Rules:

- Never commit API keys.
- Use environment variables or a secrets manager.
- Separate paper and live keys.
- Require explicit config flag for live trading.
- Default all configs to paper mode.
- Add a second live-trading confirmation flag.

Example:

```yaml
trading:
  environment: paper
  allow_live_trading: false
  require_manual_live_approval: true
```

Environment variables:

```text
ALPACA_PAPER_API_KEY
ALPACA_PAPER_SECRET_KEY
ALPACA_LIVE_API_KEY
ALPACA_LIVE_SECRET_KEY
WRDS_USERNAME
WRDS_PASSWORD
DISCORD_WEBHOOK_URL
DATABASE_URL
```

---

## 22. Testing Requirements

### 22.1 Unit Tests

Data:

```text
adjusted returns calculated correctly
missing data detected
feature lookbacks do not leak future data
scalers fit only on train data
```

Environment:

```text
reset returns valid observation
step advances correctly
reward is deterministic
action space valid
position accounting correct
transaction costs applied
```

Risk:

```text
oversized trade rejected
drawdown breach activates kill switch
duplicate order blocked
unapproved model blocked
live mode disabled by default
```

Execution:

```text
client_order_id generated deterministically
order request logged before broker submission
broker rejection handled
partial fill handled
```

Reconciliation:

```text
unexpected position detected
open order mismatch detected
cash mismatch logged
critical break activates kill switch
```

### 22.2 Contract Tests

Contract tests verify that adapters satisfy the ports without leaking implementation details.

```text
MarketDataProvider returns MarketBar objects with required fields
FeaturePipeline returns deterministic FeatureFrame objects for the same snapshot
CandidateStrategy returns CandidateTrade objects without broker dependencies
PolicyScorer returns PolicyDecision objects for all valid action modes
RiskPolicy returns RiskRuleResult with explicit approve/block reasons
BrokerAdapter supports submit/list/cancel/account/positions behavior through the common interface
Storage repositories can save/load domain objects without losing IDs or timestamps
Runtime pipeline can swap mock adapters through config only
```

No Alpaca-specific object should cross the broker-adapter boundary. No yfinance/WRDS-specific object should cross the data-provider boundary.

### 22.2 Integration Tests

```text
historical data -> features -> candidate signals
candidate signals -> RL decisions
RL decisions -> target portfolio
target portfolio -> paper order requests
paper fills -> position reconciliation
daily report generation
```

### 22.3 Simulation Tests

```text
market crash period
high volatility period
low volatility sideways period
strong bull trend period
sector rotation period
missing data day
order rejection scenario
partial fill scenario
```

---

## 23. Phase Plan

## Phase 0A — Architecture Contract Foundation

### Objective

Lock down the domain model, ports/adapters interfaces, component registry, runtime modes, and event/audit conventions before building the first strategy or RL model.

### Deliverables

```text
domain dataclasses or Pydantic models
ports/ Protocol interfaces
component registry YAML
runtime mode configs: research, shadow, paper, live_guarded
AuditEvent schema
correlation ID convention
ADR folder with first architecture decisions
contract tests for adapters
```

### Acceptance Criteria

```text
Can instantiate the dependency container from config
Can swap a mock MarketDataProvider without touching feature code
Can swap a mock BrokerAdapter without touching decision code
Can persist and replay AuditEvent records
Can run a full no-op pipeline using synthetic data and shadow broker
No Alpaca-specific imports exist outside adapters/broker and adapters/data
```

---

## Phase 0 — Project Foundation

### Objective

Create the refactored repo structure, configuration framework, storage schema, logging/event framework, and test harness.

### Deliverables

```text
refactored src/rl_swing scaffold
domain/ ports/ adapters/ services/ runtime/ folders
config files
component registry
database schema
event/audit logging framework
basic CLI commands
unit and integration test structure
```

### Acceptance Criteria

```text
Can run tests
Can create local DB
Can load runtime config
Can resolve component classes from registry
Can write/read basic records
Can emit audit events
No broker integration yet, other than mock/no-op adapters
```

---

## Phase 1 — Data Pipeline MVP

### Objective

Build yfinance and/or Alpaca historical daily data ingestion for the starter universe.

### Deliverables

```text
daily OHLCV loader
normalized bars table
data quality checks
feature builder v001
Parquet export
```

### Acceptance Criteria

```text
Can ingest 5+ years of daily bars
Can calculate core features
Can detect missing/invalid data
Can reproduce the same feature snapshot
```

---

## Phase 2 — Baseline Strategy Layer

### Objective

Create rule-based candidate trades.

### Deliverables

```text
momentum strategy
mean-reversion strategy
breakout strategy
trend-following strategy
candidate trade table
baseline backtest
```

### Acceptance Criteria

```text
Can generate candidate trades daily
Can backtest each strategy independently
Can compare against SPY/QQQ baseline
Can store strategy signals and candidates
```

---

## Phase 2A — Equity Feature Intelligence Upgrade

### Objective

Adapt the reference bot's multi-timeframe, macro, calendar, and execution-awareness feature ideas to equities without overloading the MVP.

### Deliverables

```text
features_v002_regime_event.yaml
sector ETF relative-strength features
macro regime features: VIX, yields, DXY, SPY/QQQ/IWM context
economic calendar feature builder
earnings proximity / blackout feature builder
feature-ablation report
feature leakage tests
```

### Acceptance Criteria

```text
Feature timestamps are leakage-safe
Added features improve validation or are kept research-only
Earnings and macro event flags can block/reduce trades
Feature version is reproducible and linked to model artifacts
```

---

## Phase 3 — RL Environment MVP

### Objective

Build the trade-filter Gymnasium environment.

### Deliverables

```text
SwingTradingEnv
observation schema
discrete action space
reward function
cost/slippage model
environment unit tests
debug notebook
```

### Acceptance Criteria

```text
Environment passes Gymnasium-style sanity checks
No lookahead leakage in step logic
Random policy can run end-to-end
Reward calculations are explainable
Candidate trades can be replayed historically
```

---

## Phase 4 — Initial RL Training

### Objective

Train PPO and DQN policies in Colab or local environment.

### Deliverables

```text
PPO training notebook/script
DQN training notebook/script
training metrics
validation callback
early-stopping logic
best-checkpoint saving
seeded training runs
saved model artifacts
model registry entries
Colab runtime estimate report for each run
training throughput metrics, such as steps/sec and evaluation time
```

### Acceptance Criteria

```text
Training completes without instability
50k-100k smoke test completes in a reasonable Colab session
500k-step single-seed run completes and produces evaluation metrics
3-seed 500k-step run is used as the first credible improvement check
Evaluation runs every 25k-50k steps
Best validation checkpoint is saved
Early stopping works when validation does not improve
Model artifact can be loaded
Model can score historical candidates
Model decisions are stored
Model beats random policy in training environment
Synthetic momentum and mean-reversion sanity tests pass
Random-market sanity test does not show false edge
No-validation-improvement-by-2M-steps rule is enforced
```

---

## Phase 5 — Walk-Forward Validation

### Objective

Determine whether RL adds value out-of-sample.

### Deliverables

```text
walk-forward evaluator
baseline comparison report
validation composite score report
seed stability report
robustness tests
slippage sensitivity test
year-by-year attribution
symbol-by-symbol attribution
model promotion recommendation
```

### Acceptance Criteria

```text
RL beats random policy
RL beats at least one simple baseline after costs
At least 3 out of 5 random seeds show acceptable validation behavior
Performance not dependent on one symbol/year
Drawdown acceptable
Turnover acceptable
Results remain reasonable under doubled slippage
Final test period remains untouched until model selection is complete
If criteria fail, model is rejected or environment revised
```

---

## Phase 5A — Crisis, Cost, and Robustness Validation

### Objective

Adopt the reference bot's crisis-validation mindset and prove the equity RL policy is not fragile under stress windows, higher costs, delayed entries, and difficult regimes.

### Deliverables

```text
crisis_validation.py
stress_window_config.yaml
doubled_slippage_report
delayed_entry_report
performance_by_regime_report
performance_by_symbol_and_strategy_report
feature_ablation_report
```

### Acceptance Criteria

```text
Model does not rely on one stress window, one symbol, or one strategy source
Model reduces risk or exposure in hostile regimes
Model remains acceptable under doubled costs
Model does not increase turnover during high-volatility/event windows
Promotion recommendation explicitly discusses stress behavior
```

---

## Phase 6 — WRDS Research Upgrade

### Objective

Upgrade historical research data to WRDS where available.

### Deliverables

```text
WRDS loader
WRDS/yfinance/Alpaca data comparison
survivorship-aware universe, if dataset access supports it
corporate action handling
delisting handling, if available
```

### Acceptance Criteria

```text
Can reproduce baseline results using WRDS data
Data differences are documented
Model survives cleaner research dataset
No major hidden data-quality issue
```

---

## Phase 7 — Shadow Mode

### Objective

Run the model daily without placing trades.

### Deliverables

```text
daily signal job
RL decision report
target portfolio preview
no-order execution mode
shadow performance tracker
```

### Acceptance Criteria

```text
Runs daily without manual intervention
No broker orders submitted
Signals stored in DB
Shadow results tracked against actual outcomes
At least 4–8 weeks of stable operation before paper trading
```

---

## Phase 8 — Alpaca Paper Trading

### Objective

Connect approved model to Alpaca paper trading through risk engine.

### Deliverables

```text
Alpaca paper client
order manager
fill monitor
position reconciliation
account reconciliation
daily report
Discord alerts
```

### Acceptance Criteria

```text
Paper orders execute correctly
No duplicate orders
Positions reconcile daily
Risk engine blocks invalid trades
Paper results match expected signal logic
At least 3 months of paper results before live consideration
```

---

## Phase 9 — Production Hardening

### Objective

Make the bot reliable enough for a tiny live experiment.

### Deliverables

```text
systemd or Docker deployment
PostgreSQL storage
backups
error recovery
kill switch
manual model approval flow
live config safety checks
```

### Acceptance Criteria

```text
Bot restarts safely
No live trading unless explicitly enabled
Open orders/positions reconcile after restart
Critical breaks halt trading
Manual approval required for model promotion
```

---

## Phase 10 — Tiny Live Experiment

### Objective

Run a small, tightly capped live allocation only after paper approval.

### Deliverables

```text
live-readiness checklist
tiny live risk config
live order audit
daily manual review
live performance report
```

### Acceptance Criteria

```text
Max order notional very small
Max exposure very small
No leverage
No shorting
Manual monitoring active
Immediate rollback available
```

---

## 24. Live Readiness Checklist

Before live trading:

```text
[ ] WRDS/yfinance/Alpaca data pipeline validated
[ ] No known lookahead leakage
[ ] Backtest beats baselines after costs
[ ] Walk-forward report reviewed
[ ] Shadow mode completed
[ ] Paper trading completed
[ ] Paper results acceptable
[ ] Risk engine tested
[ ] Reconciliation tested
[ ] Kill switch tested
[ ] Broker account/position sync tested
[ ] Live config defaults to disabled
[ ] Live credentials separated
[ ] Model manually approved
[ ] Max live notional capped
[ ] Daily monitoring/reporting enabled
```

---

## 25. MVP Implementation Order

The practical order should be:

```text
1. Define domain objects, ports/interfaces, component registry, runtime configs, and AuditEvent schema.
2. Build the dependency container and no-op/synthetic pipeline.
3. Build data ingestion using yfinance/Alpaca daily bars behind MarketDataProvider adapters.
4. Build baseline strategies as CandidateStrategy plug-ins.
5. Build storage repositories and event logging.
6. Build Gymnasium trade-filter environment from modular observation/action/reward/cost components.
7. Train PPO/DQN PolicyScorer adapters on candidate trades.
8. Evaluate against baselines using the same validation pipeline.
9. Upgrade to WRDS research data by adding a WRDS adapter, not by rewriting downstream code.
10. Add shadow-mode daily signal runner using NoOpShadowBrokerAdapter.
11. Add Alpaca paper execution through AlpacaPaperBrokerAdapter.
12. Add reconciliation and risk hardening.
13. Consider tiny live trading only after paper success and manual approval.
```

---

## 26. Recommended First Milestone

The first meaningful milestone should be:

```text
For a fixed universe of 12–20 liquid symbols,
generate daily candidate trades from rule-based strategies,
train an RL model to take/skip/size those candidates,
and prove through walk-forward testing that the RL layer improves
net risk-adjusted results after transaction costs.
```

If the RL layer cannot beat simple baselines, do not proceed to paper trading.

---

## 27. Reference Bot Adaptation Matrix

This section captures the specific learnings adapted from `zero-was-here/tradingbot` and how they should be implemented for this equity/Alpaca project.

### 27.1 What the Reference Bot Does Well

The reference bot is organized around a DRL trading architecture with:

```text
PPO as the first practical RL algorithm
Dreamer/world-model RL as an advanced research option
Gymnasium-style environment
large feature families
multi-timeframe context
macro and economic calendar awareness
realistic execution-cost modeling
risk supervisor concepts
model checkpoints
Colab training path
crisis validation
monitoring and production reports
```

These concepts are directionally useful for this project, but must be adapted to daily/weekly equity swing trading and Alpaca execution.

### 27.2 Direct Adaptation Decisions

| Area | Adapted Decision for This Spec |
|---|---|
| Asset class | Use U.S. equities and ETFs, not XAUUSD/gold. |
| Broker | Use Alpaca, not MT5 or MetaAPI. |
| Initial timeframe | Use daily bars first; add weekly and optional hourly context later. |
| RL role | Use RL as trade filter / size selector first, not fully autonomous trader. |
| Algorithm | PPO first; DQN as comparison; Dreamer only as later research. |
| Feature plan | Start with 40-60 core features, then add macro/event/multi-timeframe feature tiers. |
| Macro context | Replace gold-specific macro with equity/sector/risk regime features. |
| Calendar context | Add FOMC/CPI/NFP/GDP and earnings proximity/blackout features. |
| Execution model | Convert forex pip/spread model into equity bps/spread/liquidity/slippage model. |
| Risk management | Keep dynamic sizing, daily loss limits, drawdown protection, concentration limits. |
| Monitoring | Keep daily reporting, alerts, model/run audit trail, and production monitor ideas. |
| Validation | Add crisis/stress validation as a required promotion gate. |

### 27.3 Components to Add or Strengthen

Add these modules to the repository structure over time:

```text
features/multi_timeframe_equity.py
features/macro_regime_features.py
features/calendar_event_features.py
features/earnings_features.py
features/microstructure_proxies.py
env/equity_execution_model.py
eval/crisis_validation.py
eval/slippage_stress.py
eval/feature_ablation.py
models/risk_supervisor.py
monitoring/production_monitor.py
```

### 27.4 Equity Macro Mapping

| XAUUSD/gold feature concept | Equity replacement |
|---|---|
| DXY | Keep as macro risk/liquidity context. |
| US10Y | Keep as rate regime context; include 10Y change and percentile. |
| VIX | Keep as high-priority volatility/risk regime feature. |
| SPX | Use SPY/SPX as market trend and risk-on/risk-off anchor. |
| Oil | Optional macro/sector input; more relevant for XLE/energy exposure. |
| Bitcoin | Optional risk-appetite proxy; research-only unless it improves validation. |
| Silver/GLD | GLD optional as inflation/risk hedge context; not core for all equities. |
| Session patterns | Replace with U.S. market calendar and open/close behavior; optional for hourly extension. |

### 27.5 Revised Risk Targets

Do not inherit the reference bot's aggressive performance targets. For this project, use conservative paper/live gates:

```text
Backtest target:
  Beat simple strategy baselines after costs on a risk-adjusted basis.

Shadow target:
  Stable daily signal generation and plausible decision behavior for 4-8 weeks.

Paper target:
  3+ months of clean execution, reconciliation, and risk behavior.

Tiny live target:
  Process validation first; return targets secondary.
```

### 27.6 Implementation Principle

The reference bot is useful as inspiration for architecture and feature breadth. This project should stay more conservative:

```text
Fewer features at first.
More validation.
More auditability.
Alpaca-only execution.
Equity-specific risk controls.
No autonomous live learning.
No live deployment without manual model promotion.
```

---

## 28. References

- Alpaca Paper Trading Documentation: https://docs.alpaca.markets/docs/paper-trading
- Alpaca Trading API Documentation: https://docs.alpaca.markets/docs/trading-api
- Alpaca Historical Stock Bars API: https://docs.alpaca.markets/reference/stockbars
- Alpaca Orders Documentation: https://docs.alpaca.markets/docs/orders-at-alpaca
- Alpaca Positions Documentation: https://docs.alpaca.markets/docs/working-with-positions
- Alpaca Account Documentation: https://docs.alpaca.markets/docs/working-with-account
- Alpaca WebSocket Streaming: https://docs.alpaca.markets/docs/websocket-streaming
- WRDS Python Connection Documentation: https://wrds-www.wharton.upenn.edu/documents/1443/wrds_connection.html
- yfinance Project Documentation: https://github.com/ranaroussi/yfinance
- Gymnasium Documentation: https://gymnasium.farama.org/
- Gymnasium Custom Environment Guide: https://gymnasium.farama.org/introduction/create_custom_env/
- Stable-Baselines3 Documentation: https://stable-baselines3.readthedocs.io/
- Stable-Baselines3 PPO: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
- Stable-Baselines3 DQN: https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
- Reference DRL XAUUSD bot reviewed for transferable architecture ideas: https://github.com/zero-was-here/tradingbot
- FinRL: Deep Reinforcement Learning Framework to Automate Trading in Quantitative Finance: https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID3955949_code3166414.pdf?abstractid=3955949
- FinRL summary / arXiv metadata: https://ideas.repec.org/p/arx/papers/2111.09395.html
- Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996
- Empirical Analysis of Automated Stock Trading Using Deep Reinforcement Learning: https://www.mdpi.com/2076-3417/13/1/633
- A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem: https://econpapers.repec.org/paper/arxpapers/1706.10059.htm
- A framework of deep reinforcement learning for stock evaluation functions: https://journals.sagepub.com/doi/abs/10.3233/JIFS-179653
- Reinforcement Learning for Trading Strategies: A Reproducible Comparison with Classical Baselines: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6018997
- Realistic Market Impact Modeling for Reinforcement Learning Trading Environments: https://papers.cool/arxiv/2603.29086

---

## 29. Final Recommendation

Start with RL as a **trade filter and position-size controller**, not as a fully autonomous strategy inventor.

The first production-worthy architecture should be:

```text
Rule-based candidate strategies
  + RL trade filter / size selector
  + strict risk engine
  + Alpaca paper execution
  + reconciliation
  + daily reporting
```

This gives the RL model room to learn useful behavior while keeping the system explainable, testable, and controllable.
