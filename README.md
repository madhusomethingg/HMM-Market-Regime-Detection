# 📉 HMM-Based Market Regime Detection for SPY
### Identifying Hidden Market States to Build a Risk-Aware Trading Strategy

> A custom Hidden Markov Model that detects Bull, Bear, and Sideways market regimes from SPY price data — and uses those regime signals to build a trading strategy that nearly triples the Sharpe ratio of buy-and-hold.

---

## 📌 Overview

Markets don't move in a straight line. They cycle through distinct regimes — trending up, trending down, or going nowhere — and the right strategy in one regime is often the wrong one in another. The problem is that these regimes aren't directly observable. They're hidden.

This project uses a Hidden Markov Model to infer those latent states from price behavior, with real-world constraints baked into the model to prevent unrealistic regime jumps. The detected regimes then drive a risk-aware allocation strategy that is evaluated against a standard buy-and-hold baseline.

| Goal | Approach |
|---|---|
| Detect latent market regimes from price data | Custom HMM with constrained transition matrix |
| Encode real-world market behavior into the model | No direct Bull↔Bear transitions, minimum 5-day persistence |
| Use regime signals to improve portfolio performance | Regime-conditional allocation strategy |
| Evaluate robustness beyond normal conditions | Stress testing under extreme market scenarios |

---

## 📂 Dataset

**SPY Historical Price Data (S&P 500 ETF)**

- **File:** `SPY_10y_processed.csv`
- **Coverage:** 10 years of daily price data
- **Features engineered:** daily returns, rolling volatility, RSI signal

---

## 🔧 Tech Stack

| Category | Libraries / Tools |
|----------|-----------|
| Data Manipulation | `pandas`, `numpy` |
| Modeling | Custom HMM, Constrained Viterbi decoder |
| Technical Indicators | RSI, rolling volatility, return bins |
| Visualization | `matplotlib`, `seaborn` |
| Notebooks | `Jupyter` |

---

## 🔬 Methodology

### 1. Hidden States

Three latent regimes are defined:

| Regime | Market Character |
|--------|-----------------|
| **Bull** | Rising prices, low volatility, momentum building |
| **Bear** | Falling prices, high volatility, risk-off behavior |
| **Sideways** | Low directional movement, uncertain conditions |

### 2. Observed Features (Emissions)

The HMM observes three engineered features to infer the hidden state:

| Feature | Bins | Regime Signal |
|---------|------|---------------|
| **Return Bin** | Low / Medium / High | High returns → Bull; Low returns → Bear |
| **Volatility Bin** | Low / High | Low vol → Bull; High vol → Bear |
| **RSI Signal** | Oversold / Neutral / Overbought | Overbought → Bull; Oversold → Bear |

### 3. Transition Model with Real-World Constraints

A key design choice here: standard HMMs allow any state to transition to any other state, which produces unrealistic regime paths (a market doesn't go from full bull to full bear overnight). This model enforces:

- ❌ No direct **Bull → Bear** or **Bear → Bull** transitions — markets pass through Sideways first
- ⏳ Minimum **5-day regime persistence** — prevents noisy single-day flips
- 🔁 High self-transition probabilities — models trend continuation

| From \ To | Bull | Sideways | Bear |
|-----------|:----:|:--------:|:----:|
| **Bull** | 0.70 | 0.30 | 0.00 |
| **Sideways** | 0.30 | 0.40 | 0.30 |
| **Bear** | 0.00 | 0.40 | 0.60 |

### 4. Constrained Viterbi Decoding

The standard Viterbi algorithm was modified to enforce minimum regime duration and disallow incompatible state transitions — producing regime paths that are both statistically optimal and financially interpretable.

> **Key Design Insight:** Encoding financial domain knowledge directly into the transition matrix and decoder — rather than letting the model learn unconstrained — is what makes the regime paths stable and actionable.

### 5. Backtesting Strategy

Detected regimes drive a simple but principled allocation rule:

| Regime | Portfolio Action |
|--------|----------------|
| **Bull** | Fully invested in SPY |
| **Sideways** | Reduced exposure / neutral |
| **Bear** | Defensive — cash or minimal exposure |

---

## 📊 Key Results

| Strategy | Sharpe Ratio | Max Drawdown |
|----------|:------------:|:------------:|
| Buy & Hold | 0.73 | 33.72% |
| **HMM Strategy** | **1.98** | **10.56%** |

- Sharpe ratio improved by **~2.7×** over buy-and-hold
- Maximum drawdown reduced from **33.72% → 10.56%**
- Stable regime detection across full 10-year period including multiple market cycles
- Stress tests (simulated March 2020 crash, synthetic volatility spikes) showed stable transitions with no excessive regime switching

---

## ⚠️ Limitations

- Transition and emission probabilities are **manually defined** using financial intuition — not learned from data
- Assumes **stationary behavior** across time; regime dynamics may shift across market cycles
- Binned features may lose fine-grained signal present in continuous indicators
- Evaluated only on **SPY** — generalization to other assets is untested

---

## 🔮 Future Work

- **Baum-Welch (EM algorithm)** — learn transition and emission parameters directly from data rather than hand-coding them
- **Rolling / adaptive transition matrices** — allow regime dynamics to evolve over time rather than staying fixed
- **Richer feature set** — incorporate macroeconomic indicators (interest rates, inflation, VIX) alongside price-based signals
- **Multi-asset evaluation** — test on QQQ, IWM, DIA to assess generalization
- **Hidden Semi-Markov Models** — explicitly model variable regime duration rather than enforcing a hard minimum
- **Transaction cost modeling** — add slippage and commission to make backtest results more realistic

---

## 🚀 Getting Started

```bash
pip install numpy pandas matplotlib seaborn jupyter
```

1. Clone the repo
2. Open `preprocessing.ipynb` to reproduce feature engineering on `SPY_10y_processed.csv`
3. Open `hmm_template.ipynb` to run the HMM and backtest
4. The constrained Viterbi decoder lives in `viterbi.py`

---

## 👤 Author

Madhumitha Rajagopal

---

## 📄 License

This project is for educational and research purposes.
