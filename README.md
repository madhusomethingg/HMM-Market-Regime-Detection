# HMM-Based Market Regime Detection for SPY

## 📌 Overview
This project implements a **Hidden Markov Model (HMM)** to detect latent market regimes — **Bull, Bear, and Sideways** — using historical price data from the SPY (S&P 500 ETF).

The goal is to identify hidden market states in real time and use them to design a **risk-aware trading strategy** that improves risk-adjusted returns compared to a standard buy-and-hold approach.

The model incorporates **financial intuition, structural constraints, and probabilistic modeling** to produce stable and realistic regime transitions.


## 🎯 Objectives

- Detect latent market regimes (Bull, Bear, Sideways)
- Incorporate **real-world market constraints** into regime transitions
- Use regime predictions to **improve portfolio performance**
- Evaluate the model using **financial metrics** (Sharpe ratio, Max Drawdown)
- Stress test the model under **extreme market conditions**


## 🧠 Methodology

### 1. Hidden States
The model defines three latent market regimes:
- **Bull** — rising market
- **Bear** — falling market
- **Sideways** — low directional movement


### 2. Observed Features (Emissions)

The HMM uses three key financial indicators:

| Feature | Description |
|--------|-------------|
| Return Bin | Low / Medium / High |
| Volatility Bin | Low / High |
| RSI Signal | Oversold / Neutral / Overbought |

These are mapped to regimes using financial intuition:

- **Bull:** High returns, low volatility, overbought RSI  
- **Bear:** Low returns, high volatility, oversold RSI  
- **Sideways:** Medium signals across all indicators  


### 3. Transition Model (Custom Constraints)

To reflect realistic market behavior, the model enforces:

- ❌ No direct **Bull → Bear** or **Bear → Bull** transitions  
- ⏳ Minimum **5-day regime persistence**
- 🔁 High self-transition probabilities to model trend continuation

Example transition structure:

| From / To | Bull | Sideways | Bear |
|----------|------|----------|------|
| Bull     | 0.70 | 0.30     | 0.00 |
| Sideways | 0.30 | 0.40     | 0.30 |
| Bear     | 0.00 | 0.40     | 0.60 |


### 4. Constrained Viterbi Decoding

The standard Viterbi algorithm was modified to enforce:

- Minimum regime duration constraint
- Disallowed transitions between incompatible states

This ensures **stable and interpretable regime paths**.


## 📊 Backtesting Strategy

A regime-based trading strategy was constructed:

- **Bull:** Fully invested in SPY  
- **Sideways:** Reduced exposure / neutral  
- **Bear:** Defensive positioning (cash or reduced exposure)

The strategy was evaluated against a **Buy-and-Hold baseline**.


## 📈 Results

| Strategy | Sharpe Ratio | Max Drawdown |
|---------|-------------|-------------|
| Buy & Hold | 0.73 | 33.72% |
| HMM Strategy | **1.98** | **10.56%** |

### Key Observations
- Significant improvement in **risk-adjusted returns**
- Substantial reduction in **drawdowns**
- Stable regime detection across market cycles


## 📉 Visualizations

The project includes:

- 📊 **Regime-annotated SPY price chart**
- 📈 **Equity curve comparison (Strategy vs Buy-and-Hold)**

These visualizations show how the model adapts exposure based on detected regimes.


## 🧪 Stress Testing

To evaluate robustness, the model was tested on **extreme market scenarios**:

- Simulated extended crash periods (e.g., March 2020)
- Injected synthetic volatility spikes

Results:
- Stable regime transitions
- Avoided excessive switching
- Maintained improved Sharpe and reduced drawdowns


## ⚠️ Limitations

- Transition and emission probabilities are **manually defined**
- Assumes **stationary behavior** across time
- Uses **binned features**, which may lose fine-grained signals
- Evaluated only on **SPY**, limiting generalization


## 🚀 Future Improvements

- Learn parameters using **Baum-Welch (EM algorithm)**
- Introduce **rolling/adaptive transition matrices**
- Incorporate **macroeconomic features** (interest rates, inflation)
- Evaluate on multiple assets (**QQQ, IWM, DIA**)
- Extend to **Hidden Semi-Markov Models** for variable regime duration
- Add **transaction costs and slippage modeling**


## 🛠️ Tech Stack

- Python
- NumPy / Pandas
- Matplotlib / Seaborn
- Financial indicators (RSI, volatility, returns)
- Custom HMM implementation
