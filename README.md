# AlphaEdge

An institutional-grade investment analysis platform that combines fundamental valuation, technical analysis, ML-driven forecasting, news sentiment, risk modeling, and quantitative metrics into a single interactive dashboard.

Built with a **FastAPI** backend and **Next.js 14** frontend, designed for deployment on **Railway** (backend) and **Vercel** (frontend).

---

## Features

### Fundamental Analysis
- **DCF Valuation** — Discounted cash flow model with sensitivity grid across WACC and terminal growth rate assumptions
- **Comparable Company Analysis** — Peer-based valuation using P/E, EV/EBITDA, and EV/Revenue multiples with market cap proximity filtering
- **Financial Health Scoring** — Liquidity, solvency, and profitability metrics from income statement, balance sheet, and cash flow data
- **Investment Thesis Generator** — Synthesizes company data into a structured IB-style thesis with strengths, risks, catalysts, and key metrics

### Technical Analysis
- **50+ Technical Indicators** — RSI, MACD, Bollinger Bands, SMA/EMA crossovers, ATR, OBV, Stochastic, and more
- **Market Regime Detection** — Hidden Markov Model (HMM) to classify current market environment (low/medium/high volatility regimes)
- **Support & Resistance Levels** — Algorithmic identification of key price levels
- **Factor Model** — OLS-based factor exposure analysis against market and size ETF proxies (SPY, IWM)

### Forecasting
- **Multi-Horizon Ensemble** — Forecasts at 1D, 1W, 1M, 3M, and 12M horizons
- **Statistical Models** — ARIMA and ETS (exponential smoothing)
- **Machine Learning** — XGBoost and LightGBM with engineered features
- **Deep Learning** (local only) — Transformer, 1D-CNN, and GNN architectures via PyTorch
- **Walk-Forward Validation** — Out-of-sample backtesting for forecast calibration

### News & NLP
- **Multi-Source Aggregation** — RSS feeds, Google News, and financial news APIs
- **Sentiment Analysis** — FinBERT (GPU) with keyword-based fallback for deployed environments
- **Event Detection** — Identifies earnings surprises, analyst upgrades/downgrades, M&A activity, and regulatory events
- **News Synthesis** — Aggregates article-level sentiment into an overall market narrative

### Risk Analysis
- **Value at Risk (VaR)** — Historical, parametric, and Monte Carlo VaR at 95% and 99% confidence levels
- **Drawdown Analysis** — Maximum drawdown, recovery periods, and underwater equity curves
- **Historical Stress Tests** — Performance during GFC 2008, COVID-19, Dot-Com Crash, 2022 Rate Hikes, and 2010 Flash Crash
- **Event Calendar** — Upcoming earnings dates, ex-dividend dates, and options expirations

### Quantitative Metrics
- **Performance Analytics** — Sharpe, Sortino, Calmar, and Information ratios with period returns (1M through 2Y + YTD)
- **Return Distribution** — Histogram with normal overlay, skewness, kurtosis, tail ratios, and Jarque-Bera normality test
- **Cross-Asset Correlation** — Correlation matrix against SPY, QQQ, TLT, and GLD with rolling 63-day correlation and beta decomposition
- **Monte Carlo Simulation** — GBM-based simulation with percentile fan charts, terminal distribution, and probability targets
- **Signal Backtesting** — Win rate and profit factor analysis for RSI, MACD, Golden/Death Cross, Bollinger Squeeze, and Mean Reversion signals
- **Monthly Seasonality** — Historical return patterns by calendar month

### Interactive Visualizations
- Price chart with volume overlay
- Monte Carlo fan chart with sample paths
- Correlation heatmap
- Return distribution histogram
- Rolling volatility and Sharpe ratio time series
- Seasonal return bar charts
- Signal backtest performance tables

---

## Tech Stack

### Backend
- **FastAPI** — Async REST API with Pydantic validation
- **yfinance** — Market data, company info, earnings, and peer discovery
- **pandas / NumPy / SciPy** — Data processing and statistical computation
- **scikit-learn / statsmodels** — ARIMA, ETS, and preprocessing
- **XGBoost / LightGBM** — Gradient-boosted tree forecasters
- **hmmlearn** — Hidden Markov Models for regime detection
- **diskcache** — Disk-based caching to reduce API calls
- **PyTorch / Transformers / spaCy** (optional `[gpu]` extras) — Deep learning models and NLP

### Frontend
- **Next.js 14** — React framework with App Router
- **TypeScript** — Type-safe frontend
- **Tailwind CSS** — Utility-first styling with custom dark theme
- **Recharts** — Composable charting library
- **TanStack React Query** — Server state management

---

## Project Structure

```
alphaedge/
├── backend/
│   ├── alphaedge/
│   │   ├── api/                  # FastAPI app, routes, schemas
│   │   │   ├── routes/
│   │   │   │   ├── analysis.py   # Full analysis pipeline (sync + async)
│   │   │   │   ├── health.py     # Health check endpoint
│   │   │   │   ├── snapshot.py   # Quick snapshot endpoint
│   │   │   │   └── report.py     # PDF report generation
│   │   │   └── schemas/          # Pydantic request/response models
│   │   ├── data_ingestion/       # Data sources (yfinance, EDGAR, news)
│   │   ├── fundamentals/         # DCF, comps, financials, investment thesis
│   │   ├── technicals/           # Indicators, regime detection, support/resistance
│   │   ├── forecasting/          # ARIMA, ETS, XGBoost, Transformer, CNN, GNN
│   │   ├── news_nlp/             # Sentiment, event detection, synthesis
│   │   ├── risk/                 # VaR, drawdown, scenarios, event calendar
│   │   ├── quant/                # Performance, correlation, Monte Carlo, backtesting
│   │   └── config.py             # Application settings
│   ├── pyproject.toml
│   ├── requirements.txt          # Explicit deps for Railway
│   └── runtime.txt               # Python version for Railway
├── frontend/
│   ├── src/
│   │   ├── app/                  # Next.js App Router pages
│   │   ├── components/           # Panel components (Snapshot, Fundamentals, etc.)
│   │   │   └── charts/           # Recharts visualizations
│   │   └── lib/                  # API client, formatting utilities
│   ├── package.json
│   ├── tailwind.config.ts
│   └── next.config.js            # Proxy rewrites to backend
└── .gitignore
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Optional: deep learning models (requires ~3GB disk + GPU)
pip install -e ".[gpu]"

# Start the API server
uvicorn alphaedge.api.app:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend runs on `http://localhost:3000` and proxies API requests to `http://localhost:8000`.

---

## Deployment

### Backend — Railway

1. Create a new Railway service, connect your GitHub repo
2. Set **Root Directory** to `backend`
3. Set **Build Command** to `pip install -r requirements.txt && pip install .`
4. Set **Start Command** to `python -m uvicorn alphaedge.api.app:app --host 0.0.0.0 --port $PORT`
5. Generate a public domain in Railway's networking settings

### Frontend — Vercel

1. Import your GitHub repo in Vercel
2. Set **Framework Preset** to `Next.js` and **Root Directory** to `frontend`
3. Add environment variables:
   - `BACKEND_URL` — your Railway domain (e.g. `https://your-app.up.railway.app`)
   - `NEXT_PUBLIC_BACKEND_URL` — same Railway domain (used for direct browser requests)

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Health check with dependency status |
| `GET` | `/api/v1/snapshot/{ticker}` | Quick company snapshot |
| `POST` | `/api/v1/analysis/sync` | Full synchronous analysis (returns complete result) |
| `POST` | `/api/v1/analysis/run` | Start async analysis (returns run ID for polling) |
| `GET` | `/api/v1/analysis/status/{run_id}` | Poll async analysis status |
| `GET` | `/api/v1/analysis/result/{run_id}` | Retrieve completed analysis result |
| `GET` | `/api/v1/analysis/runs` | List all analysis runs |

---

## License

MIT
