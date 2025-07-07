# Auto Trading Bot for Options & Stocks

This project automates options and stock trading using OpenAI (GPT-4.1) for strategy generation and the Alpaca API for execution. It fetches account data, market quotes, and option chains; builds a rich context; prompts GPT-4.1 for trade ideas; and submits orders to Alpaca (paper or live).

---

## 📦 Dependencies

- **Python 3.7+**
- **pip**

Install required packages:

```bash
pip install openai alpaca-trade-api yfinance mibian
```

> **Note:** `mibian` requires a C compiler for the Black–Scholes calculations.

## 🔑 Authentication

Set environment variables for API keys:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export ALPACA_API_KEY="your_alpaca_api_key"
export ALPACA_SECRET_KEY="your_alpaca_secret_key"
```

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/m0rphsec/gpt_autotrader.git
   cd auto-trading-bot
   ```
   
2. Install dependencies (see above).
3. Ensure your environment variables are set.

## 🚀 Usage

```bash
python auto_trading.py --mode PAPER|LIVE --model gpt-4.1
````

- `--mode paper` runs in Alpaca paper-trading (default)
- `--mode live` runs in a live account
- `--model` specifies the OpenAI model (default `gpt-4.1`)

The script logs its progress in sections:

1. **FETCH CONTEXT**: loads positions, market data, options
2. **MODEL QUERY**: sends context + prompt to GPT-4.1
3. **MODEL OUTPUT**: prints the JSON trade array
4. **RATIONALES**: logs each trade’s rationale
5. **ORDER SUBMISSION**: submits orders via Alpaca

Trade logs are appended to `trade_log.csv`.

## 🔍 Workflow Overview

1. **Initialize & load creds**
2. **Fetch context**
   - Account cash & open positions
   - Underlying market data via yfinance (`price`, `timestamp`, `bid`, `ask`, `mid`, `spread`)
   - Options chain (top 20 liquid near-ATM contracts per symbol) with:
     - `contract_symbol`, `implied_volatility`, `open_interest`, `delta`, `theta`, `bid`, `ask`, `mid`, `spread`
3. **Build GPT prompt** with institutional-grade strategy template
4. **Call OpenAI API** (CHAT completions) to get 0–5 trade ideas in strict JSON
5. **Parse & validate** the JSON response
6. **Submit orders**:
   - Equity: via Alpaca SDK
   - Options: via REST `/v2/orders` using OCC symbols, per-leg `action` (`buy`/`sell`), `type`, `time_in_force`, and a market‑hours check
7. **Log results** and errors (CSV & console)

## 📊 Context & Metrics Passed to GPT

- **date** (`YYYY-MM-DD`)
- **available\_cash** (float)
- **positions** array:
  - `contract_symbol`, `qty`, `avg_price`, `strategy` (`stock`|`long_option`)
- **market\_data** (by symbol):
  - `price`, `timestamp`, `bid`, `ask`, `mid`, `spread`
- **options\_chain** (by symbol): up to 20 entries:
  - `contract_symbol`, `implied_volatility`, `open_interest`, `delta`, `theta`, `bid`, `ask`, `mid`, `spread`

## ⚙️ Alpaca Order Submission

- **Equities**: `client.submit_order(symbol, qty, side, type, time_in_force[, limit_price])`
- **Options**: REST POST to `/v2/orders` with JSON:
  ```json
  {
    "symbol": "AAPL250718P00210000",
    "qty": 1,
    "side": "sell",
    "type": "limit|market",
    "time_in_force": "day|gtc",
    "limit_price": 1.23      // if limit
  }
  ```
- Market orders for options are only placed during market hours; otherwise they are skipped.

---

Happy trading! 🚀

