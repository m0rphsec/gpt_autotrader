#!/usr/bin/env python3
import os
import sys
import json
import csv
import argparse
import logging
from datetime import datetime
import openai
import yfinance as yf
# Set custom User-Agent for yfinance HTTP requests
import requests
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
YF_SESSION = requests.Session()
YF_SESSION.headers.update({"User-Agent": USER_AGENT})
import mibian
# Handle OpenAI exception imports
try:
    from openai.error import RateLimitError, InvalidRequestError, OpenAIError
except ImportError:
    RateLimitError = InvalidRequestError = OpenAIError = Exception
import alpaca_trade_api
from alpaca_trade_api.rest import REST, TimeFrame

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Startup logs
def init_logging():
    logger.info("Starting auto_trading.py script")
    try:
        logger.info(f"OpenAI SDK version: {openai.__version__}")
    except Exception:
        logger.info("OpenAI SDK version: unknown")
    try:
        logger.info(f"Alpaca SDK version: {alpaca_trade_api.__version__}")
    except Exception:
        logger.info("Alpaca SDK version: unknown")

# Prompt template
PROMPT = r"""
You are an institutional-grade options strategist with 50+ years of experience. Based on the metrics provided,
please complete the below tasks in a manner that has a strong chance of high performance without being overly risky.
Input JSON (preceding this prompt) contains:
{  "date": "YYYY-MM-DD",
   "available_cash": 10000.00,
   "positions": [ /* existing positions */ ],
   "market_data": { /* latest trade prices */ },
   "options_chain": { /* IV, greeks per option */ }
}
Task:
1. Manage existing positions – close any if needed
2. Propose 0–5 new defined-risk option trades (note: script will be ran only once per day).
Allowed strategies: long_option, vertical_spread, cash_secured_put, covered_call,
 iron_condor, iron_butterfly, straddle, strangle, calendar_spread, diagonal_spread.
Return a JSON array (0–5) of trade objects only, no extra text. Schema:
{ "action":"buy"|"sell", "strategy":..., "legs":[{...}], "time_in_force":"day"|"gtc",
  "rationale":"<=200 chars", "expected_pop":0.0-1.0, "max_loss":#, "max_gain":# }
"""

# Helpers
def is_option_symbol(sym: str) -> bool:
    import re
    return bool(re.match(r'^[A-Z]+\d{6}[CP]\d{8}$', sym))

# Load credentials
def load_env():
    openai.api_key = os.getenv("OPENAI_API_KEY")
    a_key = os.getenv("ALPACA_API_KEY")
    a_sec = os.getenv("ALPACA_SECRET_KEY")
    if not all([openai.api_key, a_key, a_sec]):
        logger.error("Missing API keys")
        sys.exit(1)
    return a_key, a_sec

# Alpaca client
def get_client(a_key, a_sec, paper: bool):
    url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
    logger.info(f"Using {'paper' if paper else 'live'} Alpaca at {url}")
    return REST(a_key, a_sec, url, api_version='v2')

# Fetch context: positions, market_data, options_chain
def fetch_context(client) -> dict:
    logger.info("Fetching account and market data")

    acct = client.get_account()
    cash = float(acct.cash)
    positions = []
    for p in client.list_positions():
        positions.append({
            'contract_symbol': p.symbol,
            'qty': int(float(p.qty)),
            'avg_price': float(p.avg_entry_price),
            'strategy': 'long_option' if is_option_symbol(p.symbol) else 'stock'
        })

    symbols = ['AAPL','GOOG','MSFT','NVDA','GME','TSLA','SPY','AMZN','META',
               'PLTR','HOOD','ORCL','AMD','AVGO','NFLX','COIN','INTC','CRWD']

        # Fetch underlying market data via Alpaca bars (1-min)
    market_data = {}
    for sym in symbols:
        try:
            df = client.get_bars(sym, TimeFrame.Minute, limit=1).df
            if not df.empty:
                last = df.iloc[-1]
                price = float(last['close'])
                timestamp = last.name.to_pydatetime().isoformat()
                market_data[sym] = {
                    'price': price,
                    'timestamp': timestamp,
                    'bid': None,
                    'ask': None,
                    'mid': None,
                    'spread': None
                }
        except Exception as e:
            logger.warning(f"Error fetching market data for {sym}: {e}")
    logger.info(f"Market data fetched for {len(market_data)} symbols")

    options_chain = {}
    import time, math
    for sym in market_data:
        try:
            ticker = yf.Ticker(sym, session=YF_SESSION)
            opt_data = []
                        # yfinance option_chain requests to Yahoo API (per expiration)
            # Example URL: https://query2.finance.yahoo.com/v7/finance/options/{sym}?date={epoch}
            exps = ticker.options[:3]  # limit to next 3 expirations to reduce rate load
            for exp in exps:
                # skip if expiration is today or elapsed to avoid zero-time divide
                try:
                    days = (datetime.strptime(exp, '%Y-%m-%d') - datetime.utcnow()).days
                except Exception:
                    continue
                if days <= 0:
                    logger.debug(f"Skipping expiry {exp} for {sym} (days={days})")
                    continue
                # Now fetch chain and compute greeks
                chain = ticker.option_chain(exp)
                for df, kind in [(chain.calls, 'C'), (chain.puts, 'P')]:
                    for r in df.itertuples():
                        bid_raw = getattr(r, 'bid', None)
                        ask_raw = getattr(r, 'ask', None)
                        bid_o = float(bid_raw) if bid_raw and not math.isnan(bid_raw) else 0.0
                        ask_o = float(ask_raw) if ask_raw and not math.isnan(ask_raw) else 0.0
                        mid_o = (bid_o + ask_o)/2 if bid_o and ask_o else 0.0
                        oi_raw = getattr(r, 'openInterest', None)
                        oi = int(oi_raw) if oi_raw and not math.isnan(oi_raw) else 0
                        iv_raw = getattr(r, 'impliedVolatility', None)
                        iv = float(iv_raw) if iv_raw and not math.isnan(iv_raw) else 0.0
                                                # calculate Black-Scholes greeks
                        try:
                            d = mibian.BS([market_data[sym]['price'], r.strike, 0, days], volatility=iv*100)
                        except ZeroDivisionError:
                            logger.debug(f"Skipping {r.contractSymbol} due to zero days to expiry")
                            continue
                        # append option data
                        opt_data.append({
                            'contract_symbol': r.contractSymbol,
                            'implied_volatility': iv,
                            'open_interest': oi,
                            'delta': d.callDelta if kind=='C' else d.putDelta,
                            'theta': d.callTheta if kind=='C' else d.putTheta,
                            'bid': bid_o,
                            'ask': ask_o,
                            'mid': mid_o,
                            'spread': ask_o - bid_o
                        })([market_data[sym]['price'], r.strike, 0, days], volatility=iv*100)
                        opt_data.append({
                            'contract_symbol': r.contractSymbol,
                            'implied_volatility': iv,
                            'open_interest': oi,
                            'delta': d.callDelta if kind=='C' else d.putDelta,
                            'theta': d.callTheta if kind=='C' else d.putTheta,
                            'bid': bid_o,
                            'ask': ask_o,
                            'mid': mid_o,
                            'spread': ask_o - bid_o
                        })
                time.sleep(1)
            filtered = [o for o in opt_data if o['open_interest'] >= 100 and abs(o['delta']) <= 0.8]
            filtered.sort(key=lambda x: x['open_interest'], reverse=True)
            options_chain[sym] = filtered[:20]
            logger.info(f"Options chain for {sym}: {len(options_chain[sym])} of {len(opt_data)}")
        except Exception as e:
            logger.warning(f"Error fetching options for {sym}: {e}")
            if 'Rate limited' in str(e):
                logger.error("Rate limit reached; stopping option fetch.")
                break

    return {
        'date': datetime.utcnow().strftime('%Y-%m-%d'),
        'available_cash': cash,
        'positions': positions,
        'market_data': market_data,
        'options_chain': options_chain
    }

# Query GPT with robust JSON parsing
def query_model(context, model: str):
    logger.info(f"Querying OpenAI model: {model}")
    messages = [
        {'role':'system','content':PROMPT},
        {'role':'user','content':json.dumps(context)}
    ]
    try:
        resp = openai.chat.completions.create(model=model, messages=messages, temperature=0)
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        sys.exit(1)
    content = resp.choices[0].message.content
    start = content.find('[')
    end = content.rfind(']')
    if start == -1 or end == -1 or end <= start:
        logger.error(f"No JSON array found in model output.\n{content}")
        sys.exit(1)
    json_text = content[start:end+1]
    try:
        trades = json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed parsing extracted JSON: {e}\nExtracted JSON:\n{json_text}")
        sys.exit(1)
    logger.info(f"Model returned {len(trades)} trades")
    return trades

# Submit orders to Alpaca, using direct REST for options
def submit_orders(client, trades, paper_mode):
    try:
        clock = client.get_clock()
        is_open = clock.is_open
    except Exception:
        logger.warning("Could not fetch market clock; defaulting to closed for options")
        is_open = False
    log_file = 'trade_log.csv'
    write_header = not os.path.exists(log_file)
    base = 'https://paper-api.alpaca.markets' if paper_mode else 'https://api.alpaca.markets'
    order_url = f"{base}/v2/orders"
    headers = {
        'APCA-API-KEY-ID': os.getenv("ALPACA_API_KEY"),
        'APCA-API-SECRET-KEY': os.getenv("ALPACA_SECRET_KEY")
    }
    with open(log_file, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['timestamp','action','strategy','symbol','contract','qty','side','type','limit_price','time_in_force','order_id','status','error'])
        for trade in trades:
            action = trade['action']
            strategy = trade['strategy']
            tif = trade['time_in_force']
            for leg in trade['legs']:
                sym = leg['symbol']
                qty = abs(int(leg.get('quantity',1)))
                side_raw = leg.get('action', action)
                # normalize side to 'buy' or 'sell' for Alpaca API
                side = side_raw.split('_')[0]
                ot = leg.get('order_type','market')
                lp = leg.get('limit_price')
                is_opt = is_option_symbol(leg.get('contract_symbol', ''))
                if is_opt:
                    exp = datetime.strptime(leg['expiry'], '%Y-%m-%d')
                    ds = exp.strftime('%y%m%d')
                    kind = 'C' if leg['type']=='call' else 'P'
                    strike_int = int(float(leg['strike'])*1000)
                    cs = f"{sym}{ds}{kind}{strike_int:08d}"
                    params = {'symbol':cs,'qty':qty,'side':side,'type':ot,'time_in_force':'day'}
                    if ot=='limit' and lp is not None:
                        params['limit_price'] = lp
                    if ot=='market' and not is_open:
                        logger.error(f"Cannot place market option order for {cs} when market is closed. Skipping.")
                        w.writerow([datetime.utcnow().isoformat(),action,strategy,sym,cs,qty,side,ot,lp or '', 'day', '', 'skipped', 'market closed'])
                        continue
                    logger.info(f"Placing option order via REST: {params}")
                    try:
                        resp = requests.post(order_url,json=params,headers=headers)
                        logger.info(f"Option order HTTP {resp.status_code}: {resp.text}")
                        data = resp.json() if resp.headers.get('Content-Type','').startswith('application/json') else {}
                        oid = data.get('id','')
                        status = data.get('status','')
                        err = '' if resp.status_code in (200,201) else data.get('message',resp.text)
                        if err:
                            logger.error(f"Option order error: {err}")
                        w.writerow([datetime.utcnow().isoformat(),action,strategy,sym,cs,qty,side,ot,lp or '', 'day',oid,status,err])
                    except Exception as e:
                        logger.error(f"Option order HTTP error: {e}")
                        w.writerow([datetime.utcnow().isoformat(),action,strategy,sym,cs,qty,side,ot,lp or '', 'day','', '', str(e)])
                else:
                    params = {'symbol':sym,'qty':qty,'side':side,'type':ot,'time_in_force':tif}
                    if ot=='limit' and lp is not None:
                        params['limit_price'] = lp
                    logger.info(f"Placing equity order: {params}")
                    try:
                        o = client.submit_order(**params)
                        logger.info(f"Order submitted: id={o.id}, status={o.status}")
                        w.writerow([datetime.utcnow().isoformat(),action,strategy,sym,'',qty,side,ot,lp or '',tif,o.id,o.status,''])
                    except Exception as e:
                        logger.error(f"Order error for {sym}: {e}")
                        w.writerow([datetime.utcnow().isoformat(),action,strategy,sym,'',qty,side,ot,lp or '',tif,'','',str(e)])

# Main
if __name__ == '__main__':
    init_logging()
    parser = argparse.ArgumentParser(description='Auto Trading Bot')
    parser.add_argument('--mode', choices=['paper','live'], default='paper')
    parser.add_argument('--model', default='gpt-4.1')
    args = parser.parse_args()
    a_key,a_sec = load_env()
    client = get_client(a_key,a_sec,args.mode=='paper')
    ctx = fetch_context(client)
    trades = query_model(ctx,args.model)
    submit_orders(client,trades,args.mode=='paper')
