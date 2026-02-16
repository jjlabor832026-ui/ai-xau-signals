import yfinance as yf
import pandas as pd
from openai import OpenAI
import os
from datetime import datetime
import json
import time
from datetime import timezone

# Config
SYMBOL = "GC=F"                  # Gold Futures (XAUUSD proxy)
INTERVAL = "15m"
CANDLE_LIMIT = 60
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

def fetch_gold_data():
    print("Fetching gold data from Yahoo Finance...")
    ticker = yf.Ticker(SYMBOL)
    
    periods = ["60d", "30d", "7d", "5d"]
    for attempt in range(3):
        for period in periods:
            try:
                df = ticker.history(period=period, interval=INTERVAL, prepost=False, actions=False)
                if not df.empty and len(df) >= CANDLE_LIMIT:
                    df.index = df.index.tz_localize(None)  # Remove timezone info
                    
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
                    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    df = df.tail(CANDLE_LIMIT)
                    
                    latest_time = df['timestamp'].iloc[-1]
                    now = datetime.now(timezone.utc)
                    latest_time_utc = latest_time.replace(tzinfo=timezone.utc) if latest_time.tzinfo is None else latest_time.astimezone(timezone.utc)
                    
                    print(f"Success on attempt {attempt+1}, period {period}! Rows: {len(df)}, Latest: {latest_time}, Close: {df['close'].iloc[-1]:.2f}")
                    if (now - latest_time_utc).days > 1:
                        print("Note: Using historical data (market closed - weekend/holiday)")
                    return df
                time.sleep(3)
            except Exception as e:
                print(f"Attempt {attempt+1}, period {period} failed: {str(e)}")
                time.sleep(5)
    
    raise ValueError("Failed to fetch enough data after retries. Market closed or Yahoo issue.")

try:
    df = fetch_gold_data()
    
    data_str = df.to_string(index=False)
    current_close = df['close'].iloc[-1]

    prompt = f"""You are a world-class ICT/SMC institutional trader with 15+ years experience trading XAUUSD.

Current price: {current_close:.2f}

Here is the latest 15m chart data (last {len(df)} candles):

{data_str}

Perform a deep, professional multi-timeframe analysis using:
- Market Structure (BOS, CHOCH, HH/HL, LH/LL)
- Order Blocks, Breaker Blocks, Mitigation Blocks
- Fair Value Gaps (FVGs) and Imbalances
- Liquidity Sweeps, Judas Swings, Stop Hunts
- Displacement and strong impulsive moves
- Key Support & Resistance zones (major levels)
- Volume confirmation and momentum

Rules:
- Be extremely selective and honest — only give signals with high conviction (minimum 70%)
- Prioritize London/NY session behavior if active
- Short-term = next 15–60 minutes
- Long-term = next 2–8 hours

Then provide realistic price projections at these exact time horizons:
- After 15 minutes
- After 1 hour
- After 4 hours
- After 1 day

Output ONLY valid JSON — nothing else:
{{
  "short_term_action": "buy" or "sell" or "hold",
  "short_term_tp": float,
  "short_term_sl": float,
  "short_term_reason": "short powerful sentence",
  "long_term_action": "buy" or "sell" or "hold",
  "long_term_tp": float,
  "long_term_sl": float,
  "long_term_reason": "short powerful sentence",
  "price_after_15m": float,
  "price_after_1h": float,
  "price_after_4h": float,
  "price_after_1d": float,
  "confidence": 0-100
}}
"""

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2
    )

    ai = json.loads(response.choices[0].message.content)

    new_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "short_term_action": ai["short_term_action"],
        "short_term_tp": ai["short_term_tp"],
        "short_term_sl": ai["short_term_sl"],
        "short_term_reason": ai["short_term_reason"][:120],
        "long_term_action": ai["long_term_action"],
        "long_term_tp": ai["long_term_tp"],
        "long_term_sl": ai["long_term_sl"],
        "long_term_reason": ai["long_term_reason"][:120],
        "price_after_15m": ai["price_after_15m"],
        "price_after_1h": ai["price_after_1h"],
        "price_after_4h": ai["price_after_4h"],
        "price_after_1d": ai["price_after_1d"],
        "confidence": ai["confidence"]
    }

    df_signals = pd.read_csv("ai_signals.csv")
    df_signals = pd.concat([df_signals, pd.DataFrame([new_row])], ignore_index=True).tail(500)
    df_signals.to_csv("ai_signals.csv", index=False)

    print("Updated:", new_row)

except Exception as e:
    print("Error:", str(e))
    raise
