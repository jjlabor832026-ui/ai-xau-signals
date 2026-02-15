import ccxt
import pandas as pd
from openai import OpenAI
import os
from datetime import datetime
import json

# Config
SYMBOL = "XAUUSDT"          # MEXC uses this for Tether Gold perpetual
TIMEFRAME = "15m"
LIMIT = 60
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

try:
    # Fetch data from MEXC (public OHLCV should work)
    exchange = ccxt.mexc()
    print(f"Loading markets for {exchange.id}...")
    exchange.load_markets()  # Ensure symbol exists
    if SYMBOL not in exchange.markets:
        raise ValueError(f"Symbol {SYMBOL} not found on {exchange.id}")

    print(f"Fetching OHLCV for {SYMBOL} {TIMEFRAME}...")
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    data_str = df.to_string(index=False)
    current_close = df['close'].iloc[-1]
    print(f"Current close: {current_close:.2f} | Data rows: {len(df)}")

    prompt = f"""You are a senior ICT/SMC trader specializing in XAUUSD.
Analyze this 15m chart data for the next moves:
{data_str}
Current close ~{current_close:.2f}. Think step-by-step:
1. Swings, order blocks, breakers.
2. FVGs, liquidity sweeps, displacement.
3. Structure, volume, momentum.
4. Short-term (next 15-30 min): buy/sell/hold + quick TP/SL (realistic, in points/USD).
5. Long-term (next 1-2 hours): buy/sell/hold + TP/SL based on structure.
Output ONLY valid JSON:
{{
  "short_term_action": "buy" or "sell" or "hold",
  "short_term_tp": float,
  "short_term_sl": float,
  "short_term_reason": "short sentence",
  "long_term_action": "buy" or "sell" or "hold",
  "long_term_tp": float,
  "long_term_sl": float,
  "long_term_reason": "short sentence",
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
    print("DeepSeek response:", ai)

    # New row
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
        "confidence": ai["confidence"]
    }

    df_signals = pd.read_csv("ai_signals.csv")
    df_signals = pd.concat([df_signals, pd.DataFrame([new_row])], ignore_index=True).tail(500)
    df_signals.to_csv("ai_signals.csv", index=False)

    print("Updated:", new_row)

except Exception as e:
    print("Error occurred:", str(e))
    raise  # Re-raise to show full traceback in Actions log
