import yfinance as yf
import pandas as pd
from openai import OpenAI
import os
from datetime import datetime
import json

# Config
SYMBOL = "GC=F"  # Gold futures (tracks XAUUSD closely)
TIMEFRAME = "15m"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

try:
    print("Fetching gold data from Yahoo Finance...")
    df = yf.download(SYMBOL, period="1d", interval=TIMEFRAME)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = df.tail(60)  # Limit to last 60 candles

    if len(df) < 20:
        raise ValueError("Not enough data")

    data_str = df.to_string(index=False)
    current_close = df['close'].iloc[-1]
    print(f"Success! Close: {current_close:.2f} | Rows: {len(df)}")

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
    print("Error:", str(e))
    raise
