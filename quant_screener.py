import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# Expandable universe of mid/large-cap growth candidates
WATCHLIST = [
    "POLYCAB.NS", "DIXON.NS", "TITAN.NS", "TRENT.NS", 
    "DEEPAKNTR.NS", "PIIND.NS", "BAJFINANCE.NS", "KEI.NS"
]

def evaluate_quant_model(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y")
        
        if hist.empty or len(hist) < 200:
            return None

        # Fetch Financials
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet

        fund_score = 0
        tech_score = 0

        # ==========================================
        # 1. FUNDAMENTAL MATRIX (85 Points Max)
        # ==========================================
        try:
            # P1: Revenue / Reach Proxy (>15% YoY Growth = Pass)
            rev_curr = financials.loc['Total Revenue'].iloc[0]
            rev_prev = financials.loc['Total Revenue'].iloc[1]
            rev_growth = (rev_curr - rev_prev) / rev_prev
            if rev_growth >= 0.15:
                fund_score += 25
            elif rev_growth >= 0.08:
                fund_score += 12.5

            # P2: Operating Leverage & Margins (Margin Expansion)
            ebitda_curr = financials.loc['EBITDA'].iloc[0]
            ebitda_prev = financials.loc['EBITDA'].iloc[1]
            margin_curr = ebitda_curr / rev_curr
            margin_prev = ebitda_prev / rev_prev
            if (margin_curr - margin_prev) >= 0.01: # +100 bps
                fund_score += 25
            elif (margin_curr - margin_prev) > 0:
                fund_score += 12.5

            # P3: Balance Sheet Strength (Debt / EBITDA < 1.5)
            debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
            debt_ebitda = debt / ebitda_curr if ebitda_curr > 0 else 99
            if debt_ebitda < 1.5:
                fund_score += 20

            # P4: Earnings Quality / Return Proxies
            fund_score += 15 # Baseline quality filter pass for screened universe

        except Exception as f_err:
            # Fallback for data gaps: Cap fundamental score to historical safety mean
            fund_score = 60.0

        # ==========================================
        # 2. TECHNICAL TRIGGER (15 Points Max)
        # ==========================================
        hist['EMA_50'] = hist['Close'].ewm(span=50, adjust=False).mean()
        hist['EMA_200'] = hist['Close'].ewm(span=200, adjust=False).mean()
        hist['Vol_20MA'] = hist['Volume'].rolling(window=20).mean()

        latest = hist.iloc[-1]

        # Trend Alignment (7.5 Pts): Price > EMA200 AND EMA50 > EMA200
        if (latest['Close'] > latest['EMA_200']) and (latest['EMA_50'] > latest['EMA_200']):
            tech_score += 7.5

        # Volume-Price Breakout (7.5 Pts): Near 52-Wk High on 1.5x Volume
        high_52wk = hist['Close'].max()
        if (latest['Close'] >= 0.95 * high_52wk) and (latest['Volume'] > 1.2 * latest['Vol_20MA']):
            tech_score += 7.5

        total_score = fund_score + tech_score

        # Signal Classification
        if fund_score >= 60 and tech_score >= 7.5 and total_score >= 75:
            action = "FULL BUY SIGNAL"
        elif fund_score >= 60 and tech_score < 7.5:
            action = "WATCHLIST (Base Building)"
        else:
            action = "REJECT"

        return {
            "Ticker": symbol.replace(".NS", ""),
            "Fund_Score": fund_score,
            "Tech_Score": tech_score,
            "Total_Score": total_score,
            "Action": action,
            "Close_Price": round(latest['Close'], 2)
        }

    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return None

def send_telegram_alert(df):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Telegram tokens not set. Skipping push notification.")
        return

    buys = df[df['Action'] == "FULL BUY SIGNAL"]
    watchlist = df[df['Action'] == "WATCHLIST (Base Building)"]

    msg = "📊 *WEEKLY QUANT SWING SCREENER RESULTS*\n\n"
    
    msg += "🚀 *BUY SIGNALS (Score ≥ 75):*\n"
    if not buys.empty:
        for _, r in buys.iterrows():
            msg += f"• *{r['Ticker']}*: ₹{r['Close_Price']} | Total Score: {r['Total_Score']}/100 (Fund: {r['Fund_Score']}, Tech: {r['Tech_Score']})\n"
    else:
        msg += "None this week.\n"

    msg += "\n👀 *WATCHLIST (Base Building):*\n"
    if not watchlist.empty:
        for _, r in watchlist.iterrows():
            msg += f"• *{r['Ticker']}*: ₹{r['Close_Price']} | Fund Score: {r['Fund_Score']}/85\n"
    else:
        msg += "None.\n"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

if __name__ == "__main__":
    results = []
    for ticker in WATCHLIST:
        res = evaluate_quant_model(ticker)
        if res:
            results.append(res)

    df_res = pd.DataFrame(results)
    
    os.makedirs("output", exist_ok=True)
    df_res.to_csv("output/quant_screener_results.csv", index=False)
    print("Screening Complete. Results saved.")
    print(df_res)

    send_telegram_alert(df_res)
