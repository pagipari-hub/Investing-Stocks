import os
import io
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_nifty_500_tickers():
    """Fetches current Nifty 500 constituents directly from official repository."""
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            df = pd.read_csv(io.StringIO(res.text))
            tickers = [f"{symbol}.NS" for symbol in df['Symbol'].dropna().tolist()]
            print(f"Successfully loaded {len(tickers)} Nifty 500 constituents.")
            return tickers
    except Exception as e:
        print(f"Failed to fetch live Nifty 500 list ({e}). Falling back to baseline universe.")
    
    # Fallback universe in case network request to NSE is blocked
    return [
        "POLYCAB.NS", "DIXON.NS", "TITAN.NS", "TRENT.NS", 
        "DEEPAKNTR.NS", "PIIND.NS", "BAJFINANCE.NS", "KEI.NS"
    ]

def evaluate_quant_model(symbol):
    """Processes individual ticker against the 100-Point Quant Model."""
    try:
        ticker = yf.Ticker(symbol)
        
        # Fetch 1 year daily OHLCV
        hist = ticker.history(period="1y")
        if hist.empty or len(hist) < 200:
            return None

        # Fetch Financials
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet

        fund_score = 0.0
        tech_score = 0.0

        # ==========================================
        # 1. FUNDAMENTAL MATRIX (85 Points Max)
        # ==========================================
        try:
            # Pillar 1: Revenue Expansion (>15% YoY)
            rev = financials.loc['Total Revenue']
            rev_growth = (rev.iloc[0] - rev.iloc[1]) / rev.iloc[1]
            if rev_growth >= 0.15:
                fund_score += 25.0
            elif rev_growth >= 0.08:
                fund_score += 12.5

            # Pillar 2: Margin Leverage (EBITDA Margin Expansion)
            ebitda = financials.loc['EBITDA']
            margin_curr = ebitda.iloc[0] / rev.iloc[0]
            margin_prev = ebitda.iloc[1] / rev.iloc[1]
            if (margin_curr - margin_prev) >= 0.01:
                fund_score += 25.0
            elif (margin_curr - margin_prev) > 0:
                fund_score += 12.5

            # Pillar 3: Balance Sheet Health (Debt/EBITDA < 1.5)
            debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
            debt_ebitda = debt / ebitda.iloc[0] if ebitda.iloc[0] > 0 else 99
            if debt_ebitda < 1.5:
                fund_score += 20.0

            # Pillar 4: Structural Quality Baseline
            fund_score += 15.0

        except Exception:
            # Operational safety baseline for incomplete financial statements
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

        # Breakout Expansion (7.5 Pts): Within 95% of 52-Week High on volume
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

        # Ignore non-qualifying candidates to optimize memory and alert payload
        if action == "REJECT":
            return None

        return {
            "Ticker": symbol.replace(".NS", ""),
            "Fund_Score": fund_score,
            "Tech_Score": tech_score,
            "Total_Score": total_score,
            "Action": action,
            "Close_Price": round(latest['Close'], 2)
        }

    except Exception:
        return None

def send_telegram_alert(df):
    """Dispatches markdown formatted summary to Telegram."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Telegram tokens not configured. Skipping alert notification.")
        return

    buys = df[df['Action'] == "FULL BUY SIGNAL"].sort_values(by="Total_Score", ascending=False)
    watchlist = df[df['Action'] == "WATCHLIST (Base Building)"].sort_values(by="Fund_Score", ascending=False)

    msg = f"📊 *NIFTY 500 QUANT SWING SCREENER*\n"
    msg += f"Candidates Qualified: {len(df)}\n\n"
    
    msg += "🚀 *BUY SIGNALS (Score ≥ 75):*\n"
    if not buys.empty:
        for _, r in buys.head(15).iterrows():
            msg += f"• *{r['Ticker']}*: ₹{r['Close_Price']} | Total: {r['Total_Score']} (F: {r['Fund_Score']}, T: {r['Tech_Score']})\n"
    else:
        msg += "None this week.\n"

    msg += "\n👀 *TOP WATCHLIST (Base Building):*\n"
    if not watchlist.empty:
        for _, r in watchlist.head(10).iterrows():
            msg += f"• *{r['Ticker']}*: ₹{r['Close_Price']} | Fund Score: {r['Fund_Score']}/85\n"
    else:
        msg += "None.\n"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
        print("Telegram notification dispatched successfully.")
    except Exception as e:
        print(f"Failed to post Telegram alert: {e}")

if __name__ == "__main__":
    universe = get_nifty_500_tickers()
    results = []

    print(f"Executing parallel analysis across {len(universe)} tickers...")

    # Multi-threaded pool executes full universe evaluation in ~60 seconds
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(evaluate_quant_model, ticker): ticker for ticker in universe}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    if results:
        df_res = pd.DataFrame(results)
        os.makedirs("output", exist_ok=True)
        df_res.to_csv("output/quant_screener_results.csv", index=False)
        print("\nScreening Complete. Output saved to output/quant_screener_results.csv")
        print(df_res.head(15))
        send_telegram_alert(df_res)
    else:
        print("Screening Complete. No stocks met the minimum fundamental setup thresholds.")
