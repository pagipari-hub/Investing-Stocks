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
    
    # Static fallback universe in case network request to NSE fails
    return [
        "POLYCAB.NS", "DIXON.NS", "TITAN.NS", "TRENT.NS", 
        "DEEPAKNTR.NS", "PIIND.NS", "BAJFINANCE.NS", "KEI.NS", "HONASA.NS"
    ]

def extract_latest_two_years(series_or_df):
    """Sorts financial reporting periods chronologically to extract current vs previous year values."""
    if series_or_df is None or series_or_df.empty:
        return None, None
    
    # Sort columns/index chronologically ascending
    cleaned = series_or_df.dropna()
    if len(cleaned) < 2:
        return None, None
    
    # Sort by date index/columns if available
    try:
        sorted_series = cleaned.sort_index(ascending=False)
        curr_val = float(sorted_series.iloc[0])
        prev_val = float(sorted_series.iloc[1])
        return curr_val, prev_val
    except Exception:
        return None, None

def evaluate_quant_model_v2(symbol):
    """Evaluates individual ticker against V2 100-Point Audit Engine & Multibagger Hard Gates."""
    audit = {
        "Ticker": symbol.replace(".NS", ""),
        "Fundamental_Status": "INSUFFICIENT_DATA",
        "Data_Quality": "POOR",
        "Error_Reason": "",
        "Fund_Score": 0.0,
        "Tech_Score": 0.0,
        "Total_Score": 0.0,
        "Action": "REJECT",
        "Close_Price": 0.0,
        "Revenue_Growth_Pct": np.nan,
        "EBITDA_Margin_Delta_Pct": np.nan,
        "Debt_to_EBITDA": np.nan,
        "ROCE_Pct": np.nan,
        "FCF_Positive": False
    }

    try:
        ticker = yf.Ticker(symbol)
        
        # Fetch 1 year daily OHLCV
        hist = ticker.history(period="1y")
        if hist.empty or len(hist) < 200:
            audit["Error_Reason"] = "Insufficient daily price/volume history (<200 bars)"
            return audit
        
        latest_bar = hist.iloc[-1]
        audit["Close_Price"] = round(float(latest_bar['Close']), 2)

        # Fetch Financial Statements
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        cashflow = ticker.cashflow

        if financials is None or financials.empty or balance_sheet is None or balance_sheet.empty:
            audit["Error_Reason"] = "Missing primary financial statements from data source"
            return audit

        fund_score = 0.0
        metrics_computed = 0
        total_metrics_attempted = 4

        # =========================================================
        # PILLAR 1: REVENUE GROWTH (25 Points) + HARD GATE FLOOR
        # =========================================================
        if 'Total Revenue' in financials.index:
            rev_curr, rev_prev = extract_latest_two_years(financials.loc['Total Revenue'])
            if rev_curr and rev_prev and rev_prev > 0:
                rev_growth = (rev_curr - rev_prev) / rev_prev
                audit["Revenue_Growth_Pct"] = round(rev_growth * 100, 2)
                metrics_computed += 1

                # 🛑 HARD GATE 1: Revenue YoY Growth Must Be >= 8.0%
                if rev_growth < 0.08:
                    audit["Error_Reason"] = f"Failed Hard Gate: Revenue Growth ({audit['Revenue_Growth_Pct']}%) < 8.0%"
                    audit["Action"] = "REJECT"
                    return audit

                # Scoring
                if rev_growth >= 0.15:
                    fund_score += 25.0
                elif rev_growth >= 0.08:
                    fund_score += 12.5

        # =========================================================
        # PILLAR 2: EBITDA MARGIN EXPANSION (25 Points)
        # =========================================================
        if 'EBITDA' in financials.index and 'Total Revenue' in financials.index:
            ebitda_curr, ebitda_prev = extract_latest_two_years(financials.loc['EBITDA'])
            rev_curr, rev_prev = extract_latest_two_years(financials.loc['Total Revenue'])
            
            if all(v is not None for v in [ebitda_curr, ebitda_prev, rev_curr, rev_prev]) and rev_curr > 0 and rev_prev > 0:
                margin_curr = ebitda_curr / rev_curr
                margin_prev = ebitda_prev / rev_prev
                margin_delta = margin_curr - margin_prev
                audit["EBITDA_Margin_Delta_Pct"] = round(margin_delta * 100, 2)
                metrics_computed += 1

                # Scoring
                if margin_delta >= 0.01: # >= 1.00 percentage point expansion
                    fund_score += 25.0
                elif margin_delta > 0:
                    fund_score += 12.5

        # =========================================================
        # PILLAR 3: BALANCE SHEET HEALTH (20 Points) + DEBT TRAP GATE
        # =========================================================
        ebitda_curr, ebitda_prev = extract_latest_two_years(financials.loc['EBITDA']) if 'EBITDA' in financials.index else (None, None)
        debt_curr, debt_prev = extract_latest_two_years(balance_sheet.loc['Total Debt']) if 'Total Debt' in balance_sheet.index else (None, None)

        if debt_curr is not None and ebitda_curr is not None and ebitda_curr > 0:
            debt_ebitda_curr = debt_curr / ebitda_curr
            audit["Debt_to_EBITDA"] = round(debt_ebitda_curr, 2)
            metrics_computed += 1

            # 🛑 HARD GATE 2: Prolonged Debt Trap Filter
            if debt_ebitda_curr > 2.5:
                # Check YoY De-leveraging Trajectory if previous year data exists
                if debt_prev is not None and ebitda_prev is not None and ebitda_prev > 0:
                    debt_ebitda_prev = debt_prev / ebitda_prev
                    deleveraging_delta = debt_ebitda_prev - debt_ebitda_curr
                    
                    if deleveraging_delta < 0.5: # Must de-leverage by >0.5x YoY to remain eligible
                        audit["Error_Reason"] = f"Failed Hard Gate: High Prolonged Debt (Debt/EBITDA: {debt_ebitda_curr}x)"
                        audit["Action"] = "REJECT"
                        return audit
                else:
                    audit["Error_Reason"] = f"Failed Hard Gate: High Debt (Debt/EBITDA: {debt_ebitda_curr}x) without trajectory proof"
                    audit["Action"] = "REJECT"
                    return audit

            # Scoring
            if debt_ebitda_curr < 1.5:
                fund_score += 20.0
            elif 1.5 <= debt_ebitda_curr <= 2.5:
                fund_score += 10.0

        # =========================================================
        # PILLAR 4: STRUCTURAL QUALITY ENGINE (15 Points Max)
        # =========================================================
        p4_score = 0.0
        p4_metrics_found = 0

        # Sub-component A: Profitability Quality / Capital Efficiency (ROCE / ROE Estimate - 5 Pts)
        ebit_curr, _ = extract_latest_two_years(financials.loc['EBIT']) if 'EBIT' in financials.index else (None, None)
        eq_curr, _ = extract_latest_two_years(balance_sheet.loc['Stockholders Equity']) if 'Stockholders Equity' in balance_sheet.index else (None, None)
        
        if ebit_curr is not None and debt_curr is not None and eq_curr is not None:
            capital_employed = eq_curr + debt_curr
            if capital_employed > 0:
                roce = ebit_curr / capital_employed
                audit["ROCE_Pct"] = round(roce * 100, 2)
                p4_metrics_found += 1
                if roce >= 0.15: # 15% ROCE Threshold
                    p4_score += 5.0

        # Sub-component B: Cash Flow Quality (Free Cash Flow > 0 - 5 Pts)
        if cashflow is not None and not cashflow.empty:
            ocf_curr, _ = extract_latest_two_years(cashflow.loc['Operating Cash Flow']) if 'Operating Cash Flow' in cashflow.index else (None, None)
            capex_curr, _ = extract_latest_two_years(cashflow.loc['Capital Expenditure']) if 'Capital Expenditure' in cashflow.index else (0, 0)
            
            if ocf_curr is not None:
                capex_val = abs(capex_curr) if capex_curr is not None else 0
                fcf = ocf_curr - capex_val
                p4_metrics_found += 1
                if fcf > 0:
                    audit["FCF_Positive"] = True
                    p4_score += 5.0

        # Sub-component C: Earnings Consistency (Positive Operating Income across periods - 5 Pts)
        if 'Operating Income' in financials.index or 'Net Income' in financials.index:
            inc_key = 'Operating Income' if 'Operating Income' in financials.index else 'Net Income'
            inc_curr, inc_prev = extract_latest_two_years(financials.loc[inc_key])
            if inc_curr is not None and inc_prev is not None:
                p4_metrics_found += 1
                if inc_curr > 0 and inc_prev > 0:
                    p4_score += 5.0

        if p4_metrics_found > 0:
            fund_score += p4_score
            metrics_computed += 1

        # =========================================================
        # FUNDAMENTAL STATUS AUDIT & DATA QUALITY
        # =========================================================
        audit["Fund_Score"] = fund_score

        if metrics_computed == total_metrics_attempted:
            audit["Fundamental_Status"] = "VALID"
            audit["Data_Quality"] = "GOOD"
        elif metrics_computed >= 2:
            audit["Fundamental_Status"] = "VALID"
            audit["Data_Quality"] = "PARTIAL"
        else:
            audit["Fundamental_Status"] = "INSUFFICIENT_DATA"
            audit["Data_Quality"] = "POOR"
            audit["Error_Reason"] = "Could not parse sufficient fundamental pillars"
            return audit

        # =========================================================
        # TECHNICAL TRIGGER ENGINE (15 Points Max)
        # =========================================================
        tech_score = 0.0
        hist['EMA_50'] = hist['Close'].ewm(span=50, adjust=False).mean()
        hist['EMA_200'] = hist['Close'].ewm(span=200, adjust=False).mean()
        hist['Vol_20MA'] = hist['Volume'].rolling(window=20).mean()

        latest = hist.iloc[-1]

        # Trend Alignment (7.5 Pts): Price > EMA200 AND EMA50 > EMA200
        if (latest['Close'] > latest['EMA_200']) and (latest['EMA_50'] > latest['EMA_200']):
            tech_score += 7.5

        # Breakout Expansion (7.5 Pts): Close >= 95% of 52-Wk High AND Volume > 1.2x 20MA
        high_52wk = hist['Close'].max()
        if (latest['Close'] >= 0.95 * high_52wk) and (latest['Volume'] > 1.2 * latest['Vol_20MA']):
            tech_score += 7.5

        audit["Tech_Score"] = tech_score
        audit["Total_Score"] = fund_score + tech_score

        # =========================================================
        # FINAL ACTION CLASSIFICATION
        # =========================================================
        if audit["Fundamental_Status"] == "VALID" and fund_score >= 60.0:
            if tech_score >= 7.5 and audit["Total_Score"] >= 75.0:
                audit["Action"] = "FULL BUY SIGNAL"
            else:
                audit["Action"] = "WATCHLIST (Base Building)"
        else:
            audit["Action"] = "REJECT"

        return audit

    except Exception as e:
        audit["Fundamental_Status"] = "CALCULATION_ERROR"
        audit["Error_Reason"] = f"Unhandled Execution Exception: {str(e)}"
        return audit

def send_telegram_alert(df):
    """Dispatches markdown formatted summary to Telegram."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Telegram tokens not configured. Skipping alert notification.")
        return

    buys = df[df['Action'] == "FULL BUY SIGNAL"].sort_values(by="Total_Score", ascending=False)
    watchlist = df[df['Action'] == "WATCHLIST (Base Building)"].sort_values(by="Fund_Score", ascending=False)

    msg = f"📊 *NIFTY 500 QUANT SCREENER V2 AUDIT*\n"
    msg += f"Valid Audit Candidates: {len(df[df['Fundamental_Status'] == 'VALID'])}\n\n"
    
    msg += "🚀 *FULL BUY SIGNALS (Score ≥ 75):*\n"
    if not buys.empty:
        for _, r in buys.head(15).iterrows():
            msg += f"• *{r['Ticker']}*: ₹{r['Close_Price']} | Total: {r['Total_Score']} (F: {r['Fund_Score']}, T: {r['Tech_Score']}) | Quality: {r['Data_Quality']}\n"
    else:
        msg += "None this run.\n"

    msg += "\n👀 *TOP WATCHLIST (Base Building):*\n"
    if not watchlist.empty:
        for _, r in watchlist.head(10).iterrows():
            msg += f"• *{r['Ticker']}*: ₹{r['Close_Price']} | Fund Score: {r['Fund_Score']}/85 | Quality: {r['Data_Quality']}\n"
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

    print(f"Executing Quant Screener V2 engine across {len(universe)} tickers...")

    # Multi-threaded execution across Nifty 500 universe
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(evaluate_quant_model_v2, ticker): ticker for ticker in universe}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    if results:
        df_res = pd.DataFrame(results)
        os.makedirs("output", exist_ok=True)
        
        # Save complete auditable CSV file
        df_res.to_csv("output/quant_screener_results.csv", index=False)
        print("\nScreening Complete. Comprehensive audit saved to output/quant_screener_results.csv")
        
        # Print active candidates summary
        active = df_res[df_res['Action'] != "REJECT"].sort_values(by="Total_Score", ascending=False)
        print("\n--- QUALIFIED SCREENER CANDIDATES ---")
        print(active[["Ticker", "Fundamental_Status", "Data_Quality", "Fund_Score", "Tech_Score", "Total_Score", "Action"]])
        
        send_telegram_alert(df_res)
    else:
        print("Screening Complete. No stocks met the auditing thresholds.")
