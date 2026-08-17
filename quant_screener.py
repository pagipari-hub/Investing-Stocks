import os
import io
import logging
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_nifty_500_tickers():
    """Fetches current Nifty 500 constituents directly from official NSE repository."""
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        res = requests.get(url, headers=headers, timeout=12)
        if res.status_code == 200:
            df = pd.read_csv(io.StringIO(res.text))
            tickers = [f"{symbol}.NS" for symbol in df['Symbol'].dropna().tolist()]
            logging.info(f"Successfully loaded {len(tickers)} Nifty 500 constituents from NSE.")
            return tickers
    except Exception as e:
        print("WARNING: USING FALLBACK UNIVERSE")
        logging.warning(f"NSE universe download failed ({e}). Reverting to fallback list.")
    
    return [
        "POLYCAB.NS", "DIXON.NS", "TITAN.NS", "TRENT.NS", 
        "DEEPAKNTR.NS", "PIIND.NS", "BAJFINANCE.NS", "KEI.NS"
    ]

def extract_annual_series(df, key):
    """Extracts and sorts the latest two annual reporting periods for a financial line item."""
    if df is None or df.empty or key not in df.index:
        return None, None, "Missing " + key
    
    series = df.loc[key].dropna()
    if len(series) < 2:
        return None, None, f"Insufficient annual periods for {key}"
    
    # Ensure chronological sorting (latest date first)
    try:
        series.index = pd.to_datetime(series.index)
        series = series.sort_index(ascending=False)
    except Exception:
        pass
    
    val_curr = float(series.iloc[0])
    val_prev = float(series.iloc[1])
    
    return val_curr, val_prev, None

def evaluate_quant_model(symbol):
    """Auditable 100-Point Quant Model evaluation pipeline."""
    audit = {
        "Ticker": symbol.replace(".NS", ""),
        "Fundamental_Status": "INSUFFICIENT_DATA",
        "Data_Quality": "POOR",
        "Revenue_Current": np.nan,
        "Revenue_Previous": np.nan,
        "Revenue_Growth": np.nan,
        "Revenue_Score": 0.0,
        "EBITDA_Current": np.nan,
        "EBITDA_Previous": np.nan,
        "EBITDA_Margin_Current": np.nan,
        "EBITDA_Margin_Previous": np.nan,
        "EBITDA_Margin_Expansion": np.nan,
        "Margin_Score": 0.0,
        "Total_Debt": np.nan,
        "Debt_EBITDA": np.nan,
        "Debt_Score": 0.0,
        "ROCE": np.nan,
        "ROE": np.nan,
        "Free_Cash_Flow": np.nan,
        "Profitability_Score": 0.0,
        "Cashflow_Score": 0.0,
        "Capital_Efficiency_Score": 0.0,
        "Quality_Score": 0.0,
        "Fund_Score": 0.0,
        "Close_Price": np.nan,
        "EMA_50": np.nan,
        "EMA_200": np.nan,
        "52W_High": np.nan,
        "Distance_From_52W_High": np.nan,
        "Volume": np.nan,
        "Volume_20MA": np.nan,
        "Volume_Ratio": np.nan,
        "Trend_Score": 0.0,
        "Breakout_Score": 0.0,
        "Tech_Score": 0.0,
        "Total_Score": 0.0,
        "Action": "INSUFFICIENT_DATA",
        "Error_Reason": ""
    }

    try:
        ticker = yf.Ticker(symbol)
        
        # 1. TECHNICAL PRE-FETCH & COMPUTATION
        hist = ticker.history(period="1y")
        if hist.empty or len(hist) < 200:
            audit["Error_Reason"] = "Insufficient OHLCV history (<200 trading days)"
            audit["Fundamental_Status"] = "CALCULATION_ERROR"
            audit["Action"] = "REJECT"
            return audit

        hist['EMA_50'] = hist['Close'].ewm(span=50, adjust=False).mean()
        hist['EMA_200'] = hist['Close'].ewm(span=200, adjust=False).mean()
        hist['Vol_20MA'] = hist['Volume'].rolling(window=20).mean()

        latest = hist.iloc[-1]
        high_52wk = hist['Close'].max()
        
        audit["Close_Price"] = round(float(latest['Close']), 2)
        audit["EMA_50"] = round(float(latest['EMA_50']), 2)
        audit["EMA_200"] = round(float(latest['EMA_200']), 2)
        audit["52W_High"] = round(float(high_52wk), 2)
        audit["Distance_From_52W_High"] = round(float((latest['Close'] - high_52wk) / high_52wk * 100), 2)
        audit["Volume"] = float(latest['Volume'])
        audit["Volume_20MA"] = float(latest['Vol_20MA'])
        
        vol_ratio = latest['Volume'] / latest['Vol_20MA'] if latest['Vol_20MA'] > 0 else 0.0
        audit["Volume_Ratio"] = round(float(vol_ratio), 2)

        # Technical Scoring (15 Points Max)
        if (latest['Close'] > latest['EMA_200']) and (latest['EMA_50'] > latest['EMA_200']):
            audit["Trend_Score"] = 7.5
            
        if (latest['Close'] >= 0.95 * high_52wk) and (vol_ratio > 1.2):
            audit["Breakout_Score"] = 7.5
            
        audit["Tech_Score"] = audit["Trend_Score"] + audit["Breakout_Score"]

        # 2. FUNDAMENTAL DATA FETCH
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        cashflow = ticker.cashflow

        # Pillar 1: Revenue Expansion
        rev_curr, rev_prev, err_rev = extract_annual_series(financials, 'Total Revenue')
        if err_rev or rev_curr is None or rev_prev is None or rev_curr <= 0 or rev_prev <= 0:
            audit["Error_Reason"] = err_rev or "Invalid Total Revenue data"
            audit["Action"] = "INSUFFICIENT_DATA"
            return audit
            
        audit["Revenue_Current"] = rev_curr
        audit["Revenue_Previous"] = rev_prev
        rev_growth = (rev_curr - rev_prev) / rev_prev
        audit["Revenue_Growth"] = round(rev_growth * 100, 2)
        
        if rev_growth >= 0.15:
            audit["Revenue_Score"] = 25.0
        elif rev_growth >= 0.08:
            audit["Revenue_Score"] = 12.5

        # Pillar 2: EBITDA Margin Expansion
        ebitda_curr, ebitda_prev, err_ebitda = extract_annual_series(financials, 'EBITDA')
        if err_ebitda or ebitda_curr is None or ebitda_prev is None:
            audit["Error_Reason"] = err_ebitda or "Missing EBITDA"
            audit["Action"] = "INSUFFICIENT_DATA"
            return audit

        audit["EBITDA_Current"] = ebitda_curr
        audit["EBITDA_Previous"] = ebitda_prev
        
        m_curr = ebitda_curr / rev_curr
        m_prev = ebitda_prev / rev_prev
        m_expand = m_curr - m_prev
        
        audit["EBITDA_Margin_Current"] = round(m_curr * 100, 2)
        audit["EBITDA_Margin_Previous"] = round(m_prev * 100, 2)
        audit["EBITDA_Margin_Expansion"] = round(m_expand * 100, 2)

        if m_expand >= 0.01:
            audit["Margin_Score"] = 25.0
        elif m_expand > 0.0:
            audit["Margin_Score"] = 12.5

        # Pillar 3: Balance Sheet Debt / EBITDA
        if balance_sheet is None or balance_sheet.empty:
            audit["Error_Reason"] = "Missing balance sheet statement"
            audit["Action"] = "INSUFFICIENT_DATA"
            return audit

        if 'Total Debt' not in balance_sheet.index:
            audit["Error_Reason"] = "Missing Total Debt field in balance sheet"
            audit["Action"] = "INSUFFICIENT_DATA"
            return audit

        debt_series = balance_sheet.loc['Total Debt'].dropna()
        if debt_series.empty:
            audit["Error_Reason"] = "Total Debt series empty"
            audit["Action"] = "INSUFFICIENT_DATA"
            return audit
            
        tot_debt = float(debt_series.iloc[0])
        audit["Total_Debt"] = tot_debt

        if ebitda_curr <= 0:
            audit["Debt_EBITDA"] = np.nan
            audit["Debt_Score"] = 0.0
        else:
            debt_ebitda = tot_debt / ebitda_curr
            audit["Debt_EBITDA"] = round(debt_ebitda, 2)
            if debt_ebitda < 1.5:
                audit["Debt_Score"] = 20.0
            elif debt_ebitda <= 2.5:
                audit["Debt_Score"] = 10.0

        # Pillar 4: Structural Quality Matrix
        # A. Profitability Quality (5 pts)
        op_inc_curr, op_inc_prev, _ = extract_annual_series(financials, 'Operating Income')
        if op_inc_curr is not None and op_inc_prev is not None:
            if op_inc_curr > 0 and op_inc_prev > 0:
                audit["Profitability_Score"] = 5.0

        # B. Cash Flow Quality (5 pts)
        if cashflow is not None and not cashflow.empty and 'Free Cash Flow' in cashflow.index:
            fcf_series = cashflow.loc['Free Cash Flow'].dropna()
            if not fcf_series.empty:
                fcf_val = float(fcf_series.iloc[0])
                audit["Free_Cash_Flow"] = fcf_val
                if fcf_val > 0:
                    audit["Cashflow_Score"] = 5.0

        # C. Capital Efficiency (5 pts)
        info = ticker.info or {}
        roe_val = info.get('returnOnEquity', None)
        roce_val = None
        
        # Calculate ROCE if EBIT & Total Assets / Current Liabilities are available
        ebit_curr, _, _ = extract_annual_series(financials, 'EBIT')
        if ebit_curr and 'Total Assets' in balance_sheet.index and 'Current Liabilities' in balance_sheet.index:
            try:
                tot_assets = float(balance_sheet.loc['Total Assets'].dropna().iloc[0])
                curr_liab = float(balance_sheet.loc['Current Liabilities'].dropna().iloc[0])
                cap_employed = tot_assets - curr_liab
                if cap_employed > 0:
                    roce_val = ebit_curr / cap_employed
            except Exception:
                roce_val = None

        if roce_val is not None:
            audit["ROCE"] = round(roce_val * 100, 2)
        if roe_val is not None:
            audit["ROE"] = round(roe_val * 100, 2)

        cap_eff_pass = False
        if roce_val is not None and roce_val >= 0.15:
            cap_eff_pass = True
        elif roe_val is not None and roe_val >= 0.15:
            cap_eff_pass = True

        if cap_eff_pass:
            audit["Capital_Efficiency_Score"] = 5.0

        audit["Quality_Score"] = (
            audit["Profitability_Score"] + 
            audit["Cashflow_Score"] + 
            audit["Capital_Efficiency_Score"]
        )

        # Fundamental Aggregation
        audit["Fund_Score"] = (
            audit["Revenue_Score"] + 
            audit["Margin_Score"] + 
            audit["Debt_Score"] + 
            audit["Quality_Score"]
        )
        audit["Fundamental_Status"] = "VALID"
        
        # Data Quality Assessment
        quality_fields = [audit["ROCE"], audit["ROE"], audit["Free_Cash_Flow"]]
        missing_q = sum(1 for f in quality_fields if pd.isna(f))
        if missing_q == 0:
            audit["Data_Quality"] = "GOOD"
        elif missing_q < len(quality_fields):
            audit["Data_Quality"] = "PARTIAL"
        else:
            audit["Data_Quality"] = "POOR"

        # Final Scoring & Action Classification
        audit["Total_Score"] = audit["Fund_Score"] + audit["Tech_Score"]

        if audit["Fund_Score"] >= 60.0 and audit["Tech_Score"] >= 7.5 and audit["Total_Score"] >= 75.0:
            audit["Action"] = "FULL BUY SIGNAL"
        elif audit["Fund_Score"] >= 60.0 and audit["Tech_Score"] < 7.5:
            audit["Action"] = "WATCHLIST (Base Building)"
        else:
            audit["Action"] = "REJECT"

        return audit

    except Exception as e:
        audit["Fundamental_Status"] = "CALCULATION_ERROR"
        audit["Error_Reason"] = f"Unhandled Exception: {str(e)}"
        audit["Action"] = "REJECT"
        return audit

def send_telegram_alert(df):
    """Dispatches auditable summary alerts to Telegram."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        logging.info("Telegram credentials missing. Alert skipped.")
        return

    valid_count = len(df[df['Fundamental_Status'] == "VALID"])
    insufficient_count = len(df[df['Fundamental_Status'] == "INSUFFICIENT_DATA"])
    
    buys = df[df['Action'] == "FULL BUY SIGNAL"].sort_values(by="Total_Score", ascending=False)
    watchlist = df[df['Action'] == "WATCHLIST (Base Building)"].sort_values(by="Fund_Score", ascending=False)

    msg = f"📊 *NIFTY 500 QUANT SCREENER V2*\n"
    msg += f"Valid stocks: {valid_count}\n"
    msg += f"Insufficient data: {insufficient_count}\n"
    msg += f"BUY signals: {len(buys)}\n"
    msg += f"Watchlist: {len(watchlist)}\n\n"

    msg += "🚀 *BUY SIGNALS (Top 15):*\n"
    if not buys.empty:
        for _, r in buys.head(15).iterrows():
            rev_str = f"{r['Revenue_Growth']}%" if pd.notna(r['Revenue_Growth']) else "N/A"
            m_str = f"{r['EBITDA_Margin_Expansion']:+0.1f}%" if pd.notna(r['EBITDA_Margin_Expansion']) else "N/A"
            debt_str = f"{r['Debt_EBITDA']}x" if pd.notna(r['Debt_EBITDA']) else "N/A"
            roce_str = f"{r['ROCE']}%" if pd.notna(r['ROCE']) else "N/A"
            vol_str = f"{r['Volume_Ratio']}x" if pd.notna(r['Volume_Ratio']) else "N/A"

            msg += f"\n🚀 *{r['Ticker']}*\n"
            msg += f"₹{r['Close_Price']} | Total: {r['Total_Score']} (Fund: {r['Fund_Score']}, Tech: {r['Tech_Score']})\n"
            msg += f"Revenue Growth: {rev_str} | Margin Expansion: {m_str}\n"
            msg += f"Debt/EBITDA: {debt_str} | ROCE: {roce_str} | Volume Ratio: {vol_str}\n"
    else:
        msg += "None this week.\n"

    msg += "\n👀 *TOP WATCHLIST (Top 10 Base Building):*\n"
    if not watchlist.empty:
        for _, r in watchlist.head(10).iterrows():
            msg += f"• *{r['Ticker']}*: ₹{r['Close_Price']} | Fund Score: {r['Fund_Score']}/85\n"
    else:
        msg += "None.\n"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=12)
        logging.info("Telegram notification successfully dispatched.")
    except Exception as e:
        logging.error(f"Failed to transmit Telegram notification: {e}")

if __name__ == "__main__":
    universe = get_nifty_500_tickers()
    results = []

    logging.info(f"Initiating screening run across {len(universe)} tickers...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(evaluate_quant_model, ticker): ticker for ticker in universe}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    df_res = pd.DataFrame(results)
    
    # Audit trail output generation
    os.makedirs("output", exist_ok=True)
    df_res.to_csv("output/quant_screener_results.csv", index=False)

    # Summary Statistics
    total_univ = len(universe)
    eval_count = len(df_res)
    valid_df = df_res[df_res['Fundamental_Status'] == "VALID"]
    insuff_df = df_res[df_res['Fundamental_Status'] == "INSUFFICIENT_DATA"]
    calc_err_df = df_res[df_res['Fundamental_Status'] == "CALCULATION_ERROR"]
    buys_df = df_res[df_res['Action'] == "FULL BUY SIGNAL"]
    watchlist_df = df_res[df_res['Action'] == "WATCHLIST (Base Building)"]
    rejected_df = df_res[df_res['Action'] == "REJECT"]

    print("\n==================================================")
    print("           QUANT SCREENER V2 SUMMARY             ")
    print("==================================================")
    print(f"Nifty 500 universe:       {total_univ}")
    print(f"Successfully evaluated:   {eval_count}")
    print(f"Valid fundamentals:       {len(valid_df)}")
    print(f"Insufficient data:        {len(insuff_df)}")
    print(f"Calculation errors:       {len(calc_err_df)}")
    print(f"BUY signals:              {len(buys_df)}")
    print(f"Watchlist:                {len(watchlist_df)}")
    print(f"Rejected:                 {len(rejected_df)}")
    print("==================================================\n")

    if not buys_df.empty:
        print("TOP 5 BUY SIGNALS:")
        print(buys_df[['Ticker', 'Fund_Score', 'Tech_Score', 'Total_Score', 'Close_Price']].head(5).to_string(index=False))
    elif not watchlist_df.empty:
        print("TOP 5 WATCHLIST CANDIDATES:")
        print(watchlist_df[['Ticker', 'Fund_Score', 'Tech_Score', 'Total_Score', 'Close_Price']].head(5).to_string(index=False))

    send_telegram_alert(df_res)
