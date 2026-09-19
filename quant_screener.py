import os
import io
import json
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from historical_memory import update_history
from concurrent.futures import ThreadPoolExecutor, as_completed
import gspread
from gspread_formatting import (
    CellFormat, Color, TextFormat, format_cell_ranges,
    ConditionalFormatRule, BooleanRule, BooleanCondition, get_conditional_format_rules
)

def get_nifty_500_tickers():
    """Fetches current Nifty 500 constituents directly from official NSE repository."""
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
    
    return [
        "POLYCAB.NS", "DIXON.NS", "TITAN.NS", "TRENT.NS", 
        "DEEPAKNTR.NS", "PIIND.NS", "BAJFINANCE.NS", "KEI.NS", "ZOTA.NS", "STALLION.NS"
    ]

def fetch_benchmark_data(symbol="^NSEI"):
    """Fetches benchmark daily history for Relative ROC and relative strength calculations."""
    try:
        bench = yf.Ticker(symbol)
        hist = bench.history(period="1y")
        if not hist.empty:
            print(f"Successfully loaded {len(hist)} daily bars for Benchmark ({symbol}).")
            return hist['Close']
    except Exception as e:
        print(f"Failed to fetch benchmark history ({e}).")
    return None

def extract_latest_two_years(series_or_df):
    """Sorts financial reporting periods chronologically to extract current vs previous year values."""
    if series_or_df is None or series_or_df.empty:
        return None, None
    
    cleaned = series_or_df.dropna()
    if len(cleaned) < 2:
        return None, None
    
    try:
        sorted_series = cleaned.sort_index(ascending=False)
        curr_val = float(sorted_series.iloc[0])
        prev_val = float(sorted_series.iloc[1])
        return curr_val, prev_val
    except Exception:
        return None, None

def evaluate_quant_model_v3(symbol, benchmark_close_series):
    """
    Evaluates individual ticker against the Unified Quant Engine:
    - 100-Point Audit Engine & Hard Gates
    - Stage 1 Emerging Entity Pipeline (Tolerates temporary ROE/Margin drag for >20% growth)
    - Relative ROC (12-period) & 21 EMA Technical Engine
    """
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
        "FCF_Positive": False,
        "Rel_ROC_12": np.nan,
        "Above_21_EMA": False,
        "Stock_ROC_21": np.nan
    }

    try:
        ticker = yf.Ticker(symbol)
        
        # 1. Price/Volume History
        hist = ticker.history(period="1y")
        if hist.empty or len(hist) < 200:
            audit["Error_Reason"] = "Insufficient daily price/volume history (<200 bars)"
            return audit
        
        latest_bar = hist.iloc[-1]
        audit["Close_Price"] = round(float(latest_bar['Close']), 2)

        # =========================================================
        # TECHNICAL ENGINE: RELATIVE ROC & TREND TRAIL
        # =========================================================
        hist['EMA_21'] = hist['Close'].ewm(span=21, adjust=False).mean()
        hist['EMA_50'] = hist['Close'].ewm(span=50, adjust=False).mean()
        hist['EMA_200'] = hist['Close'].ewm(span=200, adjust=False).mean()
        hist['Vol_20MA'] = hist['Volume'].rolling(window=20).mean()

        latest_close = float(latest_bar['Close'])
        latest_ema21 = float(hist['EMA_21'].iloc[-1])
        audit["Above_21_EMA"] = latest_close > latest_ema21

        # Relative ROC (12-period) Calculation against Benchmark
        rel_roc_pass = False
        stock_roc_21 = np.nan
        bench_roc_21 = np.nan

        if benchmark_close_series is not None and not benchmark_close_series.empty:
            common_idx = hist.index.intersection(benchmark_close_series.index)
            if len(common_idx) > 30:
                s_close = hist.loc[common_idx, 'Close']
                b_close = benchmark_close_series.loc[common_idx]

                rel_ratio = s_close / b_close
                rel_roc_12_series = ((rel_ratio - rel_ratio.shift(12)) / rel_ratio.shift(12)) * 100
                audit["Rel_ROC_12"] = round(float(rel_roc_12_series.iloc[-1]), 2)

                s_roc21_series = ((s_close - s_close.shift(21)) / s_close.shift(21)) * 100
                b_roc21_series = ((b_close - b_close.shift(21)) / b_close.shift(21)) * 100
                
                stock_roc_21 = float(s_roc21_series.iloc[-1])
                bench_roc_21 = float(b_roc21_series.iloc[-1])
                audit["Stock_ROC_21"] = round(stock_roc_21, 2)

                if audit["Rel_ROC_12"] > 0 and audit["Above_21_EMA"]:
                    rel_roc_pass = True

        # =========================================================
        # FINANCIAL STATEMENTS PARSING
        # =========================================================
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        cashflow = ticker.cashflow

        if financials is None or financials.empty or balance_sheet is None or balance_sheet.empty:
            audit["Error_Reason"] = "Missing primary financial statements"
            return audit

        fund_score = 0.0
        metrics_computed = 0
        total_metrics_attempted = 4
        is_emerging_growth = False

        # ---------------------------------------------------------
        # PILLAR 1: REVENUE GROWTH (25 Pts) + HARD GATE FLOOR
        # ---------------------------------------------------------
        if 'Total Revenue' in financials.index:
            rev_curr, rev_prev = extract_latest_two_years(financials.loc['Total Revenue'])
            if rev_curr and rev_prev and rev_prev > 0:
                rev_growth = (rev_curr - rev_prev) / rev_prev
                audit["Revenue_Growth_Pct"] = round(rev_growth * 100, 2)
                metrics_computed += 1

                # Flag high-growth emerging entity (Stage 1 Candidate)
                if rev_growth >= 0.20:
                    is_emerging_growth = True

                # 🛑 HARD GATE 1: Revenue YoY Growth Must Be >= 8.0%
                if rev_growth < 0.08:
                    audit["Error_Reason"] = f"Failed Hard Gate: Revenue Growth ({audit['Revenue_Growth_Pct']}%) < 8.0%"
                    audit["Action"] = "REJECT"
                    return audit

                if rev_growth >= 0.20:
                    fund_score += 25.0
                elif rev_growth >= 0.15:
                    fund_score += 18.0
                elif rev_growth >= 0.08:
                    fund_score += 10.0

        # ---------------------------------------------------------
        # PILLAR 2: EBITDA MARGIN EXPANSION (25 Pts)
        # ---------------------------------------------------------
        ebitda_curr, ebitda_prev = (None, None)
        if 'EBITDA' in financials.index and 'Total Revenue' in financials.index:
            ebitda_curr, ebitda_prev = extract_latest_two_years(financials.loc['EBITDA'])
            rev_curr, rev_prev = extract_latest_two_years(financials.loc['Total Revenue'])
            
            if all(v is not None for v in [ebitda_curr, ebitda_prev, rev_curr, rev_prev]) and rev_curr > 0 and rev_prev > 0:
                margin_curr = ebitda_curr / rev_curr
                margin_prev = ebitda_prev / rev_prev
                margin_delta = margin_curr - margin_prev
                audit["EBITDA_Margin_Delta_Pct"] = round(margin_delta * 100, 2)
                metrics_computed += 1

                if margin_delta >= 0.01:
                    fund_score += 25.0
                elif margin_delta > 0:
                    fund_score += 12.5

        # ---------------------------------------------------------
        # PILLAR 3: BALANCE SHEET HEALTH (20 Pts) + DEBT TRAP GATE
        # ---------------------------------------------------------
        debt_curr, debt_prev = extract_latest_two_years(balance_sheet.loc['Total Debt']) if 'Total Debt' in balance_sheet.index else (None, None)

        if debt_curr is not None and ebitda_curr is not None and ebitda_curr > 0:
            debt_ebitda_curr = debt_curr / ebitda_curr
            audit["Debt_to_EBITDA"] = round(debt_ebitda_curr, 2)
            metrics_computed += 1

            # 🛑 HARD GATE 2: Prolonged Debt Trap Filter
            if debt_ebitda_curr > 2.5:
                if debt_prev is not None and ebitda_prev is not None and ebitda_prev > 0:
                    debt_ebitda_prev = debt_prev / ebitda_prev
                    deleveraging_delta = debt_ebitda_prev - debt_ebitda_curr
                    
                    if deleveraging_delta < 0.5:
                        audit["Error_Reason"] = f"Failed Hard Gate: High Prolonged Debt (Debt/EBITDA: {debt_ebitda_curr}x)"
                        audit["Action"] = "REJECT"
                        return audit
                else:
                    audit["Error_Reason"] = f"Failed Hard Gate: High Debt ({debt_ebitda_curr}x) without trajectory proof"
                    audit["Action"] = "REJECT"
                    return audit

            if debt_ebitda_curr < 1.5:
                fund_score += 20.0
            elif 1.5 <= debt_ebitda_curr <= 2.5:
                fund_score += 10.0

        # ---------------------------------------------------------
        # PILLAR 4: STRUCTURAL QUALITY ENGINE (15 Pts Max)
        # ---------------------------------------------------------
        p4_score = 0.0
        p4_metrics_found = 0

        # Sub-component A: ROCE
        ebit_curr, _ = extract_latest_two_years(financials.loc['EBIT']) if 'EBIT' in financials.index else (None, None)
        eq_curr, _ = extract_latest_two_years(balance_sheet.loc['Stockholders Equity']) if 'Stockholders Equity' in balance_sheet.index else (None, None)
        
        if ebit_curr is not None and debt_curr is not None and eq_curr is not None:
            capital_employed = eq_curr + debt_curr
            if capital_employed > 0:
                roce = ebit_curr / capital_employed
                audit["ROCE_Pct"] = round(roce * 100, 2)
                p4_metrics_found += 1
                if roce >= 0.15:
                    p4_score += 5.0

        # Sub-component B: Cash Flow Quality
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

        # Sub-component C: Earnings Consistency
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

        # Audit Assessment
        audit["Fund_Score"] = fund_score
        if metrics_computed >= 2:
            audit["Fundamental_Status"] = "VALID"
            audit["Data_Quality"] = "GOOD" if metrics_computed == total_metrics_attempted else "PARTIAL"
        else:
            audit["Fundamental_Status"] = "INSUFFICIENT_DATA"
            audit["Data_Quality"] = "POOR"

        # ---------------------------------------------------------
        # TECHNICAL SCORING ENGINE (15 Pts Max)
        # ---------------------------------------------------------
        tech_score = 0.0
        latest = hist.iloc[-1]

        # Trend Alignment (5 Pts)
        if (latest['Close'] > latest['EMA_200']) and (latest['EMA_50'] > latest['EMA_200']):
            tech_score += 5.0

        # Relative ROC & 21 EMA Alignment (5 Pts)
        if rel_roc_pass:
            tech_score += 5.0

        # Breakout Expansion (5 Pts)
        high_52wk = hist['Close'].max()
        if (latest['Close'] >= 0.95 * high_52wk) and (latest['Volume'] > 1.2 * latest['Vol_20MA']):
            tech_score += 5.0

        audit["Tech_Score"] = tech_score
        audit["Total_Score"] = fund_score + tech_score

        # =========================================================
        # INTEGRATED DECISION & STAGE CLASSIFICATION MATRIX
        # =========================================================
        has_positive_ebitda = (ebitda_curr is not None and ebitda_curr > 0)

        # 1. STAGE 2 MULTI-BAGGER TRIGGER: High Growth + EBITDA Positive + Relative Momentum Breakout
        if is_emerging_growth and has_positive_ebitda and rel_roc_pass:
            audit["Action"] = "STAGE 2 MULTI-BAGGER ALLOCATION"

        # 2. STAGE 1 OBSERVE / SETUP: High Revenue Growth (>20%) but temporary EBITDA/ROE Drag (e.g. Zota Phase B)
        elif is_emerging_growth and not has_positive_ebitda:
            audit["Action"] = "STAGE 1 OBSERVE / SETUP (Pre-EBITDA Inflection)"

        # 3. FULL BUY SIGNAL: Proven High-Quality Quant Compounder (Score >= 75)
        elif audit["Fundamental_Status"] == "VALID" and fund_score >= 60.0 and audit["Total_Score"] >= 75.0 and rel_roc_pass:
            audit["Action"] = "FULL BUY SIGNAL"

        # 4. WATCHLIST (Base Building): Strong Fundamentals, awaiting Technical / Rel ROC Confirmation
        elif audit["Fundamental_Status"] == "VALID" and fund_score >= 55.0:
            audit["Action"] = "WATCHLIST (Base Building)"

        else:
            audit["Action"] = "REJECT"

        return audit

    except Exception as e:
        audit["Fundamental_Status"] = "CALCULATION_ERROR"
        audit["Error_Reason"] = f"Unhandled Exception: {str(e)}"
        return audit

def send_telegram_alert(df):
    """Dispatches markdown formatted summary to Telegram."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Telegram tokens not configured. Skipping alert notification.")
        return

    buys = df[df['Action'] == "FULL BUY SIGNAL"].sort_values(by="Total_Score", ascending=False)
    multibaggers = df[df['Action'] == "STAGE 2 MULTI-BAGGER ALLOCATION"].sort_values(by="Revenue_Growth_Pct", ascending=False)
    stage1_observe = df[df['Action'] == "STAGE 1 OBSERVE / SETUP (Pre-EBITDA Inflection)"].sort_values(by="Revenue_Growth_Pct", ascending=False)
    watchlist = df[df['Action'] == "WATCHLIST (Base Building)"].sort_values(by="Fund_Score", ascending=False)

    msg = f"📊 *UNIFIED QUANT SCREENER V3*\n"
    msg += f"• Evaluated Universe: {len(df)} stocks\n"
    msg += f"• Stage 2 Multi-Baggers: {len(multibaggers)}\n"
    msg += f"• Stage 1 Observe Setup: {len(stage1_observe)}\n"
    msg += f"• Proven Full Buy Signals: {len(buys)}\n\n"

    msg += "🔥 *STAGE 2 MULTI-BAGGER TRIGGERS (Rel ROC > 0 + Growth > 20%):*\n"
    if not multibaggers.empty:
        for _, r in multibaggers.head(10).iterrows():
            msg += f"• *{r['Ticker']}*: ₹{r['Close_Price']} | Rev Growth: +{r['Revenue_Growth_Pct']}% | Rel ROC(12): {r['Rel_ROC_12']} | Score: {r['Total_Score']}\n"
    else:
        msg += "None this run.\n"

    msg += "\n👀 *STAGE 1 OBSERVE (Pre-EBITDA Inflection Setups):*\n"
    if not stage1_observe.empty:
        for _, r in stage1_observe.head(10).iterrows():
            msg += f"• *{r['Ticker']}*: ₹{r['Close_Price']} | Rev Growth: +{r['Revenue_Growth_Pct']}% | Rel ROC(12): {r['Rel_ROC_12']}\n"
    else:
        msg += "None.\n"

    msg += "\n🚀 *PROVEN FULL BUY SIGNALS (Score ≥ 75):*\n"
    if not buys.empty:
        for _, r in buys.head(10).iterrows():
            msg += f"• *{r['Ticker']}*: ₹{r['Close_Price']} | Total: {r['Total_Score']} (F: {r['Fund_Score']}, T: {r['Tech_Score']}) | Rel ROC: {r['Rel_ROC_12']}\n"
    else:
        msg += "None.\n"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
        print("Telegram notification dispatched successfully.")
    except Exception as e:
        print(f"Failed to post Telegram alert: {e}")

def sync_to_google_sheets(df):
    """Syncs results to Google Sheets with tab overwrite and multi-color stage formatting."""
    creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    spreadsheet_id = os.environ.get("SPREADSHEET_ID")

    if not creds_json or not spreadsheet_id:
        print("Google Sheets API credentials not configured. Skipping Sheets sync.")
        return

    try:
        creds_dict = json.loads(creds_json)
        gc = gspread.service_account_from_dict(creds_dict)
        sh = gc.open_by_key(spreadsheet_id)

        month_tab_name = datetime.now().strftime("%Y-%m")

        try:
            worksheet = sh.worksheet(month_tab_name)
            worksheet.clear()
            print(f"Overwriting existing worksheet tab: [{month_tab_name}]")
        except gspread.WorksheetNotFound:
            worksheet = sh.add_worksheet(title=month_tab_name, rows=str(len(df) + 50), cols="22")
            print(f"Created new worksheet tab: [{month_tab_name}]")

        upload_df = df.copy().fillna("")
        worksheet.update([upload_df.columns.values.tolist()] + upload_df.values.tolist())

        # Styling and Highlights
        header_format = CellFormat(
            backgroundColor=Color(0.2, 0.2, 0.2),
            textFormat=TextFormat(bold=True, color=Color(1, 1, 1))
        )
        format_cell_ranges(worksheet, [('1:1', header_format)])

        rules = get_conditional_format_rules(worksheet)
        
        # Stage 2 Multi-Bagger Rule (Purple)
        multibagger_rule = ConditionalFormatRule(
            ranges=[gspread.utils.a1_range_to_grid_range(f"A2:Z{len(df)+1}")],
            booleanRule=BooleanRule(
                condition=BooleanCondition('TEXT_EQ', ['STAGE 2 MULTI-BAGGER ALLOCATION']),
                format=CellFormat(
                    backgroundColor=Color(0.9, 0.85, 0.98),
                    textFormat=TextFormat(bold=True, color=Color(0.3, 0.0, 0.5))
                )
            )
        )

        # Full Buy Signal Rule (Green)
        buy_rule = ConditionalFormatRule(
            ranges=[gspread.utils.a1_range_to_grid_range(f"A2:Z{len(df)+1}")],
            booleanRule=BooleanRule(
                condition=BooleanCondition('TEXT_EQ', ['FULL BUY SIGNAL']),
                format=CellFormat(
                    backgroundColor=Color(0.85, 0.95, 0.85),
                    textFormat=TextFormat(bold=True, color=Color(0.0, 0.5, 0.0))
                )
            )
        )

        # Stage 1 Observe Rule (Yellow/Orange)
        stage1_rule = ConditionalFormatRule(
            ranges=[gspread.utils.a1_range_to_grid_range(f"A2:Z{len(df)+1}")],
            booleanRule=BooleanRule(
                condition=BooleanCondition('TEXT_EQ', ['STAGE 1 OBSERVE / SETUP (Pre-EBITDA Inflection)']),
                format=CellFormat(
                    backgroundColor=Color(1.0, 0.92, 0.8),
                    textFormat=TextFormat(bold=True, color=Color(0.7, 0.3, 0.0))
                )
            )
        )

        rules.append(multibagger_rule)
        rules.append(buy_rule)
        rules.append(stage1_rule)
        rules.save()

        print(f"Successfully synced {len(df)} rows to Google Sheets tab [{month_tab_name}].")

    except Exception as e:
        print(f"Failed to sync with Google Sheets: {e}")

if __name__ == "__main__":
    universe = get_nifty_500_tickers()
    benchmark_series = fetch_benchmark_data("^NSEI")
    results = []

    print(f"Executing Unified Quant Screener V3 across {len(universe)} tickers...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(evaluate_quant_model_v3, ticker, benchmark_series): ticker for ticker in universe}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)

    if results:
        df_res = pd.DataFrame(results)
        os.makedirs("output", exist_ok=True)
        
        # Update persistent historical qualification / inflection memory
        history_df = update_history(df_res)
        print(f"Historical memory updated: {len(history_df)} stocks tracked.")

        # Save CSV Artifact
        df_res.to_csv("output/quant_screener_results.csv", index=False)
        print("\nScreening Complete. File saved to output/quant_screener_results.csv")
        
        # Console Output
        active = df_res[df_res['Action'] != "REJECT"].sort_values(by="Total_Score", ascending=False)
        print("\n--- QUALIFIED SCREENER CANDIDATES ---")
        print(active[["Ticker", "Fundamental_Status", "Revenue_Growth_Pct", "Rel_ROC_12", "Fund_Score", "Total_Score", "Action"]])
        
        # Alerts & Storage
        send_telegram_alert(df_res)
        sync_to_google_sheets(df_res)
    else:
        print("Screening Complete. No valid output generated.")
