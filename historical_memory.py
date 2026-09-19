import os
from datetime import datetime, timezone
import pandas as pd

HISTORY_COLUMNS = [
    "Ticker",
    "First_Qualified_Date",
    "First_Qualified_Price",
    "First_Inflection_Date",
    "First_Inflection_Price",
    "Peak_Price",
    "Peak_Date",
    "Peak_Return_Pct",
    "Current_Return_From_Base_Pct",
    "Drawdown_From_Peak_Pct",
    "Historical_Status",
]

QUALIFYING_ACTION = "STAGE 2 MULTI-BAGGER ALLOCATION"


def _empty_history():
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def load_history(path="history/stock_history.csv"):
    if not os.path.exists(path):
        return _empty_history()
    try:
        df = pd.read_csv(path)
        for col in HISTORY_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        return df[HISTORY_COLUMNS].copy()
    except Exception:
        return _empty_history()


def update_history(current_df, path="history/stock_history.csv", run_date=None):
    """Persist Stage-2 qualification memory and price-run state.

    This first version deliberately starts memory from the first live run that
    qualifies a stock. It does not backfill historical fundamental events,
    avoiding look-ahead bias from annual statements whose publication dates
    are not available in the current data source.
    """
    history = load_history(path)
    today = run_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if current_df is None or current_df.empty:
        return history

    for _, row in current_df.iterrows():
        ticker = str(row.get("Ticker", "")).strip()
        if not ticker:
            continue

        try:
            price = float(row.get("Close_Price"))
        except (TypeError, ValueError):
            continue
        if price <= 0:
            continue

        matches = history.index[history["Ticker"].astype(str) == ticker].tolist()
        qualifying_now = row.get("Action") == QUALIFYING_ACTION

        if not matches:
            if not qualifying_now:
                continue
            new = {col: pd.NA for col in HISTORY_COLUMNS}
            new.update({
                "Ticker": ticker,
                "First_Qualified_Date": today,
                "First_Qualified_Price": price,
                "Peak_Price": price,
                "Peak_Date": today,
                "Peak_Return_Pct": 0.0,
                "Current_Return_From_Base_Pct": 0.0,
                "Drawdown_From_Peak_Pct": 0.0,
                "Historical_Status": "QUALIFIED_NOT_CONFIRMED",
            })
            history = pd.concat([history, pd.DataFrame([new])], ignore_index=True)
            continue

        i = matches[0]
        base = pd.to_numeric(pd.Series([history.at[i, "First_Qualified_Price"]]), errors="coerce").iloc[0]
        peak = pd.to_numeric(pd.Series([history.at[i, "Peak_Price"]]), errors="coerce").iloc[0]
        if pd.isna(base) or base <= 0:
            continue
        if pd.isna(peak) or peak <= 0:
            peak = base

        if price > peak:
            peak = price
            history.at[i, "Peak_Price"] = price
            history.at[i, "Peak_Date"] = today

        current_return = (price / base - 1.0) * 100.0
        peak_return = (peak / base - 1.0) * 100.0
        drawdown = (price / peak - 1.0) * 100.0 if peak > 0 else 0.0

        history.at[i, "Current_Return_From_Base_Pct"] = round(current_return, 2)
        history.at[i, "Peak_Return_Pct"] = round(peak_return, 2)
        history.at[i, "Drawdown_From_Peak_Pct"] = round(drawdown, 2)

        if pd.isna(history.at[i, "First_Inflection_Date"]) and peak_return >= 40.0:
            history.at[i, "First_Inflection_Date"] = history.at[i, "Peak_Date"]
            history.at[i, "First_Inflection_Price"] = peak
            history.at[i, "Historical_Status"] = "FIRST_INFLECTION_CONFIRMED"
        elif not pd.isna(history.at[i, "First_Inflection_Date"]):
            history.at[i, "Historical_Status"] = "POST_INFLECTION"

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    history.to_csv(path, index=False)
    return history
