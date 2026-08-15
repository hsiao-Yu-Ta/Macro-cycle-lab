
"""
Macro Cycle Lab - Taiwan Crisis / False-Alarm / Bottom-Reentry Validation

Prerequisites:
  python fetch_taiwan_full.py
  python backtest_taiwan_weights.py

Inputs:
  data/processed/taiwan_macro_inputs.csv
  data/raw/ndc_business_cycle.xlsx

Outputs:
  data/processed/taiwan_crisis_validation.csv
  data/processed/taiwan_false_alarm_summary.csv
  data/processed/taiwan_bottom_reentry.csv

Tests:
  - 65/35 vs 70/30 vs 75/25
  - warning lead time before major TAIEX drawdowns
  - false alarms
  - whether the model stays too defensive near bear-market bottoms
"""

from pathlib import Path
import numpy as np
import pandas as pd

INPUT = Path("data/processed/taiwan_macro_inputs.csv")
NDC_XLSX = Path("data/raw/ndc_business_cycle.xlsx")
OUTDIR = Path("data/processed")

MIXES = {"65_35": (0.65,0.35), "70_30": (0.70,0.30), "75_25": (0.75,0.25)}

# Approximate Taiwan-market stress windows; event statistics are calculated from
# the actual monthly TAIEX series inside each window.
EVENTS = {
    "Dotcom_2000_2002": ("2000-01-01","2003-06-01"),
    "GFC_2007_2009": ("2007-01-01","2010-06-01"),
    "EuroDebt_2011": ("2011-01-01","2012-12-01"),
    "ChinaSlowdown_2015_2016": ("2015-01-01","2016-12-01"),
    "TradeWar_2018": ("2018-01-01","2019-06-01"),
    "Covid_2020": ("2019-07-01","2020-12-01"),
    "RateHike_2022": ("2021-07-01","2023-06-01"),
}

def rolling_z(s, window=60, min_periods=24):
    mu = s.shift(1).rolling(window, min_periods=min_periods).mean()
    sd = s.shift(1).rolling(window, min_periods=min_periods).std()
    return (s-mu)/sd

def load_ndc(path):
    raw = pd.read_excel(path, sheet_name=0)
    raw = raw.iloc[1:].copy()
    raw["date"] = pd.to_datetime(raw.iloc[:,0], errors="coerce")
    rename = {}
    for c in raw.columns:
        s = str(c)
        if "景氣對策信號(分)" in s: rename[c] = "ndc_score"
        elif "領先指標不含趨勢" in s: rename[c] = "ndc_leading"
        elif "同時指標不含趨勢" in s: rename[c] = "ndc_coincident"
    raw = raw.rename(columns=rename)
    out = raw[["date","ndc_score","ndc_leading","ndc_coincident"]].copy()
    for c in out.columns[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def build_scores(df):
    bases = {
        "ndc":"ndc_leading",
        "orders_total":"export_orders_total_usd_mn",
        "orders_elec":"export_orders_electronics_usd_mn",
        "orders_ict":"export_orders_ict_usd_mn",
        "mfg_elec":"mfg_info_electronics_index",
    }
    fw = {"ndc":.50,"orders_total":.20,"orders_elec":.10,"orders_ict":.10,"mfg_elec":.10}
    for h in [1,3]:
        for k,c in bases.items():
            df[f"{k}_z{h}"] = rolling_z(pd.to_numeric(df[c], errors="coerce").pct_change(h)*100)
        df[f"macro_{h}m"] = sum(fw[k]*df[f"{k}_z{h}"] for k in fw)
    df["macro_score"] = .40*df["macro_1m"] + .60*df["macro_3m"]

    market_cols = {
        "m3":"taiex_return_3m_pct",
        "m6":"taiex_return_6m_pct",
        "ma6":"taiex_vs_ma_6m_pct",
        "ma12":"taiex_vs_ma_12m_pct",
    }
    for k,c in market_cols.items():
        df[f"market_{k}_z"] = rolling_z(pd.to_numeric(df[c], errors="coerce"))
    df["market_score"] = sum(.25*df[f"market_{k}_z"] for k in market_cols)

    for name,(mw,sw) in MIXES.items():
        df[f"score_{name}"] = mw*df["macro_score"] + sw*df["market_score"]
    return df

def train_thresholds(df):
    # Thresholds learned only from 2000-2015 to avoid peeking at later periods.
    train = df[(df.date >= "2000-01-01") & (df.date <= "2015-12-01")]
    thresholds = {}
    for name in MIXES:
        s = train[f"score_{name}"].dropna()
        thresholds[name] = {
            "warning": s.quantile(.30),  # weakest 30% = defensive warning
            "severe": s.quantile(.15),   # weakest 15% = severe warning
            "recovery": s.quantile(.50), # median recapture = recovery confirmation
        }
    return thresholds

def months_between(a,b):
    return (b.year-a.year)*12 + (b.month-a.month)

def event_table(df, thresholds):
    rows = []
    for event,(start,end) in EVENTS.items():
        e = df[(df.date>=start)&(df.date<=end)].dropna(subset=["taiex_close"]).copy()
        if e.empty: continue
        peak_i = e["taiex_close"].idxmax()
        peak_date = e.loc[peak_i,"date"]
        after_peak = e[e.date>=peak_date]
        trough_i = after_peak["taiex_close"].idxmin()
        trough_date = e.loc[trough_i,"date"]
        peak = e.loc[peak_i,"taiex_close"]
        trough = e.loc[trough_i,"taiex_close"]
        dd = (trough/peak-1)*100

        for name in MIXES:
            w = thresholds[name]["warning"]
            sev = thresholds[name]["severe"]
            rec = thresholds[name]["recovery"]
            pre = e[e.date<=peak_date]
            warn_dates = pre.loc[pre[f"score_{name}"]<=w,"date"]
            first_warn = warn_dates.iloc[-1] if len(warn_dates) else pd.NaT
            lead = months_between(first_warn, peak_date) if pd.notna(first_warn) else np.nan

            near_bottom = e[(e.date>=trough_date-pd.DateOffset(months=2)) &
                            (e.date<=trough_date+pd.DateOffset(months=2))]
            severe_near_bottom = (near_bottom[f"score_{name}"]<=sev).mean()*100 if len(near_bottom) else np.nan

            post = e[e.date>=trough_date]
            recover_dates = post.loc[post[f"score_{name}"]>=rec,"date"]
            recovery_date = recover_dates.iloc[0] if len(recover_dates) else pd.NaT
            recovery_lag = months_between(trough_date,recovery_date) if pd.notna(recovery_date) else np.nan

            rows.append({
                "event":event, "mix":name,
                "peak_date":peak_date.date(), "trough_date":trough_date.date(),
                "taiex_drawdown_pct":dd,
                "last_warning_before_peak": first_warn.date() if pd.notna(first_warn) else None,
                "warning_lead_months":lead,
                "severe_signal_near_bottom_pct":severe_near_bottom,
                "recovery_confirmation_date": recovery_date.date() if pd.notna(recovery_date) else None,
                "recovery_lag_months":recovery_lag,
            })
    return pd.DataFrame(rows)

def false_alarm_table(df, thresholds):
    # A warning is a false alarm when score is below warning threshold but
    # the following 6-month TAIEX return is not materially negative (> -5%).
    rows=[]
    d=df.copy()
    d["taiex_fwd6_pct"]=(d["taiex_close"].shift(-6)/d["taiex_close"]-1)*100
    for name in MIXES:
        valid=d[[f"score_{name}","taiex_fwd6_pct"]].dropna()
        warn=valid[valid[f"score_{name}"]<=thresholds[name]["warning"]]
        false=warn[warn["taiex_fwd6_pct"]>-5]
        rows.append({
            "mix":name,
            "warning_months":len(warn),
            "false_alarm_months":len(false),
            "false_alarm_rate_pct":100*len(false)/len(warn) if len(warn) else np.nan,
            "avg_fwd6_return_when_warning_pct":warn["taiex_fwd6_pct"].mean() if len(warn) else np.nan,
        })
    return pd.DataFrame(rows)

def bottom_table(events):
    return (events.groupby("mix",as_index=False)
            .agg(avg_warning_lead_months=("warning_lead_months","mean"),
                 avg_severe_signal_near_bottom_pct=("severe_signal_near_bottom_pct","mean"),
                 avg_recovery_lag_months=("recovery_lag_months","mean"),
                 avg_event_drawdown_pct=("taiex_drawdown_pct","mean")))

def main():
    macro=pd.read_csv(INPUT,parse_dates=["date"])
    ndc=load_ndc(NDC_XLSX)
    df=ndc.merge(macro,on="date",how="inner").sort_values("date")
    df=df[df.date>="2000-01-01"].copy()
    df=build_scores(df)
    thresholds=train_thresholds(df)

    events=event_table(df,thresholds)
    false=false_alarm_table(df,thresholds)
    bottoms=bottom_table(events)

    # Simple decision score: reward lead time, penalize false alarms and late recovery.
    summary=bottoms.merge(false,on="mix",how="left")
    summary["decision_score"] = (
        summary["avg_warning_lead_months"].fillna(0)
        - 0.08*summary["false_alarm_rate_pct"].fillna(100)
        - 0.50*summary["avg_recovery_lag_months"].fillna(12)
        - 0.03*summary["avg_severe_signal_near_bottom_pct"].fillna(100)
    )
    summary=summary.sort_values("decision_score",ascending=False)

    OUTDIR.mkdir(parents=True,exist_ok=True)
    events.to_csv(OUTDIR/"taiwan_crisis_validation.csv",index=False,encoding="utf-8-sig")
    false.to_csv(OUTDIR/"taiwan_false_alarm_summary.csv",index=False,encoding="utf-8-sig")
    summary.to_csv(OUTDIR/"taiwan_bottom_reentry.csv",index=False,encoding="utf-8-sig")

    print("===== Crisis / Investment Decision Ranking =====")
    print(summary.round(3).to_string(index=False))
    print("\n===== Event Detail =====")
    print(events.round(2).to_string(index=False))
    print("\nSaved 3 CSV files to data/processed/")

if __name__=="__main__":
    main()
