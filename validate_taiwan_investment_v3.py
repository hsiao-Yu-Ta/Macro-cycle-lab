
"""
Macro Cycle Lab - Taiwan Investment Validation V3

Non-destructive V3 test.

Major changes from V2
---------------------
1) LOCAL-CYCLE drawdown engine
   - Each bear/correction cycle has its own local peak and local trough.
   - A cycle begins when local drawdown <= -15%.
   - A cycle ends when the market rebounds >= 20% from the cycle trough
     OR makes a new high above the cycle peak.
   - After recovery, the peak reference resets. This avoids anchoring
     all later drawdowns to the 2000 all-time high.

2) Macro x Drawdown matrix
   - Core model fixed at 75% Macro / 25% Market for the investment matrix.
   - Macro regimes:
       DETERIORATING
       BOTTOMING
       RECOVERING
       EXPANDING
   - Drawdown buckets:
       0~-10, -10~-15, -15~-20, -20~-25, -25~-30, -30~-35, <=-35
   - Calculates forward 3/6/12/24 month TAIEX returns for each cell.

3) Keeps conservative 1-month macro publication lag.

4) Also compares 65/35, 70/30, 75/25 one final time using
   the corrected local-cycle event engine.

Inputs
------
data/processed/taiwan_macro_inputs.csv
data/raw/ndc_business_cycle.xlsx

Outputs
-------
data/processed/v3_local_cycles.csv
data/processed/v3_model_event_comparison.csv
data/processed/v3_macro_drawdown_matrix.csv
data/processed/v3_macro_drawdown_samples.csv
data/processed/v3_deep_bear_summary.csv
data/processed/v3_final_summary.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

INPUT = Path("data/processed/taiwan_macro_inputs.csv")
NDC_XLSX = Path("data/raw/ndc_business_cycle.xlsx")
OUTDIR = Path("data/processed")

TRAIN_START, TRAIN_END = "2000-01-01", "2015-12-01"
VAL_START, VAL_END = "2016-01-01", "2020-12-01"
TEST_START, TEST_END = "2021-01-01", "2099-12-01"

MIXES = {
    "65_35": (0.65, 0.35),
    "70_30": (0.70, 0.30),
    "75_25": (0.75, 0.25),
}

EVENT_TRIGGER_DD = -15.0
RECOVERY_FROM_TROUGH = 20.0

DRAW_BUCKETS = [
    (-10, 999, "0_to_-10"),
    (-15, -10, "-10_to_-15"),
    (-20, -15, "-15_to_-20"),
    (-25, -20, "-20_to_-25"),
    (-30, -25, "-25_to_-30"),
    (-35, -30, "-30_to_-35"),
    (-999, -35, "le_-35"),
]


def rolling_z(s, window=60, min_periods=24):
    mu = s.shift(1).rolling(window, min_periods=min_periods).mean()
    sd = s.shift(1).rolling(window, min_periods=min_periods).std()
    return (s - mu) / sd


def load_ndc(path):
    raw = pd.read_excel(path, sheet_name=0)
    raw = raw.iloc[1:].copy()
    raw["date"] = pd.to_datetime(raw.iloc[:, 0], errors="coerce")

    rename = {}
    for c in raw.columns:
        s = str(c)
        if "景氣對策信號(分)" in s:
            rename[c] = "ndc_score"
        elif "領先指標不含趨勢" in s:
            rename[c] = "ndc_leading"
        elif "同時指標不含趨勢" in s:
            rename[c] = "ndc_coincident"

    raw = raw.rename(columns=rename)
    out = raw[["date", "ndc_score", "ndc_leading", "ndc_coincident"]].copy()
    for c in out.columns[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def apply_publication_lag(df):
    macro_cols = [
        "ndc_leading",
        "export_orders_total_usd_mn",
        "export_orders_electronics_usd_mn",
        "export_orders_ict_usd_mn",
        "mfg_info_electronics_index",
    ]
    for c in macro_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").shift(1)
    return df


def build_macro_market_scores(df):
    bases = {
        "ndc": "ndc_leading",
        "orders_total": "export_orders_total_usd_mn",
        "orders_elec": "export_orders_electronics_usd_mn",
        "orders_ict": "export_orders_ict_usd_mn",
        "mfg_elec": "mfg_info_electronics_index",
    }

    fw = {
        "ndc": 0.50,
        "orders_total": 0.20,
        "orders_elec": 0.10,
        "orders_ict": 0.10,
        "mfg_elec": 0.10,
    }

    for h in [1, 3]:
        for k, c in bases.items():
            mom = pd.to_numeric(df[c], errors="coerce").pct_change(h) * 100
            df[f"{k}_z{h}"] = rolling_z(mom)

        df[f"macro_{h}m"] = sum(
            fw[k] * df[f"{k}_z{h}"] for k in fw
        )

    df["macro_score"] = 0.40 * df["macro_1m"] + 0.60 * df["macro_3m"]

    market_cols = {
        "m3": "taiex_return_3m_pct",
        "m6": "taiex_return_6m_pct",
        "ma6": "taiex_vs_ma_6m_pct",
        "ma12": "taiex_vs_ma_12m_pct",
    }

    for k, c in market_cols.items():
        df[f"market_{k}_z"] = rolling_z(pd.to_numeric(df[c], errors="coerce"))

    df["market_score"] = (
        0.25 * df["market_m3_z"]
        + 0.25 * df["market_m6_z"]
        + 0.25 * df["market_ma6_z"]
        + 0.25 * df["market_ma12_z"]
    )

    for name, (mw, sw) in MIXES.items():
        df[f"score_{name}"] = mw * df["macro_score"] + sw * df["market_score"]

    return df


def local_cycle_engine(df):
    """
    Local-cycle drawdown engine.

    Outside event:
      track local peak.

    Event starts:
      price falls >=15% below local peak.

    During event:
      track trough.

    Event ends:
      - price returns to / exceeds old peak, OR
      - price rebounds >=20% from trough.

    After recovery:
      reset peak reference to recovery date price and begin a new cycle.
    """
    d = df[["date", "taiex_close"]].dropna().copy().reset_index(drop=True)
    price = d["taiex_close"].astype(float).to_numpy()

    local_dd = np.full(len(d), np.nan)
    cycle_id = np.full(len(d), np.nan)

    events = []
    peak_i = 0
    trough_i = 0
    in_event = False
    current_cycle = 0
    event_start_i = None

    for i in range(len(d)):
        if i == 0:
            local_dd[i] = 0.0
            continue

        if not in_event:
            if price[i] >= price[peak_i]:
                peak_i = i

            local_dd[i] = (price[i] / price[peak_i] - 1) * 100

            if local_dd[i] <= EVENT_TRIGGER_DD:
                in_event = True
                current_cycle += 1
                event_start_i = i
                trough_i = i
                cycle_id[i] = current_cycle

        else:
            cycle_id[i] = current_cycle
            local_dd[i] = (price[i] / price[peak_i] - 1) * 100

            if price[i] < price[trough_i]:
                trough_i = i

            rebound_from_trough = (price[i] / price[trough_i] - 1) * 100
            new_high_recovery = price[i] >= price[peak_i]
            rebound_recovery = rebound_from_trough >= RECOVERY_FROM_TROUGH

            if new_high_recovery or rebound_recovery:
                events.append({
                    "cycle_id": current_cycle,
                    "peak_date": d.loc[peak_i, "date"],
                    "trigger_date": d.loc[event_start_i, "date"],
                    "trough_date": d.loc[trough_i, "date"],
                    "recovery_date": d.loc[i, "date"],
                    "peak_close": price[peak_i],
                    "trough_close": price[trough_i],
                    "drawdown_pct": (price[trough_i] / price[peak_i] - 1) * 100,
                    "rebound_to_recovery_pct": rebound_from_trough,
                    "recovered_to_old_peak": bool(new_high_recovery),
                })

                # Reset local reference after a confirmed recovery.
                in_event = False
                peak_i = i
                trough_i = i
                event_start_i = None

    # Open cycle at dataset end
    if in_event:
        events.append({
            "cycle_id": current_cycle,
            "peak_date": d.loc[peak_i, "date"],
            "trigger_date": d.loc[event_start_i, "date"],
            "trough_date": d.loc[trough_i, "date"],
            "recovery_date": pd.NaT,
            "peak_close": price[peak_i],
            "trough_close": price[trough_i],
            "drawdown_pct": (price[trough_i] / price[peak_i] - 1) * 100,
            "rebound_to_recovery_pct": np.nan,
            "recovered_to_old_peak": False,
        })

    d["local_drawdown_pct"] = local_dd
    d["cycle_id"] = cycle_id

    return d, pd.DataFrame(events)


def attach_local_drawdown(df, local):
    return df.merge(
        local[["date", "local_drawdown_pct", "cycle_id"]],
        on="date",
        how="left"
    )


def train_thresholds(df):
    train = df[(df.date >= TRAIN_START) & (df.date <= TRAIN_END)]
    out = {}
    for model in MIXES:
        s = train[f"score_{model}"].dropna()
        # V2 showed the extreme 10% threshold materially reduced false alarms.
        out[model] = s.quantile(0.10)
    return out


def event_model_comparison(df, events, thresholds):
    rows = []

    for _, ev in events.iterrows():
        peak = ev["peak_date"]
        trough = ev["trough_date"]

        pre = df[
            (df.date >= peak - pd.DateOffset(months=12))
            & (df.date <= peak)
        ].copy()

        for model in MIXES:
            col = f"score_{model}"
            threshold = thresholds[model]

            warn_dates = pre.loc[pre[col] <= threshold, "date"]
            first_warn = warn_dates.iloc[0] if len(warn_dates) else pd.NaT
            last_warn = warn_dates.iloc[-1] if len(warn_dates) else pd.NaT

            lead = np.nan
            if pd.notna(first_warn):
                lead = (
                    (peak.year - first_warn.year) * 12
                    + peak.month - first_warn.month
                )

            # Recovery confirmation after trough = score > training median
            train_median = df[
                (df.date >= TRAIN_START) & (df.date <= TRAIN_END)
            ][col].median()

            post = df[df.date >= trough]
            rec_dates = post.loc[post[col] >= train_median, "date"]
            rec = rec_dates.iloc[0] if len(rec_dates) else pd.NaT

            lag = np.nan
            if pd.notna(rec):
                lag = (
                    (rec.year - trough.year) * 12
                    + rec.month - trough.month
                )

            rows.append({
                "cycle_id": ev["cycle_id"],
                "model": model,
                "peak_date": peak,
                "trough_date": trough,
                "drawdown_pct": ev["drawdown_pct"],
                "first_warning_date": first_warn,
                "last_warning_date": last_warn,
                "warning_lead_months": lead,
                "macro_recovery_date": rec,
                "macro_recovery_lag_months": lag,
            })

    return pd.DataFrame(rows)


def classify_macro_regime(df):
    """
    Regime is based on the 75/25 score because V2 selected it as the leading candidate.

    Uses TRAINING-period quantiles only.
    """
    train = df[(df.date >= TRAIN_START) & (df.date <= TRAIN_END)]
    score = train["score_75_25"].dropna()

    q25 = score.quantile(0.25)
    q50 = score.quantile(0.50)

    df["score_75_25_change_3m"] = df["score_75_25"].diff(3)

    conditions = [
        (df["score_75_25"] <= q25) & (df["score_75_25_change_3m"] < 0),
        (df["score_75_25"] <= q25) & (df["score_75_25_change_3m"] >= 0),
        (df["score_75_25"] > q25) & (df["score_75_25"] <= q50) & (df["score_75_25_change_3m"] >= 0),
    ]
    choices = [
        "DETERIORATING",
        "BOTTOMING",
        "RECOVERING",
    ]

    df["macro_regime"] = np.select(
        conditions,
        choices,
        default="EXPANDING"
    )

    return df


def draw_bucket(x):
    if pd.isna(x):
        return np.nan

    for low, high, label in DRAW_BUCKETS:
        if low < x <= high:
            return label

    return "0_to_-10"


def add_forward_returns(df):
    for h in [3, 6, 12, 24]:
        df[f"fwd_{h}m_return_pct"] = (
            df["taiex_close"].shift(-h) / df["taiex_close"] - 1
        ) * 100
    return df


def macro_drawdown_matrix(df):
    d = df.copy()
    d["drawdown_bucket"] = d["local_drawdown_pct"].map(draw_bucket)

    sample_cols = [
        "date",
        "local_drawdown_pct",
        "drawdown_bucket",
        "macro_regime",
        "score_75_25",
        "fwd_3m_return_pct",
        "fwd_6m_return_pct",
        "fwd_12m_return_pct",
        "fwd_24m_return_pct",
    ]

    samples = d[sample_cols].dropna(subset=["drawdown_bucket", "macro_regime"]).copy()

    matrix = (
        samples.groupby(["drawdown_bucket", "macro_regime"], as_index=False)
        .agg(
            observations=("date", "count"),
            avg_fwd_3m_return_pct=("fwd_3m_return_pct", "mean"),
            median_fwd_3m_return_pct=("fwd_3m_return_pct", "median"),
            avg_fwd_6m_return_pct=("fwd_6m_return_pct", "mean"),
            median_fwd_6m_return_pct=("fwd_6m_return_pct", "median"),
            avg_fwd_12m_return_pct=("fwd_12m_return_pct", "mean"),
            median_fwd_12m_return_pct=("fwd_12m_return_pct", "median"),
            avg_fwd_24m_return_pct=("fwd_24m_return_pct", "mean"),
            median_fwd_24m_return_pct=("fwd_24m_return_pct", "median"),
            positive_12m_rate_pct=(
                "fwd_12m_return_pct",
                lambda s: 100 * (s > 0).mean()
            ),
            positive_24m_rate_pct=(
                "fwd_24m_return_pct",
                lambda s: 100 * (s > 0).mean()
            ),
        )
    )

    return samples, matrix


def deep_bear_summary(samples):
    d = samples[samples["local_drawdown_pct"] <= -20].copy()

    if d.empty:
        return pd.DataFrame()

    # Summarize by rounded 5%-step threshold and macro regime.
    bins = [-100, -35, -30, -25, -20]
    labels = ["<=-35", "-30~-35", "-25~-30", "-20~-25"]
    d["deep_level"] = pd.cut(
        d["local_drawdown_pct"],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True
    )

    return (
        d.groupby(["deep_level", "macro_regime"], observed=True, as_index=False)
        .agg(
            observations=("date", "count"),
            avg_6m=("fwd_6m_return_pct", "mean"),
            avg_12m=("fwd_12m_return_pct", "mean"),
            avg_24m=("fwd_24m_return_pct", "mean"),
            median_12m=("fwd_12m_return_pct", "median"),
            positive_12m_rate_pct=(
                "fwd_12m_return_pct",
                lambda s: 100 * (s > 0).mean()
            ),
            positive_24m_rate_pct=(
                "fwd_24m_return_pct",
                lambda s: 100 * (s > 0).mean()
            ),
        )
    )


def final_summary(event_cmp, matrix):
    model_summary = (
        event_cmp.groupby("model", as_index=False)
        .agg(
            events=("cycle_id", "nunique"),
            events_with_prior_warning=("first_warning_date", lambda s: s.notna().sum()),
            avg_warning_lead_months=("warning_lead_months", "mean"),
            avg_macro_recovery_lag_months=("macro_recovery_lag_months", "mean"),
        )
    )

    model_summary["event_warning_capture_pct"] = (
        100
        * model_summary["events_with_prior_warning"]
        / model_summary["events"]
    )

    # V3 ranking emphasizes event capture first, then lead time and recovery speed.
    model_summary["v3_event_score"] = (
        0.08 * model_summary["event_warning_capture_pct"]
        + 0.30 * model_summary["avg_warning_lead_months"].fillna(0)
        - 0.40 * model_summary["avg_macro_recovery_lag_months"].fillna(12)
    )

    return model_summary.sort_values("v3_event_score", ascending=False)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    macro = pd.read_csv(INPUT, parse_dates=["date"])
    ndc = load_ndc(NDC_XLSX)

    df = (
        ndc.merge(macro, on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )
    df = df[df.date >= TRAIN_START].reset_index(drop=True)

    df = apply_publication_lag(df)
    df = build_macro_market_scores(df)

    local, events = local_cycle_engine(df)
    df = attach_local_drawdown(df, local)

    df = classify_macro_regime(df)
    df = add_forward_returns(df)

    thresholds = train_thresholds(df)
    event_cmp = event_model_comparison(df, events, thresholds)

    samples, matrix = macro_drawdown_matrix(df)
    deep = deep_bear_summary(samples)
    summary = final_summary(event_cmp, matrix)

    events.to_csv(
        OUTDIR / "v3_local_cycles.csv",
        index=False,
        encoding="utf-8-sig"
    )

    event_cmp.to_csv(
        OUTDIR / "v3_model_event_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    matrix.to_csv(
        OUTDIR / "v3_macro_drawdown_matrix.csv",
        index=False,
        encoding="utf-8-sig"
    )

    samples.to_csv(
        OUTDIR / "v3_macro_drawdown_samples.csv",
        index=False,
        encoding="utf-8-sig"
    )

    deep.to_csv(
        OUTDIR / "v3_deep_bear_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        OUTDIR / "v3_final_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("===== V3 LOCAL MARKET CYCLES =====")
    print(events.round(2).to_string(index=False))

    print("\n===== V3 MODEL EVENT COMPARISON =====")
    print(summary.round(3).to_string(index=False))

    print("\n===== V3 MACRO x DRAWDOWN MATRIX =====")
    print(matrix.round(2).to_string(index=False))

    print("\n===== V3 DEEP BEAR SUMMARY =====")
    print(deep.round(2).to_string(index=False))

    print("\nSaved V3 outputs to data/processed/")


if __name__ == "__main__":
    main()
