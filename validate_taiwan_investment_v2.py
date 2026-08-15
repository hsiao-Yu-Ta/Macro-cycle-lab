
"""
Macro Cycle Lab - Taiwan Investment Decision Validation V2

Non-destructive test version.
Keeps the existing model untouched and writes new V2 result files only.

Main improvements
1) Conservative publication-lag proxy:
   Macro data are shifted by 1 month before use to reduce look-ahead bias.
2) Automatic TAIEX drawdown-event detection instead of hand-written crisis windows.
3) Warning threshold / persistence grid search using TRAINING PERIOD ONLY.
4) Deep-bear price override:
   -20%, -25%, -30%, -35% drawdowns are evaluated separately from macro risk.
5) Static vs dynamic Macro/Market weights:
   65/35, 70/30, 75/25, plus dynamic weighting.
6) Investment-oriented metrics:
   false alarms, event capture, lead time, recovery lag,
   forward returns after deep drawdowns, CAGR/MaxDD/Sharpe for a simple risk-overlay test.

Inputs
------
data/processed/taiwan_macro_inputs.csv
data/raw/ndc_business_cycle.xlsx

Outputs
-------
data/processed/v2_model_grid.csv
data/processed/v2_drawdown_events.csv
data/processed/v2_event_capture.csv
data/processed/v2_false_alarms.csv
data/processed/v2_deep_bear_forward_returns.csv
data/processed/v2_strategy_metrics.csv
data/processed/v2_final_ranking.csv
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

STATIC_MIXES = {
    "65_35": (0.65, 0.35),
    "70_30": (0.70, 0.30),
    "75_25": (0.75, 0.25),
}

THRESHOLD_Q = [0.10, 0.15, 0.20, 0.25, 0.30]
PERSISTENCE = [1, 2]
EVENT_DD_THRESHOLD = -15.0
FALSE_ALARM_FWD_DD = -10.0
DEEP_BEAR_LEVELS = [-20, -25, -30, -35]


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
    """
    Conservative monthly availability proxy.

    The source files are month-labelled, but most macro releases are not known on
    the first day of that same month. To avoid treating month t data as instantly
    investable, all macro inputs are shifted by 1 month.

    This is intentionally conservative and can later be replaced by exact release
    calendars if desired.
    """
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


def build_macro_score(df):
    bases = {
        "ndc": "ndc_leading",
        "orders_total": "export_orders_total_usd_mn",
        "orders_elec": "export_orders_electronics_usd_mn",
        "orders_ict": "export_orders_ict_usd_mn",
        "mfg_elec": "mfg_info_electronics_index",
    }
    factor_w = {
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
            factor_w[k] * df[f"{k}_z{h}"] for k in factor_w
        )

    df["macro_score"] = 0.40 * df["macro_1m"] + 0.60 * df["macro_3m"]
    return df


def build_market_score(df):
    raw = {
        "m3": "taiex_return_3m_pct",
        "m6": "taiex_return_6m_pct",
        "ma6": "taiex_vs_ma_6m_pct",
        "ma12": "taiex_vs_ma_12m_pct",
    }
    for k, c in raw.items():
        df[f"market_{k}_z"] = rolling_z(pd.to_numeric(df[c], errors="coerce"))

    df["market_score"] = (
        0.25 * df["market_m3_z"]
        + 0.25 * df["market_m6_z"]
        + 0.25 * df["market_ma6_z"]
        + 0.25 * df["market_ma12_z"]
    )
    return df


def add_drawdown(df):
    close = pd.to_numeric(df["taiex_close"], errors="coerce")
    running_peak = close.cummax()
    df["taiex_drawdown_pct"] = (close / running_peak - 1) * 100
    return df


def add_model_scores(df):
    for name, (mw, sw) in STATIC_MIXES.items():
        df[f"score_{name}"] = mw * df["macro_score"] + sw * df["market_score"]

    # Dynamic model:
    # normal: 70/30
    # correction below -15%: 60/40 to demand more market confirmation
    # deep bear below -30%: return to 70/30 for risk score, while price override
    # is handled separately so that very bearish market momentum does not block buying.
    mw = pd.Series(0.70, index=df.index)
    sw = pd.Series(0.30, index=df.index)

    correction = (df["taiex_drawdown_pct"] <= -15) & (df["taiex_drawdown_pct"] > -30)
    mw.loc[correction] = 0.60
    sw.loc[correction] = 0.40

    df["score_dynamic"] = mw * df["macro_score"] + sw * df["market_score"]
    df["dynamic_macro_weight"] = mw
    df["dynamic_market_weight"] = sw
    return df


def future_max_drawdown(close, horizon=6):
    vals = close.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        if not np.isfinite(vals[i]):
            continue
        j = min(len(vals), i + horizon + 1)
        future = vals[i:j]
        if len(future) < 2:
            continue
        out[i] = (np.nanmin(future) / vals[i] - 1) * 100
    return pd.Series(out, index=close.index)


def detect_drawdown_events(df, threshold=-15.0):
    """
    Automatically identify non-overlapping peak-to-trough events.
    Event begins when drawdown first crosses threshold.
    Peak is the running high immediately before the event.
    Trough is the minimum before recovery to prior peak, or dataset end.
    """
    d = df[["date", "taiex_close"]].dropna().copy().reset_index(drop=True)
    prices = d["taiex_close"].astype(float).to_numpy()
    dates = d["date"].to_numpy()

    events = []
    peak_i = 0
    i = 1

    while i < len(d):
        if prices[i] >= prices[peak_i]:
            peak_i = i
            i += 1
            continue

        dd = (prices[i] / prices[peak_i] - 1) * 100
        if dd > threshold:
            i += 1
            continue

        # threshold crossed: define event
        cross_i = i
        trough_i = i
        j = i + 1

        while j < len(d):
            if prices[j] < prices[trough_i]:
                trough_i = j
            # recovered to previous peak
            if prices[j] >= prices[peak_i]:
                break
            j += 1

        recovery_i = j if j < len(d) else None

        events.append({
            "event_id": len(events) + 1,
            "peak_date": pd.Timestamp(dates[peak_i]),
            "cross_date": pd.Timestamp(dates[cross_i]),
            "trough_date": pd.Timestamp(dates[trough_i]),
            "recovery_date": pd.Timestamp(dates[recovery_i]) if recovery_i is not None else pd.NaT,
            "peak_close": prices[peak_i],
            "trough_close": prices[trough_i],
            "drawdown_pct": (prices[trough_i] / prices[peak_i] - 1) * 100,
        })

        if recovery_i is None:
            break

        peak_i = recovery_i
        i = recovery_i + 1

    return pd.DataFrame(events)


def persistent_warning(s, threshold, months):
    raw = s <= threshold
    if months == 1:
        return raw
    return raw.rolling(months).sum().ge(months)


def evaluate_grid(df):
    train = df[(df.date >= TRAIN_START) & (df.date <= TRAIN_END)].copy()
    val = df[(df.date >= VAL_START) & (df.date <= VAL_END)].copy()
    test = df[(df.date >= TEST_START) & (df.date <= TEST_END)].copy()

    rows = []
    model_names = list(STATIC_MIXES) + ["dynamic"]

    for model in model_names:
        score_col = f"score_{model}"
        train_scores = train[score_col].dropna()

        for q in THRESHOLD_Q:
            threshold = train_scores.quantile(q)

            for persistence in PERSISTENCE:
                row = {
                    "model": model,
                    "threshold_quantile": q,
                    "persistence_months": persistence,
                    "threshold_value": threshold,
                }

                for label, part in [("train", train), ("validation", val), ("test", test)]:
                    w = persistent_warning(part[score_col], threshold, persistence)
                    valid = part["future_max_dd_6m_pct"].notna() & part[score_col].notna()
                    w = w & valid

                    warning_n = int(w.sum())
                    true_n = int((w & (part["future_max_dd_6m_pct"] <= FALSE_ALARM_FWD_DD)).sum())
                    false_n = warning_n - true_n

                    row[f"{label}_warning_months"] = warning_n
                    row[f"{label}_true_warning_months"] = true_n
                    row[f"{label}_false_alarm_rate_pct"] = (
                        100 * false_n / warning_n if warning_n else np.nan
                    )
                    row[f"{label}_avg_fwd6_max_dd_pct"] = (
                        part.loc[w, "future_max_dd_6m_pct"].mean() if warning_n else np.nan
                    )

                # Selection objective emphasizes validation/test-style behavior,
                # but test is NOT used to select the threshold later.
                row["train_quality"] = (
                    -0.07 * row["train_false_alarm_rate_pct"]
                    -0.15 * row["train_avg_fwd6_max_dd_pct"]
                )
                row["validation_quality"] = (
                    -0.07 * row["validation_false_alarm_rate_pct"]
                    -0.15 * row["validation_avg_fwd6_max_dd_pct"]
                )
                rows.append(row)

    return pd.DataFrame(rows)


def choose_configs(grid):
    """
    Choose one threshold/persistence per model using training first,
    then use validation only as tie-breaker. Test period remains untouched.
    """
    chosen = []
    for model, g in grid.groupby("model"):
        g = g.copy()
        g["selection_score"] = 0.70 * g["train_quality"] + 0.30 * g["validation_quality"]
        chosen.append(g.sort_values("selection_score", ascending=False).iloc[0])
    return pd.DataFrame(chosen).reset_index(drop=True)


def event_capture(df, events, chosen):
    rows = []
    for _, ev in events.iterrows():
        peak = ev["peak_date"]
        trough = ev["trough_date"]

        pre_start = peak - pd.DateOffset(months=12)
        pre = df[(df.date >= pre_start) & (df.date <= peak)].copy()
        around_bottom = df[
            (df.date >= trough - pd.DateOffset(months=2))
            & (df.date <= trough + pd.DateOffset(months=2))
        ].copy()

        for _, cfg in chosen.iterrows():
            model = cfg["model"]
            threshold = cfg["threshold_value"]
            persistence = int(cfg["persistence_months"])
            score_col = f"score_{model}"

            pre["warn"] = persistent_warning(pre[score_col], threshold, persistence)
            warning_dates = pre.loc[pre["warn"], "date"]
            first_warn = warning_dates.iloc[0] if len(warning_dates) else pd.NaT
            last_warn = warning_dates.iloc[-1] if len(warning_dates) else pd.NaT

            if pd.notna(first_warn):
                lead = (peak.year - first_warn.year) * 12 + (peak.month - first_warn.month)
            else:
                lead = np.nan

            # Recovery = first month after trough with score above training median.
            train_median = df[
                (df.date >= TRAIN_START) & (df.date <= TRAIN_END)
            ][score_col].median()
            post = df[df.date >= trough]
            rec_dates = post.loc[post[score_col] >= train_median, "date"]
            rec = rec_dates.iloc[0] if len(rec_dates) else pd.NaT
            lag = (
                (rec.year - trough.year) * 12 + (rec.month - trough.month)
                if pd.notna(rec) else np.nan
            )

            severe_near_bottom = np.nan
            if len(around_bottom):
                severe_threshold = df[
                    (df.date >= TRAIN_START) & (df.date <= TRAIN_END)
                ][score_col].quantile(0.15)
                severe_near_bottom = 100 * (around_bottom[score_col] <= severe_threshold).mean()

            rows.append({
                "event_id": ev["event_id"],
                "model": model,
                "peak_date": peak,
                "trough_date": trough,
                "drawdown_pct": ev["drawdown_pct"],
                "first_warning_date": first_warn,
                "last_warning_date": last_warn,
                "warning_lead_months": lead,
                "severe_near_bottom_pct": severe_near_bottom,
                "recovery_date": rec,
                "recovery_lag_months": lag,
            })
    return pd.DataFrame(rows)


def deep_bear_forward_returns(df):
    rows = []
    d = df.copy()

    for level in DEEP_BEAR_LEVELS:
        hit = d["taiex_drawdown_pct"] <= level
        first_hit = hit & (~hit.shift(1, fill_value=False))

        for idx in d.index[first_hit]:
            row = {
                "date": d.loc[idx, "date"],
                "drawdown_level": level,
                "actual_drawdown_pct": d.loc[idx, "taiex_drawdown_pct"],
            }
            for h in [3, 6, 12, 24]:
                if idx + h < len(d):
                    row[f"forward_{h}m_return_pct"] = (
                        d.loc[idx + h, "taiex_close"] / d.loc[idx, "taiex_close"] - 1
                    ) * 100
                else:
                    row[f"forward_{h}m_return_pct"] = np.nan
            rows.append(row)

    return pd.DataFrame(rows)


def simple_strategy_metrics(df, chosen):
    """
    Simple diagnostic overlay, not a recommended trading strategy.

    Exposure:
      1.00 normally
      0.70 when warning is active
      deep-bear override -> 1.00 whenever drawdown <= -30%

    This tests whether risk warnings reduce drawdown without causing excessive
    opportunity loss. No transaction costs/taxes are included.
    """
    d = df.copy()
    d["market_return"] = d["taiex_close"].pct_change()

    rows = []
    for _, cfg in chosen.iterrows():
        model = cfg["model"]
        threshold = cfg["threshold_value"]
        persistence = int(cfg["persistence_months"])
        score_col = f"score_{model}"

        warning = persistent_warning(d[score_col], threshold, persistence)
        exposure = pd.Series(1.0, index=d.index)
        exposure.loc[warning] = 0.70

        # Price override: do not stay underweight in a deep bear.
        exposure.loc[d["taiex_drawdown_pct"] <= -30] = 1.00

        strat_ret = exposure.shift(1).fillna(1.0) * d["market_return"].fillna(0)
        equity = (1 + strat_ret).cumprod()
        benchmark = (1 + d["market_return"].fillna(0)).cumprod()

        years = max((d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25, 1)
        cagr = equity.iloc[-1] ** (1 / years) - 1
        bench_cagr = benchmark.iloc[-1] ** (1 / years) - 1

        dd = equity / equity.cummax() - 1
        bench_dd = benchmark / benchmark.cummax() - 1

        ann_vol = strat_ret.std() * np.sqrt(12)
        sharpe = (strat_ret.mean() * 12) / ann_vol if ann_vol > 0 else np.nan

        rows.append({
            "model": model,
            "threshold_quantile": cfg["threshold_quantile"],
            "persistence_months": persistence,
            "strategy_cagr_pct": 100 * cagr,
            "benchmark_cagr_pct": 100 * bench_cagr,
            "strategy_max_drawdown_pct": 100 * dd.min(),
            "benchmark_max_drawdown_pct": 100 * bench_dd.min(),
            "strategy_sharpe_approx": sharpe,
        })

    return pd.DataFrame(rows)


def final_ranking(chosen, event_capture_df, strategy_df):
    test_cols = [
        "model",
        "test_false_alarm_rate_pct",
        "test_avg_fwd6_max_dd_pct",
    ]
    c = chosen[test_cols].copy()

    ev = (
        event_capture_df.groupby("model", as_index=False)
        .agg(
            avg_warning_lead_months=("warning_lead_months", "mean"),
            avg_recovery_lag_months=("recovery_lag_months", "mean"),
            avg_severe_near_bottom_pct=("severe_near_bottom_pct", "mean"),
            captured_events=("first_warning_date", lambda s: s.notna().sum()),
        )
    )

    r = c.merge(ev, on="model", how="left").merge(strategy_df, on="model", how="left")

    # Investment-useful composite:
    # reward event capture/lead & lower max drawdown,
    # penalize false alarms and delayed recovery.
    r["v2_score"] = (
        1.5 * r["captured_events"].fillna(0)
        + 0.25 * r["avg_warning_lead_months"].fillna(0)
        - 0.06 * r["test_false_alarm_rate_pct"].fillna(100)
        - 0.40 * r["avg_recovery_lag_months"].fillna(12)
        + 0.08 * (r["benchmark_max_drawdown_pct"] - r["strategy_max_drawdown_pct"]).fillna(0)
        + 0.10 * (r["strategy_cagr_pct"] - r["benchmark_cagr_pct"]).fillna(0)
    )
    return r.sort_values("v2_score", ascending=False)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    macro = pd.read_csv(INPUT, parse_dates=["date"])
    ndc = load_ndc(NDC_XLSX)

    df = ndc.merge(macro, on="date", how="inner").sort_values("date").reset_index(drop=True)
    df = df[df.date >= TRAIN_START].reset_index(drop=True)

    # The lag must be applied BEFORE calculating macro momentum.
    df = apply_publication_lag(df)
    df = build_macro_score(df)
    df = build_market_score(df)
    df = add_drawdown(df)
    df = add_model_scores(df)

    df["future_max_dd_6m_pct"] = future_max_drawdown(df["taiex_close"], horizon=6)

    events = detect_drawdown_events(df, threshold=EVENT_DD_THRESHOLD)
    grid = evaluate_grid(df)
    chosen = choose_configs(grid)
    capture = event_capture(df, events, chosen)
    deep = deep_bear_forward_returns(df)
    strategy = simple_strategy_metrics(df, chosen)
    ranking = final_ranking(chosen, capture, strategy)

    grid.to_csv(OUTDIR / "v2_model_grid.csv", index=False, encoding="utf-8-sig")
    events.to_csv(OUTDIR / "v2_drawdown_events.csv", index=False, encoding="utf-8-sig")
    capture.to_csv(OUTDIR / "v2_event_capture.csv", index=False, encoding="utf-8-sig")
    chosen.to_csv(OUTDIR / "v2_false_alarms.csv", index=False, encoding="utf-8-sig")
    deep.to_csv(OUTDIR / "v2_deep_bear_forward_returns.csv", index=False, encoding="utf-8-sig")
    strategy.to_csv(OUTDIR / "v2_strategy_metrics.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(OUTDIR / "v2_final_ranking.csv", index=False, encoding="utf-8-sig")

    print("===== V2 FINAL RANKING =====")
    print(ranking.round(3).to_string(index=False))
    print("\n===== SELECTED CONFIGS =====")
    print(chosen[[
        "model","threshold_quantile","persistence_months",
        "train_false_alarm_rate_pct","validation_false_alarm_rate_pct",
        "test_false_alarm_rate_pct"
    ]].round(3).to_string(index=False))
    print("\n===== AUTOMATIC DRAWDOWN EVENTS =====")
    print(events.round(2).to_string(index=False))
    print("\nSaved V2 outputs to data/processed/")


if __name__ == "__main__":
    main()
