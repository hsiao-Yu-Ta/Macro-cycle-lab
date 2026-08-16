"""
Macro Cycle Lab - QQQ Formal Validation V4
==========================================

Goal
----
Validate whether the U.S. cycle model is useful for QQQ reserve deployment,
without turning macro deterioration into an automatic sell signal.

Uses existing project modules:
- src.config.MODEL_SPECS
- src.data.fetch_public_macro / fetch_yahoo_monthly / build_market_features
- src.data.derive_proxy_indicators / merge_all_data
- src.model.run_model

Design
------
1. Existing U.S. model SCORE = macro-cycle base signal.
2. QQQ market confirmation:
   3M return, 6M return, vs 6M MA, vs 12M MA (equal weighted Z scores).
3. Formal composite:
   75% U.S. cycle score + 25% QQQ market confirmation.
4. 3M change = early warning.
5. 6M change = confirmation.
6. QQQ drawdown is a separate decision axis.
7. Core position is NEVER sold because of a macro quadrant.
   The backtest tests deployment of a 15% bear-market reserve.

Outputs
-------
data/processed/qqq_v4_monthly_signals.csv
data/processed/qqq_v4_strategy_metrics.csv
data/processed/qqq_v4_crisis_review.csv
data/processed/qqq_v4_drawdown_forward_returns.csv
data/processed/qqq_v4_leave_one_crisis_out.csv
data/processed/qqq_v4_final_summary.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.config import MODEL_SPECS
from src.data import (
    fetch_public_macro,
    fetch_yahoo_monthly,
    build_market_features,
    derive_proxy_indicators,
    merge_all_data,
)
from src.model import run_model

START = "2000-01-01"
END = "2099-12-31"
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

CORE = 0.85
RESERVE = 0.15
CASH_ANNUAL = 0.03
TC_BPS = 5.0

CRISES = {
    "DOTCOM_2000_2003": ("2000-03-01", "2003-12-31"),
    "GFC_2007_2009": ("2007-07-01", "2009-12-31"),
    "COVID_2020": ("2020-01-01", "2020-12-31"),
    "RATE_HIKE_2022": ("2021-11-01", "2023-06-30"),
}


def rolling_z(s, window=60, min_periods=24):
    s = pd.to_numeric(s, errors="coerce")
    mu = s.shift(1).rolling(window, min_periods=min_periods).mean()
    sd = s.shift(1).rolling(window, min_periods=min_periods).std()
    return (s - mu) / sd


def prepare():
    # Existing U.S. macro model.
    macro, status = fetch_public_macro(START, END)
    macro = derive_proxy_indicators(macro)

    # Conservative one-month availability lag for macro data.
    macro = macro.copy()
    macro_cols = [c for c in macro.columns if c != "date"]
    for c in macro_cols:
        macro[c] = pd.to_numeric(macro[c], errors="coerce").shift(1)

    # QQQ price and market features.
    qqq = fetch_yahoo_monthly("QQQ", START, END)
    feat = build_market_features(qqq)

    data = merge_all_data(macro, feat, None).sort_index()

    spec = MODEL_SPECS["US"]
    us_model = run_model(
        data,
        dict(spec["weights"]),
        dict(spec["directions"]),
        3,
    )

    df = us_model.frame.copy()
    df["cycle_score"] = pd.to_numeric(df["SCORE"], errors="coerce")

    # QQQ market confirmation constructed explicitly.
    # Prefer existing feature names if present; otherwise derive from PRICE.
    px = pd.to_numeric(df["PRICE"], errors="coerce")

    m3 = px.pct_change(3) * 100
    m6 = px.pct_change(6) * 100
    ma6 = px.rolling(6).mean()
    ma12 = px.rolling(12).mean()
    vs6 = (px / ma6 - 1) * 100
    vs12 = (px / ma12 - 1) * 100

    df["mkt_z3"] = rolling_z(m3)
    df["mkt_z6"] = rolling_z(m6)
    df["mkt_zma6"] = rolling_z(vs6)
    df["mkt_zma12"] = rolling_z(vs12)
    df["market_score"] = df[
        ["mkt_z3", "mkt_z6", "mkt_zma6", "mkt_zma12"]
    ].mean(axis=1)

    # Standardize existing U.S. cycle score before mixing.
    df["cycle_score_z"] = rolling_z(df["cycle_score"])
    df["formal_score"] = 0.75 * df["cycle_score_z"] + 0.25 * df["market_score"]

    df["mom3"] = df["formal_score"].diff(3)
    df["mom6"] = df["formal_score"].diff(6)

    # Drawdown from trailing peak. For reserve deployment we want the actual
    # investable price drawdown, not a macro-cycle event label.
    peak = px.cummax()
    df["drawdown_pct"] = (px / peak - 1) * 100

    for h in [3, 6, 12, 24]:
        df[f"fwd_{h}m_pct"] = (px.shift(-h) / px - 1) * 100

    df["phase_3m"] = np.select(
        [
            (df["formal_score"] >= 0) & (df["mom3"] >= 0),
            (df["formal_score"] >= 0) & (df["mom3"] < 0),
            (df["formal_score"] < 0) & (df["mom3"] < 0),
        ],
        ["STRONG_EXPANSION", "COOLING_EXPANSION", "CONTRACTION"],
        default="RECOVERY",
    )

    df["fast_slow"] = np.select(
        [
            (df["mom3"] >= 0) & (df["mom6"] >= 0),
            (df["mom3"] < 0) & (df["mom6"] >= 0),
            (df["mom3"] >= 0) & (df["mom6"] < 0),
        ],
        ["BOTH_UP", "3M_DOWN_6M_UP", "3M_UP_6M_DOWN"],
        default="BOTH_DOWN",
    )

    return df


def dd_reserve_fraction(dd):
    """Fraction of the 15% reserve deployed from price alone."""
    if pd.isna(dd):
        return 0.0
    if dd > -15:
        return 0.0
    if dd > -20:
        return 0.25
    if dd > -25:
        return 0.50
    if dd > -30:
        return 0.75
    return 1.00


def exposure_drawdown_only(df):
    frac = df["drawdown_pct"].map(dd_reserve_fraction)
    return CORE + RESERVE * frac


def exposure_3m(df):
    """
    3M early-warning strategy:
    core never sold.
    Reserve deployment uses drawdown, but 3M deterioration can delay early adds.
    At <= -30%, price overrides the macro veto.
    """
    out = []
    for _, r in df.iterrows():
        dd = r["drawdown_pct"]
        m3 = r["mom3"]

        base = dd_reserve_fraction(dd)
        if pd.isna(dd):
            frac = 0.0
        elif dd <= -30:
            frac = base
        elif pd.isna(m3):
            frac = 0.0
        elif m3 >= 0:
            frac = base
        else:
            frac = max(0.0, base - 0.25)
        out.append(CORE + RESERVE * frac)
    return pd.Series(out, index=df.index)


def exposure_3m6m(df):
    """
    3M warning + 6M confirmation:
    BOTH_UP: full price-based reserve tranche
    3M_DOWN_6M_UP: one step more conservative (short-term cooling)
    3M_UP_6M_DOWN: early recovery, allow one partial step
    BOTH_DOWN: conservative until deep drawdown
    """
    out = []
    for _, r in df.iterrows():
        dd = r["drawdown_pct"]
        m3, m6 = r["mom3"], r["mom6"]
        base = dd_reserve_fraction(dd)

        if pd.isna(dd):
            frac = 0.0
        elif dd <= -35:
            frac = 1.0
        elif pd.isna(m3) or pd.isna(m6):
            frac = 0.0
        elif m3 >= 0 and m6 >= 0:
            frac = base
        elif m3 < 0 and m6 >= 0:
            frac = max(0.0, base - 0.25)
        elif m3 >= 0 and m6 < 0:
            frac = max(0.25 if dd <= -20 else 0.0, base - 0.25)
        else:
            # both down: no veto after -30, but still staged
            frac = 0.0 if dd > -25 else (0.50 if dd > -30 else base)

        out.append(CORE + RESERVE * min(1.0, frac))
    return pd.Series(out, index=df.index)


def exposure_hybrid(df):
    """
    Final policy candidate:
    - No selling of the 85% core due to macro.
    - 3M = warning, 6M = confirmation.
    - Price increasingly dominates as drawdown deepens.
    """
    out = []
    for _, r in df.iterrows():
        dd = r["drawdown_pct"]
        fs = r["fast_slow"]

        if pd.isna(dd) or dd > -15:
            frac = 0.0
        elif dd > -20:
            frac = 0.25 if fs in ("BOTH_UP", "3M_UP_6M_DOWN") else 0.0
        elif dd > -25:
            if fs == "BOTH_UP":
                frac = 0.50
            elif fs == "3M_UP_6M_DOWN":
                frac = 0.25
            elif fs == "3M_DOWN_6M_UP":
                frac = 0.25
            else:
                frac = 0.0
        elif dd > -30:
            # price opportunity starts to override
            frac = 0.75 if fs != "BOTH_DOWN" else 0.50
        elif dd > -35:
            frac = 1.00 if fs in ("BOTH_UP", "3M_UP_6M_DOWN") else 0.75
        else:
            # Deep bear: macro has no veto, but reserve is still only 15%.
            frac = 1.00

        out.append(CORE + RESERVE * frac)
    return pd.Series(out, index=df.index)


def backtest(px, exposure):
    ret = px.pct_change().fillna(0.0)
    cash_m = (1 + CASH_ANNUAL) ** (1 / 12) - 1
    turnover = exposure.diff().abs().fillna(0.0)
    cost = turnover * (TC_BPS / 10000.0)
    strat_ret = exposure.shift(1).fillna(exposure.iloc[0]) * ret + (1 - exposure.shift(1).fillna(exposure.iloc[0])) * cash_m - cost
    equity = (1 + strat_ret).cumprod()
    return strat_ret, equity


def metrics(ret, equity):
    n = max(1, len(ret))
    years = n / 12
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = ret.std() * np.sqrt(12)
    sharpe = ((ret.mean() * 12) - CASH_ANNUAL) / vol if vol and np.isfinite(vol) else np.nan
    dd = equity / equity.cummax() - 1
    maxdd = dd.min()
    return {
        "CAGR": cagr,
        "annual_vol": vol,
        "Sharpe": sharpe,
        "max_drawdown": maxdd,
        "ending_multiple": equity.iloc[-1],
    }


def strategy_comparison(df):
    px = df["PRICE"].dropna()
    d = df.loc[px.index].copy()

    exposures = {
        "BUY_HOLD_100": pd.Series(1.0, index=d.index),
        "STATIC_85_15": pd.Series(CORE, index=d.index),
        "DRAWDOWN_ONLY": exposure_drawdown_only(d),
        "MACRO_3M": exposure_3m(d),
        "MACRO_3M_6M": exposure_3m6m(d),
        "HYBRID_FORMAL": exposure_hybrid(d),
    }

    rows = []
    for name, exp in exposures.items():
        ret, eq = backtest(d["PRICE"], exp)
        m = metrics(ret, eq)
        rows.append({
            "strategy": name,
            **m,
            "avg_exposure": exp.mean(),
            "min_exposure": exp.min(),
            "max_exposure": exp.max(),
            "allocation_changes": int((exp.diff().abs() > 1e-9).sum()),
        })

    return pd.DataFrame(rows), exposures


def crisis_id(dt):
    dt = pd.Timestamp(dt)
    for name, (s, e) in CRISES.items():
        if pd.Timestamp(s) <= dt <= pd.Timestamp(e):
            return name
    return "OTHER"


def crisis_review(df, exposures):
    rows = []
    for crisis, (s, e) in CRISES.items():
        x = df[(df.index >= s) & (df.index <= e)].dropna(subset=["PRICE"])
        if x.empty:
            continue

        peak = x["PRICE"].cummax()
        dd = x["PRICE"] / peak - 1
        trough_date = dd.idxmin()
        trough_dd = dd.min()

        for name, exp in exposures.items():
            ex = exp.reindex(x.index).dropna()
            rows.append({
                "crisis": crisis,
                "strategy": name,
                "trough_date": trough_date,
                "crisis_price_drawdown_pct": trough_dd * 100,
                "min_exposure": ex.min() if len(ex) else np.nan,
                "exposure_at_trough": ex.reindex([trough_date]).iloc[0] if trough_date in ex.index else np.nan,
                "avg_exposure": ex.mean() if len(ex) else np.nan,
            })
    return pd.DataFrame(rows)


def forward_return_table(df):
    d = df.copy()
    d["crisis"] = [crisis_id(i) for i in d.index]
    d["dd_bucket"] = pd.cut(
        d["drawdown_pct"],
        bins=[-100, -35, -30, -25, -20, -15, 0.0001],
        labels=["<=-35", "-30~-35", "-25~-30", "-20~-25", "-15~-20", "0~-15"],
        include_lowest=True,
    )

    return (
        d.dropna(subset=["dd_bucket"])
        .groupby(["dd_bucket", "fast_slow"], observed=True)
        .agg(
            observations=("PRICE", "count"),
            crises=("crisis", lambda s: s[s != "OTHER"].nunique()),
            avg_fwd_6m=("fwd_6m_pct", "mean"),
            avg_fwd_12m=("fwd_12m_pct", "mean"),
            median_fwd_12m=("fwd_12m_pct", "median"),
            avg_fwd_24m=("fwd_24m_pct", "mean"),
            positive_12m_rate=("fwd_12m_pct", lambda s: 100 * (s > 0).mean()),
        )
        .reset_index()
    )


def leave_one_crisis_out(df):
    rows = []
    for omitted in CRISES:
        d = df[[crisis_id(i) != omitted for i in df.index]].copy()
        m, _ = strategy_comparison(d)
        for _, r in m.iterrows():
            rows.append({
                "omitted_crisis": omitted,
                "strategy": r["strategy"],
                "CAGR": r["CAGR"],
                "Sharpe": r["Sharpe"],
                "max_drawdown": r["max_drawdown"],
            })
    return pd.DataFrame(rows)


def final_summary(metrics_df, loco):
    base = metrics_df.set_index("strategy")
    rows = []
    for strategy in ["DRAWDOWN_ONLY", "MACRO_3M", "MACRO_3M_6M", "HYBRID_FORMAL"]:
        if strategy not in base.index:
            continue
        l = loco[loco["strategy"] == strategy]
        rows.append({
            "strategy": strategy,
            "full_CAGR": base.loc[strategy, "CAGR"],
            "full_Sharpe": base.loc[strategy, "Sharpe"],
            "full_max_drawdown": base.loc[strategy, "max_drawdown"],
            "LOCO_min_CAGR": l["CAGR"].min(),
            "LOCO_min_Sharpe": l["Sharpe"].min(),
            "LOCO_worst_max_drawdown": l["max_drawdown"].min(),
            "beats_drawdown_only_sharpe": (
                base.loc[strategy, "Sharpe"] >= base.loc["DRAWDOWN_ONLY", "Sharpe"]
            ),
        })
    return pd.DataFrame(rows)


def main():
    df = prepare()
    df.to_csv(OUT / "qqq_v4_monthly_signals.csv", encoding="utf-8-sig")

    m, exposures = strategy_comparison(df)
    m.to_csv(OUT / "qqq_v4_strategy_metrics.csv", index=False, encoding="utf-8-sig")

    crises = crisis_review(df, exposures)
    crises.to_csv(OUT / "qqq_v4_crisis_review.csv", index=False, encoding="utf-8-sig")

    fw = forward_return_table(df)
    fw.to_csv(OUT / "qqq_v4_drawdown_forward_returns.csv", index=False, encoding="utf-8-sig")

    loco = leave_one_crisis_out(df)
    loco.to_csv(OUT / "qqq_v4_leave_one_crisis_out.csv", index=False, encoding="utf-8-sig")

    final = final_summary(m, loco)
    final.to_csv(OUT / "qqq_v4_final_summary.csv", index=False, encoding="utf-8-sig")

    print("===== QQQ V4 STRATEGY METRICS =====")
    print(m.round(4).to_string(index=False))
    print("\n===== QQQ V4 FINAL SUMMARY =====")
    print(final.round(4).to_string(index=False))
    print("\n===== QQQ V4 CRISIS REVIEW =====")
    print(crises.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
