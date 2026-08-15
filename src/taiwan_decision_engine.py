
"""
Taiwan Macro Cycle Lab - Official Decision Engine v1.0

Locked research decisions:
- Macro/Market composite: 75% / 25%
- Macro data availability: conservative 1-month lag
- Macro factors:
    NDC leading 50%
    Export orders total 20%
    Electronics orders 10%
    ICT orders 10%
    Info-electronics production 10%
- Macro momentum: 1M 40% + 3M 60%
- Market layer:
    TAIEX 3M return 25%
    TAIEX 6M return 25%
    vs 6M MA 25%
    vs 12M MA 25%
- Drawdown is a SEPARATE decision axis; it is not folded back into the 75/25 score.
- Local-cycle engine resets after a 20% rebound from trough or a new local high.

Decision philosophy:
- 0~-15%: normal DCA; no bear-reserve deployment
- -15~-20%: observation / small adds
- -20~-25%: Macro regime has strong decision value
- -25~-30%: price starts to override Macro veto
- -30~-35%: price-led staged deployment
- <=-35%: Macro loses veto power, but evidence does NOT support one-shot all-in.
"""

from pathlib import Path
import numpy as np
import pandas as pd

TRAIN_START, TRAIN_END = "2000-01-01", "2015-12-01"

ACTION_RANK = {
    "NORMAL_DCA": 0,
    "OBSERVE": 1,
    "SMALL_ADD": 2,
    "ADD": 3,
    "AGGRESSIVE_ADD": 4,
    "HEAVY_STAGED_ADD": 5,
    "FORCED_STAGED_ADD": 6,
}

ACTION_LABEL_ZH = {
    "NORMAL_DCA": "正常定投",
    "OBSERVE": "觀察／保留彈藥",
    "SMALL_ADD": "小額加碼",
    "ADD": "加碼",
    "AGGRESSIVE_ADD": "積極加碼",
    "HEAVY_STAGED_ADD": "大幅分批加碼",
    "FORCED_STAGED_ADD": "深熊強制分批",
}

# Percent of REMAINING bear reserve suggested for a new trigger.
# These are portfolio-policy defaults, not estimates of expected return.
DEFAULT_TRANCHE_PCT = {
    "NORMAL_DCA": 0,
    "OBSERVE": 0,
    "SMALL_ADD": 5,
    "ADD": 10,
    "AGGRESSIVE_ADD": 15,
    "HEAVY_STAGED_ADD": 20,
    "FORCED_STAGED_ADD": 20,
}


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
    cols = ["date", "ndc_score", "ndc_leading", "ndc_coincident"]
    out = raw[[c for c in cols if c in raw.columns]].copy()
    for c in out.columns:
        if c != "date":
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


def build_scores(df):
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
        df[f"macro_{h}m"] = sum(fw[k] * df[f"{k}_z{h}"] for k in fw)

    df["macro_score"] = 0.40 * df["macro_1m"] + 0.60 * df["macro_3m"]

    market = {
        "m3": "taiex_return_3m_pct",
        "m6": "taiex_return_6m_pct",
        "ma6": "taiex_vs_ma_6m_pct",
        "ma12": "taiex_vs_ma_12m_pct",
    }
    for k, c in market.items():
        df[f"market_{k}_z"] = rolling_z(pd.to_numeric(df[c], errors="coerce"))

    df["market_score"] = (
        .25 * df["market_m3_z"]
        + .25 * df["market_m6_z"]
        + .25 * df["market_ma6_z"]
        + .25 * df["market_ma12_z"]
    )

    df["official_score"] = 0.75 * df["macro_score"] + 0.25 * df["market_score"]
    return df


def local_cycle_engine(df, trigger_dd=-15.0, recovery_from_trough=20.0):
    d = df[["date", "taiex_close"]].dropna().copy().reset_index(drop=True)
    price = d["taiex_close"].astype(float).to_numpy()

    local_dd = np.full(len(d), np.nan)
    cycle_id = np.full(len(d), np.nan)

    peak_i = 0
    trough_i = 0
    in_event = False
    current_cycle = 0

    for i in range(len(d)):
        if i == 0:
            local_dd[i] = 0.0
            continue

        if not in_event:
            if price[i] >= price[peak_i]:
                peak_i = i
            local_dd[i] = (price[i] / price[peak_i] - 1) * 100

            if local_dd[i] <= trigger_dd:
                in_event = True
                current_cycle += 1
                trough_i = i
                cycle_id[i] = current_cycle
        else:
            cycle_id[i] = current_cycle
            local_dd[i] = (price[i] / price[peak_i] - 1) * 100

            if price[i] < price[trough_i]:
                trough_i = i

            rebound = (price[i] / price[trough_i] - 1) * 100
            recovered = price[i] >= price[peak_i] or rebound >= recovery_from_trough

            if recovered:
                in_event = False
                peak_i = i
                trough_i = i

    d["local_drawdown_pct"] = local_dd
    d["cycle_id"] = cycle_id
    return d


def classify_regime(df):
    train = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)]
    s = train["official_score"].dropna()

    q25 = s.quantile(.25)
    q50 = s.quantile(.50)
    df["official_score_change_3m"] = df["official_score"].diff(3)

    conditions = [
        (df["official_score"] <= q25) & (df["official_score_change_3m"] < 0),
        (df["official_score"] <= q25) & (df["official_score_change_3m"] >= 0),
        (df["official_score"] > q25) & (df["official_score"] <= q50) & (df["official_score_change_3m"] >= 0),
    ]
    choices = ["DETERIORATING", "BOTTOMING", "RECOVERING"]

    df["macro_regime"] = np.select(conditions, choices, default="EXPANDING")
    return df


def decide_action(drawdown, regime):
    """Return action, confidence, and rationale."""
    if pd.isna(drawdown):
        return "NORMAL_DCA", "LOW", "回撤資料不足，維持正常定投。"

    # 0~-15%
    if drawdown > -15:
        return "NORMAL_DCA", "NORMAL", "尚未進入主要熊市加碼區，維持正常定投。"

    # -15~-20%
    if drawdown > -20:
        if regime in ["BOTTOMING", "RECOVERING"]:
            return "SMALL_ADD", "TENTATIVE", "回撤已達觀察區，Macro停止惡化，可小額加碼。"
        return "OBSERVE", "TENTATIVE", "回撤約15~20%，Macro尚未確認轉折，保留主要彈藥。"

    # -20~-25% -- V3.1 STRONG evidence for Macro discrimination
    if drawdown > -25:
        if regime == "DETERIORATING":
            return "OBSERVE", "STRONG", "V3.1顯示此區間Macro惡化時12M表現偏弱，應保守。"
        if regime == "BOTTOMING":
            return "ADD", "STRONG", "V3.1支持同跌幅下，Macro築底優於持續惡化。"
        if regime == "RECOVERING":
            return "AGGRESSIVE_ADD", "STRONG", "價格已深度修正且Macro復甦，屬高品質加碼組合。"
        return "SMALL_ADD", "STRONG", "價格已達20~25%修正，但尚未形成明確築底訊號。"

    # -25~-30% -- V3.1 STRONG price opportunity
    if drawdown > -30:
        if regime == "DETERIORATING":
            return "ADD", "STRONG", "V3.1支持此區間12M報酬轉為有利；Macro不再具有完全否決權。"
        return "AGGRESSIVE_ADD", "STRONG", "回撤25~30%且Macro非惡化，價格與景氣條件同步改善。"

    # -30~-35% -- limited direct robustness evidence
    if drawdown > -35:
        if regime == "DETERIORATING":
            return "AGGRESSIVE_ADD", "TENTATIVE", "價格主導，但樣本較少；採分批而非一次投入。"
        return "HEAVY_STAGED_ADD", "TENTATIVE", "深度回撤且Macro改善，可大幅分批，但避免一次All-in。"

    # <= -35% -- V3.1 says long-run opportunity plausible but robustness WEAK
    if regime == "DETERIORATING":
        return "FORCED_STAGED_ADD", "WEAK", "Macro取消否決權，但V3.1不支持無腦All-in；應強制分批。"
    return "HEAVY_STAGED_ADD", "TENTATIVE", "超深熊且Macro已非持續惡化，可強力分批，仍避免一次All-in。"


def build_official_history(macro_csv, ndc_xlsx):
    macro = pd.read_csv(macro_csv, parse_dates=["date"])
    ndc = load_ndc(ndc_xlsx)

    df = (
        ndc.merge(macro, on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )
    df = df[df["date"] >= TRAIN_START].reset_index(drop=True)

    df = apply_publication_lag(df)
    df = build_scores(df)

    local = local_cycle_engine(df)
    df = df.merge(local[["date", "local_drawdown_pct", "cycle_id"]], on="date", how="left")
    df = classify_regime(df)

    decisions = df.apply(
        lambda r: decide_action(r["local_drawdown_pct"], r["macro_regime"]),
        axis=1,
        result_type="expand"
    )
    decisions.columns = ["action", "confidence", "rationale"]
    df = pd.concat([df, decisions], axis=1)

    df["action_label_zh"] = df["action"].map(ACTION_LABEL_ZH)
    df["suggested_tranche_pct_of_remaining_reserve"] = df["action"].map(DEFAULT_TRANCHE_PCT)
    df["action_rank"] = df["action"].map(ACTION_RANK)

    prev_rank = df["action_rank"].shift(1).fillna(0)
    df["new_trigger"] = df["action_rank"] > prev_rank

    return df


def latest_signal(history):
    usable = history.dropna(subset=["official_score", "taiex_close"]).copy()
    if usable.empty:
        raise RuntimeError("No usable official signal rows.")
    return usable.iloc[-1].to_dict()
