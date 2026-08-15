
"""
Macro Cycle Lab - Taiwan Investment Robustness Test V3.1

Purpose
-------
Validate whether V3 investment conclusions remain robust after:
1) Treating 2000-2003 as ONE macro bear regime (not multiple independent cycles).
2) Leave-One-Crisis-Out (LOCO) tests.
3) Crisis-level bootstrap resampling.
4) Testing whether key investment rules remain directionally stable.

Prerequisites
-------------
Run V3 first so these files exist:
  data/processed/v3_macro_drawdown_samples.csv
  data/processed/v3_local_cycles.csv

Outputs
-------
data/processed/v31_crisis_tagged_samples.csv
data/processed/v31_rule_summary.csv
data/processed/v31_leave_one_crisis_out.csv
data/processed/v31_bootstrap_summary.csv
data/processed/v31_final_robustness.csv

Interpretation
--------------
The goal is NOT to optimize parameters again.
The goal is to test whether the V3 conclusions survive after removing any one
major crisis and after resampling crises as independent units.
"""

from pathlib import Path
import numpy as np
import pandas as pd

INDIR = Path("data/processed")
OUTDIR = Path("data/processed")

SAMPLES_FILE = INDIR / "v3_macro_drawdown_samples.csv"
CYCLES_FILE = INDIR / "v3_local_cycles.csv"

RANDOM_SEED = 42
BOOTSTRAP_N = 5000

# Independent macro-bear / correction regimes.
# 2000-2003 is intentionally treated as one event.
CRISIS_WINDOWS = {
    "DOTCOM_2000_2003": ("2000-01-01", "2003-12-31"),
    "CORRECTION_2004": ("2004-01-01", "2005-12-31"),
    "GFC_2008_2009": ("2007-07-01", "2009-12-31"),
    "EURO_DEBT_2011": ("2010-07-01", "2012-12-31"),
    "CHINA_SLOWDOWN_2015_2016": ("2015-01-01", "2016-12-31"),
    "TRADE_WAR_2018": ("2018-01-01", "2019-06-30"),
    "COVID_2020": ("2020-01-01", "2020-12-31"),
    "RATE_HIKE_2022": ("2021-07-01", "2023-06-30"),
}

# Rules to validate.
# Each rule selects observations and evaluates forward returns.
RULES = {
    "R1_-20_-25_Deteriorating": {
        "dd_min": -25,
        "dd_max": -20,
        "regimes": ["DETERIORATING"],
        "claim": "CAUTIOUS",
        "description": "-20~-25% 且 Macro 惡化：不宜重押，12M報酬應弱於同跌幅之築底/復甦。",
    },
    "R2_-20_-25_BottomingRecovering": {
        "dd_min": -25,
        "dd_max": -20,
        "regimes": ["BOTTOMING", "RECOVERING"],
        "claim": "STRONGER",
        "description": "-20~-25% 且 Macro 築底/復甦：未來12M應明顯優於 Macro 惡化。",
    },
    "R3_-25_-30_AnyMacro": {
        "dd_min": -30,
        "dd_max": -25,
        "regimes": ["DETERIORATING", "BOTTOMING", "RECOVERING", "EXPANDING"],
        "claim": "POSITIVE_12M",
        "description": "-25~-30%：即使短期仍可能下跌，12M中期報酬應轉為有利。",
    },
    "R4_le_-35_AnyMacro": {
        "dd_min": -100,
        "dd_max": -35,
        "regimes": ["DETERIORATING", "BOTTOMING", "RECOVERING", "EXPANDING"],
        "claim": "STRONG_LONG_TERM",
        "description": "<=-35%：價格優先，12M/24M長期報酬應大致維持正向。",
    },
}

def tag_crisis(date):
    if pd.isna(date):
        return "OTHER"
    d = pd.Timestamp(date)
    for name, (start, end) in CRISIS_WINDOWS.items():
        if pd.Timestamp(start) <= d <= pd.Timestamp(end):
            return name
    return "OTHER"


def select_rule(df, rule):
    dd = pd.to_numeric(df["local_drawdown_pct"], errors="coerce")
    mask = (
        (dd > rule["dd_min"])
        & (dd <= rule["dd_max"])
        & (df["macro_regime"].isin(rule["regimes"]))
    )
    return df.loc[mask].copy()


def aggregate_rule(df, rule_name, rule):
    d = select_rule(df, rule)

    if d.empty:
        return {
            "rule": rule_name,
            "observations": 0,
            "crises": 0,
            "avg_6m": np.nan,
            "avg_12m": np.nan,
            "median_12m": np.nan,
            "avg_24m": np.nan,
            "positive_12m_rate_pct": np.nan,
            "positive_24m_rate_pct": np.nan,
        }

    return {
        "rule": rule_name,
        "observations": len(d),
        "crises": d.loc[d["crisis_id"] != "OTHER", "crisis_id"].nunique(),
        "avg_6m": d["fwd_6m_return_pct"].mean(),
        "avg_12m": d["fwd_12m_return_pct"].mean(),
        "median_12m": d["fwd_12m_return_pct"].median(),
        "avg_24m": d["fwd_24m_return_pct"].mean(),
        "positive_12m_rate_pct": 100 * (d["fwd_12m_return_pct"] > 0).mean(),
        "positive_24m_rate_pct": 100 * (d["fwd_24m_return_pct"] > 0).mean(),
    }


def crisis_level_rule_returns(df, rule):
    """
    Convert monthly observations into one observation per crisis.
    This prevents long crises (e.g. 2000-03) from receiving excessive weight.
    """
    d = select_rule(df, rule)
    d = d[d["crisis_id"] != "OTHER"].copy()

    if d.empty:
        return pd.DataFrame(columns=[
            "crisis_id", "ret6", "ret12", "ret24", "n_months"
        ])

    g = (
        d.groupby("crisis_id", as_index=False)
        .agg(
            ret6=("fwd_6m_return_pct", "mean"),
            ret12=("fwd_12m_return_pct", "mean"),
            ret24=("fwd_24m_return_pct", "mean"),
            n_months=("date", "count"),
        )
    )
    return g


def leave_one_crisis_out(df):
    rows = []
    crises = sorted([x for x in df["crisis_id"].unique() if x != "OTHER"])

    for rule_name, rule in RULES.items():
        for omitted in crises:
            d = df[df["crisis_id"] != omitted].copy()
            agg = aggregate_rule(d, rule_name, rule)
            agg["omitted_crisis"] = omitted
            rows.append(agg)

    return pd.DataFrame(rows)


def bootstrap_rule(crisis_returns, n=BOOTSTRAP_N, seed=RANDOM_SEED):
    if crisis_returns.empty:
        return {}

    rng = np.random.default_rng(seed)
    vals12 = crisis_returns["ret12"].dropna().to_numpy()
    vals24 = crisis_returns["ret24"].dropna().to_numpy()

    out = {}

    if len(vals12):
        means12 = np.empty(n)
        pos12 = np.empty(n)
        for i in range(n):
            sample = rng.choice(vals12, size=len(vals12), replace=True)
            means12[i] = np.mean(sample)
            pos12[i] = np.mean(sample > 0)

        out.update({
            "bootstrap_crises_12m": len(vals12),
            "boot_avg12_mean": means12.mean(),
            "boot_avg12_p05": np.quantile(means12, 0.05),
            "boot_avg12_p50": np.quantile(means12, 0.50),
            "boot_avg12_p95": np.quantile(means12, 0.95),
            "boot_prob_avg12_positive_pct": 100 * np.mean(means12 > 0),
            "boot_positive12_rate_mean_pct": 100 * pos12.mean(),
        })

    if len(vals24):
        means24 = np.empty(n)
        pos24 = np.empty(n)
        for i in range(n):
            sample = rng.choice(vals24, size=len(vals24), replace=True)
            means24[i] = np.mean(sample)
            pos24[i] = np.mean(sample > 0)

        out.update({
            "bootstrap_crises_24m": len(vals24),
            "boot_avg24_mean": means24.mean(),
            "boot_avg24_p05": np.quantile(means24, 0.05),
            "boot_avg24_p50": np.quantile(means24, 0.50),
            "boot_avg24_p95": np.quantile(means24, 0.95),
            "boot_prob_avg24_positive_pct": 100 * np.mean(means24 > 0),
            "boot_positive24_rate_mean_pct": 100 * pos24.mean(),
        })

    return out


def bootstrap_all(df):
    rows = []

    for idx, (rule_name, rule) in enumerate(RULES.items()):
        crisis_ret = crisis_level_rule_returns(df, rule)
        stats = bootstrap_rule(
            crisis_ret,
            n=BOOTSTRAP_N,
            seed=RANDOM_SEED + idx
        )

        row = {
            "rule": rule_name,
            "independent_crises": len(crisis_ret),
        }
        row.update(stats)
        rows.append(row)

    return pd.DataFrame(rows)


def compare_r1_r2(df):
    """
    Directly compare -20~-25 deteriorating vs bottoming/recovering
    at the independent-crisis level when both are available.
    """
    a = crisis_level_rule_returns(df, RULES["R1_-20_-25_Deteriorating"])
    b = crisis_level_rule_returns(df, RULES["R2_-20_-25_BottomingRecovering"])

    merged = a[["crisis_id", "ret12"]].rename(columns={"ret12": "deteriorating_12m"}).merge(
        b[["crisis_id", "ret12"]].rename(columns={"ret12": "bottoming_recovering_12m"}),
        on="crisis_id",
        how="inner"
    )

    if merged.empty:
        return {
            "paired_crises": 0,
            "avg_advantage_bottoming_recovering_12m": np.nan,
            "share_bottoming_recovering_better_pct": np.nan,
        }

    merged["advantage"] = (
        merged["bottoming_recovering_12m"]
        - merged["deteriorating_12m"]
    )

    return {
        "paired_crises": len(merged),
        "avg_advantage_bottoming_recovering_12m": merged["advantage"].mean(),
        "share_bottoming_recovering_better_pct": 100 * (merged["advantage"] > 0).mean(),
    }


def robustness_grade(rule_name, full_row, loco_group, boot_row, r1r2_cmp):
    """
    Conservative qualitative grading.

    VERY_STRONG:
      supported by bootstrap + LOCO stability with enough crisis breadth.
    STRONG:
      direction holds but crisis count is limited.
    TENTATIVE:
      direction is interesting but sample dependence / CI remains wide.
    WEAK:
      conclusion is not robust.
    """
    independent = int(boot_row.get("independent_crises", 0))
    loco12 = loco_group["avg_12m"].dropna()

    loco_positive_share = (
        100 * (loco12 > 0).mean() if len(loco12) else np.nan
    )
    boot_prob12 = boot_row.get("boot_prob_avg12_positive_pct", np.nan)
    p05_12 = boot_row.get("boot_avg12_p05", np.nan)

    supported = False

    if rule_name == "R1_-20_-25_Deteriorating":
        # This rule claims caution, not necessarily negative 12M returns.
        # We validate it mainly against R2.
        paired = r1r2_cmp.get("paired_crises", 0)
        share_better = r1r2_cmp.get("share_bottoming_recovering_better_pct", np.nan)
        advantage = r1r2_cmp.get("avg_advantage_bottoming_recovering_12m", np.nan)
        supported = (
            paired >= 2
            and np.isfinite(share_better)
            and share_better >= 60
            and np.isfinite(advantage)
            and advantage > 0
        )
    else:
        supported = (
            np.isfinite(boot_prob12)
            and boot_prob12 >= 80
            and np.isfinite(loco_positive_share)
            and loco_positive_share >= 70
        )

    if supported and independent >= 5 and np.isfinite(p05_12) and p05_12 > 0:
        return "VERY_STRONG"
    elif supported and independent >= 3:
        return "STRONG"
    elif supported:
        return "TENTATIVE"
    else:
        return "WEAK"


def final_robustness(summary, loco, boot, r1r2_cmp):
    rows = []

    for rule_name, rule in RULES.items():
        full_row = summary[summary["rule"] == rule_name].iloc[0]
        loco_group = loco[loco["rule"] == rule_name]
        boot_row = (
            boot[boot["rule"] == rule_name].iloc[0]
            if not boot[boot["rule"] == rule_name].empty
            else pd.Series(dtype=float)
        )

        loco12 = loco_group["avg_12m"].dropna()
        loco24 = loco_group["avg_24m"].dropna()

        row = {
            "rule": rule_name,
            "description": rule["description"],
            "full_observations": full_row["observations"],
            "full_crises": full_row["crises"],
            "full_avg12": full_row["avg_12m"],
            "full_median12": full_row["median_12m"],
            "full_avg24": full_row["avg_24m"],
            "loco_min_avg12": loco12.min() if len(loco12) else np.nan,
            "loco_max_avg12": loco12.max() if len(loco12) else np.nan,
            "loco_positive_avg12_share_pct": (
                100 * (loco12 > 0).mean() if len(loco12) else np.nan
            ),
            "loco_min_avg24": loco24.min() if len(loco24) else np.nan,
            "bootstrap_independent_crises": boot_row.get("independent_crises", np.nan),
            "boot_avg12_p05": boot_row.get("boot_avg12_p05", np.nan),
            "boot_avg12_p50": boot_row.get("boot_avg12_p50", np.nan),
            "boot_avg12_p95": boot_row.get("boot_avg12_p95", np.nan),
            "boot_prob_avg12_positive_pct": boot_row.get(
                "boot_prob_avg12_positive_pct", np.nan
            ),
            "boot_prob_avg24_positive_pct": boot_row.get(
                "boot_prob_avg24_positive_pct", np.nan
            ),
        }

        row["robustness_grade"] = robustness_grade(
            rule_name,
            full_row,
            loco_group,
            boot_row,
            r1r2_cmp
        )

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    if not SAMPLES_FILE.exists():
        raise FileNotFoundError(
            f"{SAMPLES_FILE} not found. Run Taiwan Macro Investment Validation V3 first."
        )

    samples = pd.read_csv(SAMPLES_FILE, parse_dates=["date"])
    cycles = pd.read_csv(CYCLES_FILE, parse_dates=[
        "peak_date", "trigger_date", "trough_date", "recovery_date"
    ])

    # Tag every monthly sample to an INDEPENDENT crisis regime.
    samples["crisis_id"] = samples["date"].map(tag_crisis)

    # Keep full tagged data for auditability.
    samples.to_csv(
        OUTDIR / "v31_crisis_tagged_samples.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary_rows = []
    for rule_name, rule in RULES.items():
        row = aggregate_rule(samples, rule_name, rule)
        row["description"] = rule["description"]
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    loco = leave_one_crisis_out(samples)
    boot = bootstrap_all(samples)
    r1r2_cmp = compare_r1_r2(samples)
    final = final_robustness(summary, loco, boot, r1r2_cmp)

    # Add direct comparison metadata to R1/R2 rows.
    final["paired_-20_-25_crises"] = np.nan
    final["bottoming_recovering_advantage_12m"] = np.nan
    final["bottoming_recovering_better_share_pct"] = np.nan

    for rule_name in [
        "R1_-20_-25_Deteriorating",
        "R2_-20_-25_BottomingRecovering"
    ]:
        mask = final["rule"] == rule_name
        final.loc[mask, "paired_-20_-25_crises"] = r1r2_cmp["paired_crises"]
        final.loc[mask, "bottoming_recovering_advantage_12m"] = (
            r1r2_cmp["avg_advantage_bottoming_recovering_12m"]
        )
        final.loc[mask, "bottoming_recovering_better_share_pct"] = (
            r1r2_cmp["share_bottoming_recovering_better_pct"]
        )

    summary.to_csv(
        OUTDIR / "v31_rule_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    loco.to_csv(
        OUTDIR / "v31_leave_one_crisis_out.csv",
        index=False,
        encoding="utf-8-sig"
    )

    boot.to_csv(
        OUTDIR / "v31_bootstrap_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    final.to_csv(
        OUTDIR / "v31_final_robustness.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("===== V3.1 FINAL ROBUSTNESS =====")
    print(final.round(2).to_string(index=False))

    print("\n===== -20~-25 DIRECT MACRO COMPARISON =====")
    print(r1r2_cmp)

    print("\n===== INDEPENDENT CRISIS WINDOWS =====")
    for k, v in CRISIS_WINDOWS.items():
        print(k, v)

    print("\nV3.1 complete. No parameter optimization was performed.")


if __name__ == "__main__":
    main()
