from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class ModelResult:
    frame: pd.DataFrame
    used_weights: Dict[str, float]
    missing_indicators: list[str]


def expanding_zscore(series: pd.Series, min_periods: int = 24, clip: float = 2.5) -> pd.Series:
    mean = series.expanding(min_periods=min_periods).mean()
    std = series.expanding(min_periods=min_periods).std()
    z = (series - mean) / std.replace(0, np.nan)
    return z.clip(-clip, clip)


def transform_indicator(name: str, s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)

    level_indicators = {
        "US_YIELD_CURVE",
        "US_CREDIT_SPREAD",
        "VIX",
        "VOLATILITY_REGIME",
        "PRICE_TREND",
        "MARKET_BREADTH",
        "MARKET_MOMENTUM",
        "US_LEI_PROXY",
    }

    if name in level_indicators:
        return s

    # 大多數總經序列以年增率評估，降低長期趨勢與量綱影響。
    return s.pct_change(12)


def normalize_weights(weights: Dict[str, float], available: list[str]) -> Dict[str, float]:
    selected = {k: v for k, v in weights.items() if k in available and v > 0}
    total = sum(selected.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in selected.items()}


def classify_cycle(score: pd.Series, momentum: pd.Series) -> pd.Series:
    conditions = [
        (score >= 0.35) & (momentum >= 0),
        (score >= 0.35) & (momentum < 0),
        (score > -0.35) & (score < 0.35) & (momentum >= 0),
        (score > -0.35) & (score < 0.35) & (momentum < 0),
        (score <= -0.35) & (momentum >= 0),
        (score <= -0.35) & (momentum < 0),
    ]
    labels = [
        "擴張加速",
        "擴張減速",
        "復甦",
        "放緩",
        "谷底改善",
        "衰退惡化",
    ]
    return pd.Series(np.select(conditions, labels, default="資料不足"), index=score.index)


def score_to_exposure(score: pd.Series, phase: pd.Series) -> pd.Series:
    base = ((score + 1.25) / 2.5).clip(0, 1)
    adjustment = phase.map({
        "擴張加速": 0.15,
        "擴張減速": -0.05,
        "復甦": 0.10,
        "放緩": -0.10,
        "谷底改善": 0.05,
        "衰退惡化": -0.20,
        "資料不足": 0.0,
    }).fillna(0.0)
    return (base + adjustment).clip(0, 1)


def run_model(
    data: pd.DataFrame,
    weights: Dict[str, float],
    directions: Optional[Dict[str, int]] = None,
    momentum_months: int = 3,
) -> ModelResult:
    directions = directions or {}
    available = [k for k in weights if k in data.columns and data[k].notna().sum() >= 24]
    missing = [k for k in weights if k not in available]
    used_weights = normalize_weights(weights, available)

    scored = {}
    for name, weight in used_weights.items():
        transformed = transform_indicator(name, data[name])
        z = expanding_zscore(transformed)
        direction = directions.get(name, 1)
        scored[name] = z * direction

    if not scored:
        empty = pd.DataFrame(index=data.index)
        empty["SCORE"] = np.nan
        empty["MOMENTUM_3M"] = np.nan
        empty["PHASE"] = "資料不足"
        empty["EXPOSURE"] = np.nan
        return ModelResult(empty, {}, missing)

    component_scores = pd.DataFrame(scored)
    weighted = component_scores.mul(pd.Series(used_weights), axis=1)

    # 當月缺值時，以當月實際可用權重重新縮放。
    valid_weight = component_scores.notna().mul(pd.Series(used_weights), axis=1).sum(axis=1)
    score = weighted.sum(axis=1).div(valid_weight.replace(0, np.nan))
    score = (score / 1.5).clip(-1.25, 1.25)

    momentum = score - score.shift(momentum_months)
    phase = classify_cycle(score, momentum)
    exposure = score_to_exposure(score, phase)

    result = data.copy()
    for col in component_scores.columns:
        result[f"Z_{col}"] = component_scores[col]
        result[f"C_{col}"] = weighted[col]
    result["SCORE"] = score
    result["MOMENTUM_3M"] = momentum
    result["PHASE"] = phase
    result["EXPOSURE"] = exposure

    return ModelResult(result, used_weights, missing)
