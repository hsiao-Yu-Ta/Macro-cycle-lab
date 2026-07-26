from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(equity: pd.Series, monthly_returns: pd.Series, rf: float = 0.02) -> dict:
    equity = equity.dropna()
    monthly_returns = monthly_returns.dropna()

    if len(equity) < 2:
        return {
            "總報酬": np.nan,
            "CAGR": np.nan,
            "年化波動": np.nan,
            "Sharpe": np.nan,
            "最大回撤": np.nan,
        }

    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 12)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    vol = monthly_returns.std() * np.sqrt(12)
    excess = monthly_returns.mean() * 12 - rf
    sharpe = excess / vol if vol and not np.isnan(vol) else np.nan
    drawdown = equity / equity.cummax() - 1

    return {
        "總報酬": total_return,
        "CAGR": cagr,
        "年化波動": vol,
        "Sharpe": sharpe,
        "最大回撤": drawdown.min(),
    }


def run_exposure_backtest(
    price: pd.Series,
    exposure: pd.Series,
    initial_capital: float = 1_000_000,
    transaction_cost_bps: float = 5.0,
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    df = pd.concat([price.rename("PRICE"), exposure.rename("TARGET_EXPOSURE")], axis=1).dropna()
    df["ASSET_RETURN"] = df["PRICE"].pct_change().fillna(0)
    df["EXPOSURE"] = df["TARGET_EXPOSURE"].shift(1).fillna(0)
    turnover = df["EXPOSURE"].diff().abs().fillna(df["EXPOSURE"].abs())
    cost = turnover * transaction_cost_bps / 10_000
    cash_return = (1 - df["EXPOSURE"]) * ((1 + risk_free_rate) ** (1 / 12) - 1)
    df["STRATEGY_RETURN"] = df["EXPOSURE"] * df["ASSET_RETURN"] + cash_return - cost
    df["STRATEGY_EQUITY"] = initial_capital * (1 + df["STRATEGY_RETURN"]).cumprod()
    df["BUY_HOLD_RETURN"] = df["ASSET_RETURN"]
    df["BUY_HOLD_EQUITY"] = initial_capital * (1 + df["BUY_HOLD_RETURN"]).cumprod()
    df["DRAWDOWN"] = df["STRATEGY_EQUITY"] / df["STRATEGY_EQUITY"].cummax() - 1
    df["ASSET_DRAWDOWN"] = df["PRICE"] / df["PRICE"].cummax() - 1
    return df


def drawdown_staged_exposure(
    price: pd.Series,
    macro_exposure: pd.Series,
    reserve_ratio: float = 0.15,
) -> pd.Series:
    """
    依使用者既有加碼框架：
    - 一般修正 -20%～-30%：由景氣模型決定是否投入最後資金。
    - 深熊 -35% 以下：直接依價格啟用保留資金。
    """
    price = price.dropna()
    dd = price / price.cummax() - 1
    macro = macro_exposure.reindex(price.index).ffill().fillna(0.5)

    base = pd.Series(1 - reserve_ratio, index=price.index, dtype=float)
    release = pd.Series(0.0, index=price.index)

    normal = (dd <= -0.20) & (dd > -0.35)
    release.loc[normal] = reserve_ratio * macro.loc[normal]

    deep = dd <= -0.35
    release.loc[deep] = reserve_ratio

    return (base + release).clip(0, 1)
