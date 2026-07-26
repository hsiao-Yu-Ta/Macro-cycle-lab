from __future__ import annotations

APP_TITLE = "Macro Cycle Lab"

DEFAULT_SETTINGS = {
    "start_date": "2000-01-01",
    "end_date": None,
    "score_window": 60,
    "momentum_months": 3,
    "use_pmi_extension": False,
    "pmi_start": "2012-01-01",
    "transaction_cost_bps": 5.0,
    "risk_free_rate": 0.02,
    "rebalance_frequency": "M",
    "initial_capital": 1_000_000,
}

# 修正版：市場確認合計 25%，其中股價均線 7%。
MODEL_SPECS = {
    "US": {
        "name": "美股修正版",
        "benchmark": "QQQ",
        "market_label": "Nasdaq-100 / QQQ",
        "weights": {
            "US_LEI_PROXY": 0.15,
            "US_YIELD_CURVE": 0.12,
            "US_CREDIT_SPREAD": 0.10,
            "US_INITIAL_CLAIMS": 0.10,
            "US_NEW_ORDERS": 0.10,
            "US_HOUSING": 0.08,
            "US_LIQUIDITY": 0.10,
            "PRICE_TREND": 0.07,
            "MARKET_BREADTH": 0.06,
            "MARKET_MOMENTUM": 0.06,
            "VOLATILITY_REGIME": 0.06,
        },
        "directions": {
            "US_LEI_PROXY": 1,
            "US_YIELD_CURVE": 1,
            "US_CREDIT_SPREAD": -1,
            "US_INITIAL_CLAIMS": -1,
            "US_NEW_ORDERS": 1,
            "US_HOUSING": 1,
            "US_LIQUIDITY": 1,
            "PRICE_TREND": 1,
            "MARKET_BREADTH": 1,
            "MARKET_MOMENTUM": 1,
            "VOLATILITY_REGIME": -1,
        },
    },
    "TW": {
        "name": "台股修正版",
        "benchmark": "^TWII",
        "market_label": "台灣加權指數",
        "weights": {
            "TW_EXPORTS": 0.14,
            "TW_EXPORT_ORDERS": 0.12,
            "TW_SEMI_EXPORTS": 0.12,
            "TW_INDUSTRIAL_PRODUCTION": 0.10,
            "TW_MONEY_M1B": 0.08,
            "GLOBAL_SEMI": 0.10,
            "US_LEI_PROXY": 0.09,
            "PRICE_TREND": 0.07,
            "MARKET_BREADTH": 0.06,
            "MARKET_MOMENTUM": 0.06,
            "VOLATILITY_REGIME": 0.06,
        },
        "directions": {
            "TW_EXPORTS": 1,
            "TW_EXPORT_ORDERS": 1,
            "TW_SEMI_EXPORTS": 1,
            "TW_INDUSTRIAL_PRODUCTION": 1,
            "TW_MONEY_M1B": 1,
            "GLOBAL_SEMI": 1,
            "US_LEI_PROXY": 1,
            "PRICE_TREND": 1,
            "MARKET_BREADTH": 1,
            "MARKET_MOMENTUM": 1,
            "VOLATILITY_REGIME": -1,
        },
        "pmi_extension": {
            "indicator": "TW_PMI",
            "weight": 0.08,
            "direction": 1,
            "start": "2012-01-01",
        },
    },
}

# 原模型供平行比較。設定較偏總經分數、較少市場確認。
ORIGINAL_MODEL_SPECS = {
    "US": {
        "name": "美股原模型",
        "benchmark": "QQQ",
        "weights": {
            "US_LEI_PROXY": 0.22,
            "US_YIELD_CURVE": 0.16,
            "US_CREDIT_SPREAD": 0.12,
            "US_INITIAL_CLAIMS": 0.12,
            "US_NEW_ORDERS": 0.12,
            "US_HOUSING": 0.10,
            "US_LIQUIDITY": 0.10,
            "PRICE_TREND": 0.06,
        },
    },
    "TW": {
        "name": "台股原模型",
        "benchmark": "^TWII",
        "weights": {
            "TW_EXPORTS": 0.18,
            "TW_EXPORT_ORDERS": 0.16,
            "TW_SEMI_EXPORTS": 0.15,
            "TW_INDUSTRIAL_PRODUCTION": 0.13,
            "TW_MONEY_M1B": 0.10,
            "GLOBAL_SEMI": 0.10,
            "US_LEI_PROXY": 0.10,
            "PRICE_TREND": 0.08,
        },
    },
}

FRED_SERIES = {
    "US_YIELD_CURVE": "T10Y2Y",
    "US_CREDIT_SPREAD": "BAMLH0A0HYM2",
    "US_INITIAL_CLAIMS": "ICSA",
    "US_NEW_ORDERS": "DGORDER",
    "US_HOUSING": "HOUST",
    "US_LIQUIDITY": "M2SL",
    "US_INDUSTRIAL_PRODUCTION": "INDPRO",
    "US_UNEMPLOYMENT": "UNRATE",
    "VIX": "VIXCLS",
    "GLOBAL_SEMI": "PCU334413334413",
}

YAHOO_TICKERS = {
    "QQQ": "QQQ",
    "^TWII": "^TWII",
    "SPY": "SPY",
    "SOXX": "SOXX",
    "VIX": "^VIX",
}
