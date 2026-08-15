from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from .backtest import (
    drawdown_staged_exposure,
    performance_metrics,
    run_exposure_backtest,
)
from .charts import comparison_chart, cycle_chart, equity_chart
from .config import MODEL_SPECS, ORIGINAL_MODEL_SPECS
from .data import (
    build_market_features,
    derive_proxy_indicators,
    fetch_public_macro,
    fetch_yahoo_monthly,
    load_uploaded_csv,
    merge_all_data,
)
from .model import run_model
from .taiwan_decision_engine import build_official_history, latest_signal, local_cycle_engine, decide_action


def render_sidebar(defaults: dict) -> dict:
    st.sidebar.header("模型設定")
    start_date = st.sidebar.date_input(
        "起始日期",
        value=pd.Timestamp(defaults["start_date"]).date(),
        min_value=date(1990, 1, 1),
    )
    end_date = st.sidebar.date_input("結束日期", value=date.today())
    momentum_months = st.sidebar.slider(
        "景氣動能（月）", min_value=1, max_value=12,
        value=int(defaults["momentum_months"])
    )
    use_pmi_extension = st.sidebar.toggle(
        "台股 2012 年後加入 PMI 擴充版",
        value=bool(defaults["use_pmi_extension"])
    )
    transaction_cost_bps = st.sidebar.number_input(
        "單向交易成本（bps）",
        min_value=0.0, max_value=100.0,
        value=float(defaults["transaction_cost_bps"]),
        step=1.0
    )
    risk_free_rate = st.sidebar.number_input(
        "現金／短債年化報酬",
        min_value=0.0, max_value=0.20,
        value=float(defaults["risk_free_rate"]),
        step=0.005,
        format="%.3f"
    )
    initial_capital = st.sidebar.number_input(
        "回測初始資金（新台幣）",
        min_value=10_000,
        value=int(defaults["initial_capital"]),
        step=100_000
    )
    uploaded_file = st.sidebar.file_uploader(
        "上傳台灣或自訂月資料 CSV",
        type=["csv"],
        help="日期欄可用 date、month、年月或日期；其餘欄名須對應指標代碼。"
    )

    return {
        **defaults,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "momentum_months": momentum_months,
        "use_pmi_extension": use_pmi_extension,
        "transaction_cost_bps": transaction_cost_bps,
        "risk_free_rate": risk_free_rate,
        "initial_capital": initial_capital,
        "uploaded_file": uploaded_file,
    }


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def _base_data(start_date: str, end_date: str, ticker: str):
    macro, status = fetch_public_macro(start_date, end_date)
    price = fetch_yahoo_monthly(ticker, start_date, end_date)
    features = build_market_features(price)
    macro = derive_proxy_indicators(macro)
    return macro, features, status


def _load_market_data(market: str, settings: dict):
    spec = MODEL_SPECS[market]
    uploaded = None
    if settings.get("uploaded_file") is not None:
        uploaded = load_uploaded_csv(settings["uploaded_file"])

    macro, features, status = _base_data(
        settings["start_date"],
        settings["end_date"],
        spec["benchmark"],
    )
    data = merge_all_data(macro, features, uploaded)
    return data, status


def _weights_for_market(market: str, settings: dict):
    spec = MODEL_SPECS[market]
    weights = dict(spec["weights"])
    directions = dict(spec["directions"])

    if market == "TW" and settings["use_pmi_extension"]:
        pmi = spec["pmi_extension"]
        # 加入 PMI 後，將原有權重按比例壓縮，使總和仍為 1。
        scale = 1 - pmi["weight"]
        weights = {k: v * scale for k, v in weights.items()}
        weights[pmi["indicator"]] = pmi["weight"]
        directions[pmi["indicator"]] = pmi["direction"]

    return weights, directions


def _run_market(market: str, settings: dict):
    data, status = _load_market_data(market, settings)
    weights, directions = _weights_for_market(market, settings)
    revised = run_model(
        data,
        weights,
        directions,
        settings["momentum_months"],
    )
    original_spec = ORIGINAL_MODEL_SPECS[market]
    original = run_model(
        data,
        original_spec["weights"],
        MODEL_SPECS[market]["directions"],
        settings["momentum_months"],
    )
    return data, status, revised, original


def render_overview(settings: dict):
    st.subheader("模型設計總覽")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 美股")
        st.info(
            "75% 經濟及金融領先指標＋25% 市場確認；"
            "股價均線訊號占總分 7%。"
        )
        st.markdown(
            "- 經濟／金融：殖利率曲線、信用利差、初領失業救濟、"
            "新訂單、住宅、流動性與 LEI 代理。\n"
            "- 市場確認：價格趨勢、breadth 代理、6 月動能與波動環境。"
        )

    with c2:
        st.markdown("### 台股")
        st.info(
            "75% 出口、半導體及全球領先＋25% 市場確認；"
            "股價均線訊號占總分 7%。"
        )
        st.markdown(
            "- 2000 年主回測不強制使用 PMI。\n"
            "- 2012 年後可加入 PMI 擴充版。\n"
            "- 台灣官方指標可透過 CSV 上傳。"
        )

    st.markdown("### 景氣階段判定")
    st.dataframe(pd.DataFrame([
        ["擴張加速", "分數 ≥ 0.35", "3 月動能 ≥ 0"],
        ["擴張減速", "分數 ≥ 0.35", "3 月動能 < 0"],
        ["復甦", "-0.35 < 分數 < 0.35", "3 月動能 ≥ 0"],
        ["放緩", "-0.35 < 分數 < 0.35", "3 月動能 < 0"],
        ["谷底改善", "分數 ≤ -0.35", "3 月動能 ≥ 0"],
        ["衰退惡化", "分數 ≤ -0.35", "3 月動能 < 0"],
    ], columns=["階段", "分數條件", "動能條件"]), hide_index=True, use_container_width=True)

    st.warning(
        "台灣官方月資料沒有穩定、統一且免驗證的公開 API。"
        "程式不會虛構缺少的台灣出口或 PMI 數據；請以上傳 CSV 補齊，"
        "之後模型會自動納入並重新正規化權重。"
    )


def render_market_page(market: str, settings: dict):
    spec = MODEL_SPECS[market]
    st.subheader(spec["name"])

    try:
        data, status, revised, original = _run_market(market, settings)
    except Exception as exc:
        st.error(f"資料載入失敗：{exc}")
        return

    frame = revised.frame.dropna(subset=["SCORE"])
    if frame.empty:
        st.error("目前可用指標不足，無法計算模型。請檢查資料頁或上傳 CSV。")
        return

    latest = frame.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最新分數", f"{latest['SCORE']:.2f}")
    c2.metric("3 月動能", f"{latest['MOMENTUM_3M']:.2f}")
    c3.metric("景氣階段", str(latest["PHASE"]))
    c4.metric("建議風險曝險", f"{latest['EXPOSURE']:.0%}")

    st.plotly_chart(cycle_chart(frame, f"{spec['market_label']} 與景氣循環"), use_container_width=True)
    st.plotly_chart(
        comparison_chart(revised.frame["SCORE"], original.frame["SCORE"]),
        use_container_width=True
    )

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("### 實際使用權重")
        weight_df = pd.DataFrame({
            "指標": revised.used_weights.keys(),
            "正規化權重": revised.used_weights.values(),
        }).sort_values("正規化權重", ascending=False)
        st.dataframe(
            weight_df.style.format({"正規化權重": "{:.1%}"}),
            hide_index=True,
            use_container_width=True
        )
    with right:
        st.markdown("### 缺少指標")
        if revised.missing_indicators:
            st.warning("、".join(revised.missing_indicators))
        else:
            st.success("所有設定指標均有足夠資料。")

    output_cols = [
        c for c in ["PRICE", "SCORE", "MOMENTUM_3M", "PHASE", "EXPOSURE"]
        if c in revised.frame.columns
    ]
    csv = revised.frame[output_cols].to_csv(index=True).encode("utf-8-sig")
    st.download_button(
        "下載模型月度結果 CSV",
        data=csv,
        file_name=f"macro_cycle_{market.lower()}_result.csv",
        mime="text/csv",
    )


def render_backtest_page(settings: dict):
    st.subheader("策略回測")
    market = st.radio("市場", ["US", "TW"], horizontal=True)
    spec = MODEL_SPECS[market]

    try:
        _, _, revised, original = _run_market(market, settings)
    except Exception as exc:
        st.error(f"資料載入失敗：{exc}")
        return

    frame = revised.frame
    if "PRICE" not in frame or frame["SCORE"].dropna().empty:
        st.error("回測所需資料不足。")
        return

    price = frame["PRICE"]
    revised_exposure = frame["EXPOSURE"]
    original_exposure = original.frame["EXPOSURE"]

    staged = drawdown_staged_exposure(price, revised_exposure, reserve_ratio=0.15)
    buy_hold_exp = pd.Series(1.0, index=price.index)

    bt_revised = run_exposure_backtest(
        price, revised_exposure,
        settings["initial_capital"],
        settings["transaction_cost_bps"],
        settings["risk_free_rate"],
    )
    bt_original = run_exposure_backtest(
        price, original_exposure,
        settings["initial_capital"],
        settings["transaction_cost_bps"],
        settings["risk_free_rate"],
    )
    bt_staged = run_exposure_backtest(
        price, staged,
        settings["initial_capital"],
        settings["transaction_cost_bps"],
        settings["risk_free_rate"],
    )
    bt_hold = run_exposure_backtest(
        price, buy_hold_exp,
        settings["initial_capital"],
        0,
        settings["risk_free_rate"],
    )

    chart_data = {
        "修正版動態配置": bt_revised,
        "原模型動態配置": bt_original,
        "85%核心＋15%回撤彈藥": bt_staged,
        "Buy & Hold": bt_hold,
    }
    st.plotly_chart(equity_chart(chart_data), use_container_width=True)

    rows = []
    for name, bt in chart_data.items():
        ret_col = "STRATEGY_RETURN"
        eq_col = "STRATEGY_EQUITY"
        metrics = performance_metrics(
            bt[eq_col], bt[ret_col], settings["risk_free_rate"]
        )
        rows.append({"策略": name, **metrics})

    metrics_df = pd.DataFrame(rows).set_index("策略")
    styled = metrics_df.style.format({
        "總報酬": "{:.1%}",
        "CAGR": "{:.1%}",
        "年化波動": "{:.1%}",
        "Sharpe": "{:.2f}",
        "最大回撤": "{:.1%}",
    })
    st.dataframe(styled, use_container_width=True)

    st.markdown("### 15% 保留籌碼規則")
    st.write(
        "平時維持 85% 核心曝險；指數自高點下跌 20%～35% 時，"
        "最後 15% 依景氣模型曝險比例釋放；跌幅達 35% 以下時，"
        "不等待模型，直接釋放完整 15%。"
    )


def render_data_page(settings: dict):
    st.subheader("資料與診斷")
    market = st.radio("檢查市場", ["US", "TW"], horizontal=True)

    try:
        data, status, revised, _ = _run_market(market, settings)
    except Exception as exc:
        st.error(f"資料載入失敗：{exc}")
        return

    status_df = pd.DataFrame(
        [{"指標": k, "抓取狀態": v} for k, v in status.items()]
    )
    st.dataframe(status_df, hide_index=True, use_container_width=True)

    st.markdown("### 最新資料日期與有效筆數")
    diag = []
    for col in data.columns:
        s = data[col].dropna()
        diag.append({
            "欄位": col,
            "有效筆數": len(s),
            "起始": s.index.min().date() if len(s) else None,
            "最新": s.index.max().date() if len(s) else None,
        })
    st.dataframe(pd.DataFrame(diag), hide_index=True, use_container_width=True)

    st.markdown("### CSV 欄名")
    st.code(
        "TW_EXPORTS, TW_EXPORT_ORDERS, TW_SEMI_EXPORTS,\n"
        "TW_INDUSTRIAL_PRODUCTION, TW_MONEY_M1B, TW_PMI,\n"
        "US_LEI_PROXY, GLOBAL_SEMI"
    )


def render_taiwan_official_page(settings: dict):
    st.subheader("台股投資決策｜正式版")
    st.caption("75% Macro / 25% Market｜Local Drawdown｜V3.1 Robustness Rules")

    macro_path = Path("data/processed/taiwan_macro_inputs.csv")
    ndc_path = Path("data/raw/ndc_business_cycle.xlsx")
    if not macro_path.exists() or not ndc_path.exists():
        st.error("正式版資料尚未就緒：請先執行 Macro Cycle Lab Official Refresh。")
        return

    try:
        hist = build_official_history(macro_path, ndc_path)
        sig = latest_signal(hist)
        raw = pd.read_csv(macro_path, parse_dates=["date"]).sort_values("date")
        market = raw.dropna(subset=["taiex_close"]).copy()
        local = local_cycle_engine(market[["date", "taiex_close"]])
    except Exception as exc:
        st.error(f"正式模型載入失敗：{exc}")
        return

    if market.empty or local.empty:
        st.error("TAIEX 資料不足。")
        return

    mkt = market.iloc[-1]
    market_date = pd.Timestamp(mkt["date"])
    market_price = float(mkt["taiex_close"])
    current_dd = float(local.iloc[-1]["local_drawdown_pct"])
    macro_date = pd.Timestamp(sig["date"])
    regime = str(sig["macro_regime"])
    action, confidence, rationale = decide_action(current_dd, regime)

    labels = {
        "NORMAL_DCA":"正常定投","OBSERVE":"觀察／保留彈藥","SMALL_ADD":"小額加碼",
        "ADD":"加碼","AGGRESSIVE_ADD":"積極加碼","HEAVY_STAGED_ADD":"大幅分批加碼",
        "FORCED_STAGED_ADD":"深熊強制分批",
    }
    tranches = {
        "NORMAL_DCA":0,"OBSERVE":0,"SMALL_ADD":5,"ADD":10,
        "AGGRESSIVE_ADD":15,"HEAVY_STAGED_ADD":20,"FORCED_STAGED_ADD":20,
    }
    gap=max(0,(market_date.year-macro_date.year)*12+market_date.month-macro_date.month)

    c1,c2,c3,c4=st.columns(4)
    c1.metric("最新市場月份",market_date.strftime("%Y-%m"))
    c2.metric("TAIEX",f"{market_price:,.0f}")
    c3.metric("Local Drawdown",f"{current_dd:.1f}%")
    c4.metric("Macro 資料截至",macro_date.strftime("%Y-%m"))

    c1,c2,c3=st.columns(3)
    c1.metric("Macro Regime",regime)
    c2.metric("75/25 正式分數",f"{float(sig['official_score']):.2f}")
    c3.metric("市場領先 Macro",f"{gap} 個月")

    st.divider()
    c1,c2,c3=st.columns([1.3,1,1])
    with c1:
        st.markdown("### 目前投資訊號")
        st.markdown(f"## {labels[action]}")
        st.write(rationale)
    with c2:
        st.markdown("### 信心")
        st.markdown(f"## {confidence}")
        st.caption("STRONG 為 V3.1 較穩健；TENTATIVE / WEAK 代表樣本較少。")
    with c3:
        st.markdown("### 單次建議")
        st.markdown(f"## {tranches[action]}%")
        st.caption("占『剩餘熊市預備金』，不是總資產。")

    st.info("決策使用最新 TAIEX 回撤＋最後一個已完整公布並套用1個月發布延遲的 Macro regime。")

    st.markdown("### 正式決策矩陣")
    matrix=pd.DataFrame([
        ["0~-15%","正常定投","正常定投","正常定投","不動用熊市彈藥"],
        ["-15~-20%","觀察","小額加碼","小額加碼","觀察區"],
        ["-20~-25%","觀察／保守","加碼","積極加碼","Macro辨識價值 STRONG"],
        ["-25~-30%","開始加碼","積極加碼","積極加碼","價格機會 STRONG"],
        ["-30~-35%","積極分批","大幅分批","大幅分批","避免一次All-in"],
        ["≤-35%","深熊強制分批","強力分批","強力分批","Macro無否決權；不一次All-in"],
    ],columns=["TAIEX回撤","Macro惡化","Macro築底","Macro復甦","說明"])
    st.dataframe(matrix,hide_index=True,use_container_width=True)

    st.markdown("### 最近 24 個已成熟 Macro 月份")
    cols=["date","taiex_close","local_drawdown_pct","macro_regime","official_score",
          "action_label_zh","confidence","suggested_tranche_pct_of_remaining_reserve"]
    recent=hist[cols].dropna(subset=["official_score"]).tail(24).copy()
    recent["date"]=recent["date"].dt.strftime("%Y-%m")
    st.dataframe(recent,hide_index=True,use_container_width=True)
