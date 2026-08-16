from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    """台股投資決策正式版：四象限 + 核心指標拆解 + 輔助確認 + Drawdown 決策。"""
    st.subheader("台股投資決策｜正式版")
    st.caption("75% Macro / 25% Market｜景氣四象限｜Local Drawdown｜V3.1 Robustness Rules")

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
    except ImportError as exc:
        st.error(f"正式模型載入失敗：{exc}")
        st.info("若錯誤包含 openpyxl，請在 GitHub requirements.txt 新增：openpyxl>=3.1.2")
        return
    except Exception as exc:
        st.error(f"正式模型載入失敗：{exc}")
        return

    if market.empty or local.empty:
        st.error("TAIEX 資料不足。")
        return

    # ---------- Latest market + latest mature macro ----------
    mkt = market.iloc[-1]
    market_date = pd.Timestamp(mkt["date"])
    market_price = float(mkt["taiex_close"])
    current_dd = float(local.iloc[-1]["local_drawdown_pct"])

    macro_date = pd.Timestamp(sig["date"])
    regime = str(sig["macro_regime"])
    action, confidence, rationale = decide_action(current_dd, regime)

    labels = {
        "NORMAL_DCA": "正常定投",
        "OBSERVE": "觀察／保留彈藥",
        "SMALL_ADD": "小額加碼",
        "ADD": "加碼",
        "AGGRESSIVE_ADD": "積極加碼",
        "HEAVY_STAGED_ADD": "大幅分批加碼",
        "FORCED_STAGED_ADD": "深熊強制分批",
    }
    tranches = {
        "NORMAL_DCA": 0, "OBSERVE": 0, "SMALL_ADD": 5, "ADD": 10,
        "AGGRESSIVE_ADD": 15, "HEAVY_STAGED_ADD": 20, "FORCED_STAGED_ADD": 20,
    }
    regime_zh = {
        "EXPANDING": "擴張",
        "DETERIORATING": "惡化",
        "CONTRACTION": "衰退",
        "BOTTOMING": "築底",
        "RECOVERING": "復甦",
    }

    gap = max(
        0,
        (market_date.year - macro_date.year) * 12
        + market_date.month - macro_date.month
    )

    st.markdown("## 一眼看懂現在在哪裡")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最新市場月份", market_date.strftime("%Y-%m"))
    c2.metric("TAIEX", f"{market_price:,.0f}")
    c3.metric("Local Drawdown", f"{current_dd:.1f}%")
    c4.metric("Macro 資料截至", macro_date.strftime("%Y-%m"))

    c1, c2, c3 = st.columns(3)
    c1.metric("景氣階段", regime_zh.get(regime, regime))
    c2.metric("75/25 正式分數", f"{float(sig['official_score']):.2f}")
    c3.metric("市場領先 Macro", f"{gap} 個月")

    st.caption(
        "75/25 正式分數是標準化後的相對強弱，不是百分比、勝率或報酬率。"
        " 正值代表相對偏強、負值代表相對偏弱；真正的投資動作仍要搭配回撤。"
    )

    # ---------- Quadrant ----------
    st.markdown("## 景氣四象限")
    st.caption("X 軸＝景氣水準（75/25正式分數）；Y 軸＝3個月動能變化。軌跡顯示最近12個已成熟月份。")

    q = hist.dropna(subset=["official_score"]).copy()
    q["quadrant_x"] = q["official_score"]
    q["quadrant_y"] = q["official_score"].diff(3)
    q = q.dropna(subset=["quadrant_x", "quadrant_y"]).tail(12).copy()

    if len(q) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axhline(0, linewidth=1)
        ax.axvline(0, linewidth=1)

        ax.plot(q["quadrant_x"], q["quadrant_y"], marker="o", linewidth=1.5)

        # label first, recent few, and latest point
        for i, (_, r) in enumerate(q.iterrows()):
            if i == 0 or i >= len(q) - 4:
                ax.annotate(
                    pd.Timestamp(r["date"]).strftime("%y/%m"),
                    (r["quadrant_x"], r["quadrant_y"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        latest_q = q.iloc[-1]
        ax.scatter([latest_q["quadrant_x"]], [latest_q["quadrant_y"]], s=100)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.text(xmax * 0.60, ymax * 0.75, "擴張\nEXPANDING", ha="center", va="center")
        ax.text(xmax * 0.60, ymin * 0.75, "惡化\nDETERIORATING", ha="center", va="center")
        ax.text(xmin * 0.60, ymin * 0.75, "衰退\nCONTRACTION", ha="center", va="center")
        ax.text(xmin * 0.60, ymax * 0.75, "復甦／築底\nRECOVERING", ha="center", va="center")

        ax.set_xlabel("景氣水準 →")
        ax.set_ylabel("景氣動能 →")
        ax.set_title("最近 12 個月景氣四象限軌跡")
        ax.grid(alpha=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        latest_x = float(latest_q["quadrant_x"])
        latest_y = float(latest_q["quadrant_y"])
        if latest_x >= 0 and latest_y >= 0:
            quadrant_now = "擴張"
            quadrant_explain = "景氣水準偏強、動能仍上升。"
        elif latest_x >= 0 and latest_y < 0:
            quadrant_now = "惡化"
            quadrant_explain = "景氣水準仍偏強，但動能正在下降，屬高檔轉弱區。"
        elif latest_x < 0 and latest_y < 0:
            quadrant_now = "衰退"
            quadrant_explain = "景氣水準偏弱且動能仍下降，基本面壓力較大。"
        else:
            quadrant_now = "復甦／築底"
            quadrant_explain = "景氣水準仍偏弱，但動能已轉升，是逆週期投資最值得關注的區域。"

        st.info(f"目前四象限位置：**{quadrant_now}**。{quadrant_explain}")
    else:
        st.warning("目前成熟資料不足，尚無法畫出12個月四象限軌跡。")

    # ---------- Current decision ----------
    st.divider()
    c1, c2, c3 = st.columns([1.3, 1, 1])
    with c1:
        st.markdown("### 目前投資訊號")
        st.markdown(f"## {labels[action]}")
        st.write(rationale)
    with c2:
        st.markdown("### 信心")
        st.markdown(f"## {confidence}")
        st.caption("STRONG：V3.1較穩健；TENTATIVE / WEAK：歷史獨立危機樣本較少。")
    with c3:
        st.markdown("### 單次建議")
        st.markdown(f"## {tranches[action]}%")
        st.caption("占『剩餘熊市預備金』，不是總資產，也不是每月定期定額。")

    st.info("目前決策 = 最新 TAIEX Local Drawdown × 最後一個已成熟的 Macro Regime。")

    # ---------- Helpers ----------
    latest = hist.dropna(subset=["official_score"]).iloc[-1]

    def arrow(v):
        if pd.isna(v):
            return "—"
        return "↑" if float(v) >= 0 else "↓"

    def impact(v):
        if pd.isna(v):
            return "資料不足"
        v = float(v)
        if v >= 0.75:
            return "明顯正向"
        if v >= 0.20:
            return "偏正向"
        if v <= -0.75:
            return "明顯負向"
        if v <= -0.20:
            return "偏負向"
        return "中性"

    # ---------- Core Macro 75 ----------
    st.markdown("## 為什麼模型這樣判斷？")
    st.markdown("### ① Macro 層：75%｜景氣與企業基本面")

    macro_rows = [
        ["國發會景氣領先指標", "Macro領先 × 50%", arrow(latest.get("ndc_z3")), impact(latest.get("ndc_z3")), "最核心綜合領先訊號，判斷景氣方向是否轉折。"],
        ["外銷訂單總額", "Macro領先 × 20%", arrow(latest.get("orders_total_z3")), impact(latest.get("orders_total_z3")), "台灣出口需求溫度，通常早於實體生產。"],
        ["電子產品外銷訂單", "科技領先 × 10%", arrow(latest.get("orders_elec_z3")), impact(latest.get("orders_elec_z3")), "半導體／電子供應鏈需求。"],
        ["資訊通信外銷訂單", "科技領先 × 10%", arrow(latest.get("orders_ict_z3")), impact(latest.get("orders_ict_z3")), "伺服器、網通、AI與雲端需求。"],
        ["資訊電子生產指數", "Macro確認 × 10%", arrow(latest.get("mfg_elec_z3")), impact(latest.get("mfg_elec_z3")), "確認訂單是否真正轉成生產。"],
    ]
    st.dataframe(
        pd.DataFrame(macro_rows, columns=["指標", "類型×權重", "趨勢", "模型影響", "怎麼看"]),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Macro層內部權重：國發會50%＋外銷訂單20%＋電子10%＋資通10%＋資訊電子生產10%；"
        " 各指標先看1M/3M動能，再標準化後合成。"
    )

    # ---------- Market 25 ----------
    st.markdown("### ② Market 層：25%｜股市是否確認")

    market_rows = [
        ["TAIEX 3個月動能", "Market × 25%", arrow(latest.get("market_m3_z")), impact(latest.get("market_m3_z")), "市場短中期是否正在加速或轉弱。"],
        ["TAIEX 6個月動能", "Market × 25%", arrow(latest.get("market_m6_z")), impact(latest.get("market_m6_z")), "中期趨勢，降低單月價格雜訊。"],
        ["TAIEX vs 6月均線", "Market × 25%", arrow(latest.get("market_ma6_z")), impact(latest.get("market_ma6_z")), "是否站在中期趨勢之上。"],
        ["TAIEX vs 12月均線", "Market × 25%", arrow(latest.get("market_ma12_z")), impact(latest.get("market_ma12_z")), "長期市場確認，避免過度依賴Macro。"],
    ]
    st.dataframe(
        pd.DataFrame(market_rows, columns=["指標", "類型×權重", "趨勢", "模型影響", "怎麼看"]),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Market層只占總模型25%；用途是確認股市是否已開始反映基本面。"
        " V2/V3顯示提高到30~35%沒有增加重大熊市捕捉，因此正式版鎖定25%。"
    )

    # ---------- Secondary confirmation ----------
    st.markdown("### ③ 輔助確認｜不納入核心75/25權重")
    st.caption("這些指標主要用來『解釋與確認』，避免高度相關指標在核心模型內重複投票。")

    secondary = []

    # 景氣對策信號：NDC workbook provides it but official history may not expose it.
    ndc_score = latest.get("ndc_score", np.nan)
    if pd.notna(ndc_score):
        ndc_score = float(ndc_score)
        secondary.append([
            "景氣對策信號分數",
            f"{ndc_score:.0f}",
            "同步／確認",
            "不計權重",
            "用來確認目前景氣溫度；不直接加入核心模型，以避免與國發會領先指標重複。",
        ])
    else:
        secondary.append([
            "景氣對策信號",
            "資料未接入",
            "同步／確認",
            "不計權重",
            "適合做景氣溫度確認，但不影響正式75/25分數。",
        ])

    # PMI is not currently in the validated dataset.
    pmi_candidates = [c for c in raw.columns if "pmi" in c.lower()]
    if pmi_candidates:
        pmi_col = pmi_candidates[0]
        pmi_series = pd.to_numeric(raw[pmi_col], errors="coerce").dropna()
        pmi_val = pmi_series.iloc[-1] if len(pmi_series) else np.nan
        secondary.append([
            "製造業 PMI",
            f"{pmi_val:.1f}" if pd.notna(pmi_val) else "資料不足",
            "領先／擴散",
            "不計權重",
            "50以上代表擴張、50以下代表收縮；最適合輔助判斷四象限方向。",
        ])
    else:
        secondary.append([
            "製造業 PMI",
            "尚未接入",
            "領先／擴散",
            "不計權重",
            "值得保留作輔助確認；未經同一套回測前不加入核心權重。",
        ])

    # Customs exports if available.
    if "exports_yoy_pct" in raw.columns:
        ex = pd.to_numeric(raw["exports_yoy_pct"], errors="coerce").dropna()
        ex_val = ex.iloc[-1] if len(ex) else np.nan
        secondary.append([
            "出口年增率",
            f"{ex_val:.1f}%" if pd.notna(ex_val) else "資料不足",
            "同步／確認",
            "不計權重",
            "確認外銷訂單是否已反映到實際出口；因關務署資料偶爾逾時，列為輔助。",
        ])
    else:
        secondary.append([
            "出口年增率",
            "可選資料",
            "同步／確認",
            "不計權重",
            "確認外銷訂單是否落實到實際出口；不影響核心模型。",
        ])

    st.dataframe(
        pd.DataFrame(secondary, columns=["輔助指標", "最新值", "類型", "核心權重", "用途"]),
        hide_index=True,
        use_container_width=True,
    )

    # ---------- Drawdown decision matrix ----------
    st.markdown("### ④ 最後怎麼變成『買多少』？")
    decision = pd.DataFrame([
        ["0~-15%", "Macro影響低", "正常定投", "不動用熊市彈藥"],
        ["-15~-20%", "開始參考", "觀察／小額", "避免太早耗盡彈藥"],
        ["-20~-25%", "影響最高", "依惡化／築底／復甦差異化", "V3.1：Macro辨識價值 STRONG"],
        ["-25~-30%", "否決權下降", "至少開始分批", "V3.1：價格機會 STRONG"],
        ["-30~-35%", "價格主導", "積極分批", "樣本較少，避免All-in"],
        ["≤-35%", "Macro無否決權", "強制分批", "V3.1不支持一次All-in"],
    ], columns=["市場回撤", "Macro角色", "資金動作", "歷史驗證"])
    st.dataframe(decision, hide_index=True, use_container_width=True)

    with st.expander("如何理解 0.10、0.07 這種模型分數？"):
        st.write(
            "這些是標準化強弱分數，不是10%或7%。"
            " 例如 +1 代表目前動能約高於自身歷史平均1個標準差；-1代表低於平均1個標準差。"
            " 正式版應優先看『景氣四象限、指標方向、模型影響、回撤區間』，"
            " 不需要用單一小數做投資決策。"
        )
