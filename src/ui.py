from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .backtest import (
    drawdown_staged_exposure,
    performance_metrics,
    run_exposure_backtest,
)
from .charts import equity_chart
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
from .taiwan_decision_engine import (
    build_official_history,
    latest_signal,
    local_cycle_engine,
    decide_action,
)


STATUS_LABELS = {
    "positive": ("🟢", "正向"),
    "neutral": ("🟡", "中性"),
    "negative": ("🔴", "負向"),
    "low": ("🟢", "很低"),
    "medium": ("🟡", "中等"),
    "high": ("🟠", "大幅"),
    "severe": ("🔴", "深熊"),
}

PHASE_ZH = {
    "EXPANDING": "擴張",
    "DETERIORATING": "惡化",
    "CONTRACTION": "衰退",
    "BOTTOMING": "築底",
    "RECOVERING": "復甦",
    "擴張加速": "擴張",
    "擴張減速": "惡化",
    "復甦": "復甦",
    "放緩": "惡化",
    "谷底改善": "築底",
    "衰退惡化": "衰退",
}

ACTION_LABELS = {
    "NORMAL_DCA": "正常定投",
    "OBSERVE": "觀察／保留彈藥",
    "SMALL_ADD": "小額加碼",
    "ADD": "加碼",
    "AGGRESSIVE_ADD": "積極加碼",
    "HEAVY_STAGED_ADD": "大幅分批加碼",
    "FORCED_STAGED_ADD": "深熊強制分批",
}

TRANCHE_MAP = {
    "NORMAL_DCA": 0,
    "OBSERVE": 0,
    "SMALL_ADD": 5,
    "ADD": 10,
    "AGGRESSIVE_ADD": 15,
    "HEAVY_STAGED_ADD": 20,
    "FORCED_STAGED_ADD": 20,
}


def _inject_css():
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Noto Sans TC",
                         "PingFang TC", "Microsoft JhengHei", sans-serif;
        }
        .block-container {padding-top: 1.2rem; padding-bottom: 3rem;}
        .mcl-card {
            border: 1px solid rgba(148,163,184,.22);
            background: rgba(15,23,42,.45);
            border-radius: 16px;
            padding: 16px 18px;
            margin-bottom: 10px;
        }
        .mcl-card h3 {margin: 0 0 6px 0;}
        .mcl-big {
            font-size: 2rem;
            font-weight: 800;
            margin: 4px 0;
        }
        .mcl-muted {color: #94a3b8; font-size: .92rem;}
        .mcl-good {color:#65d46e; font-weight:700;}
        .mcl-warn {color:#f4bf4f; font-weight:700;}
        .mcl-bad {color:#ef6a6a; font-weight:700;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _status_from_value(v: float, pos=0.10, neg=-0.10):
    if pd.isna(v):
        return "⚪", "資料不足"
    v = float(v)
    if v >= pos:
        return "🟢", "正向"
    if v <= neg:
        return "🔴", "負向"
    return "🟡", "中性"


def _drawdown_status(dd: float):
    if pd.isna(dd):
        return "⚪", "資料不足"
    if dd > -15:
        return "🟢", "很低"
    if dd > -25:
        return "🟡", "中等"
    if dd > -35:
        return "🟠", "大幅"
    return "🔴", "深熊"


def _phase_from_xy(x: float, y: float):
    if x >= 0 and y >= 0:
        return "擴張", "景氣水準偏強、動能仍向上"
    if x >= 0 and y < 0:
        return "惡化", "景氣水準仍偏強，但動能轉弱"
    if x < 0 and y < 0:
        return "衰退", "景氣水準偏弱、動能仍下滑"
    return "復甦／築底", "景氣仍偏弱，但動能已開始回升"


def _quadrant_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    q = df.dropna(subset=[x_col, y_col]).tail(12).copy()
    fig = go.Figure()

    lim = 1.2
    if len(q):
        lim = max(
            1.0,
            float(np.nanmax(np.abs(q[[x_col, y_col]].to_numpy()))) * 1.35
        )

    fig.add_shape(type="rect", x0=0, x1=lim, y0=0, y1=lim,
                  fillcolor="rgba(34,197,94,.12)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=-lim, x1=0, y0=0, y1=lim,
                  fillcolor="rgba(59,130,246,.12)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=-lim, x1=0, y0=-lim, y1=0,
                  fillcolor="rgba(239,68,68,.10)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0, x1=lim, y0=-lim, y1=0,
                  fillcolor="rgba(245,158,11,.12)", line_width=0, layer="below")

    if len(q):
        fig.add_trace(go.Scatter(
            x=q[x_col], y=q[y_col],
            mode="lines+markers+text",
            text=[pd.Timestamp(d).strftime("%y/%m") for d in q["date"]],
            textposition="top center",
            line=dict(width=3),
            marker=dict(size=[8]*(max(0, len(q)-1))+[14]),
            hovertemplate="月份 %{text}<br>景氣水準 %{x:.2f}<br>3M動能 %{y:.2f}<extra></extra>",
        ))

    fig.add_annotation(x=lim*.58, y=lim*.72, text="<b>擴張</b><br>水準強、動能升",
                       showarrow=False, font=dict(size=16))
    fig.add_annotation(x=-lim*.58, y=lim*.72, text="<b>復甦／築底</b><br>水準弱、動能升",
                       showarrow=False, font=dict(size=15))
    fig.add_annotation(x=-lim*.58, y=-lim*.72, text="<b>衰退</b><br>水準弱、動能降",
                       showarrow=False, font=dict(size=16))
    fig.add_annotation(x=lim*.58, y=-lim*.72, text="<b>惡化</b><br>水準強、動能降",
                       showarrow=False, font=dict(size=16))

    fig.update_xaxes(range=[-lim, lim], zeroline=True, zerolinewidth=1,
                     title="景氣水準（弱 ← → 強）")
    fig.update_yaxes(range=[-lim, lim], zeroline=True, zerolinewidth=1,
                     title="3個月景氣動能（弱 ← → 強）")
    fig.update_layout(
        title=title,
        height=500,
        margin=dict(l=20, r=20, t=55, b=20),
        showlegend=False,
        hovermode="closest",
    )
    return fig


def _gauge_drawdown(dd: float, label: str):
    dd = float(dd) if pd.notna(dd) else 0.0
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=abs(dd),
        number={"suffix": "%", "valueformat": ".1f"},
        title={"text": label},
        gauge={
            "axis": {"range": [0, 45]},
            "bar": {"color": "rgba(255,255,255,.85)"},
            "steps": [
                {"range": [0, 15], "color": "rgba(34,197,94,.45)"},
                {"range": [15, 25], "color": "rgba(234,179,8,.45)"},
                {"range": [25, 35], "color": "rgba(249,115,22,.45)"},
                {"range": [35, 45], "color": "rgba(239,68,68,.45)"},
            ],
        },
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=15))
    return fig


def _lamp_row(label: str, status: str, note: str = ""):
    color = {"正向":"🟢","中性":"🟡","負向":"🔴",
             "很低":"🟢","中等":"🟡","大幅":"🟠","深熊":"🔴"}.get(status, "⚪")
    return {"面向": label, "燈號": f"{color} {status}", "說明": note}


def _reference_tables():
    st.markdown("## 數值與燈號對照表")
    tabs = st.tabs(["四象限 X/Y 計算", "Macro 75%", "Market 25%", "回撤程度", "單次預備金"])

    with tabs[0]:
        st.markdown("### X 軸：景氣水準")
        st.write("X = 75% Macro 分數 + 25% Market 分數。")
        st.dataframe(pd.DataFrame([
            ["Macro", "75%", "國發會領先、外銷訂單、電子訂單、資通訂單、資訊電子生產"],
            ["Market", "25%", "3M動能、6M動能、vs 6M均線、vs 12M均線"],
        ], columns=["構成", "總權重", "來源"]), hide_index=True, use_container_width=True)

        st.markdown("### Y 軸：景氣動能")
        st.write("Y = X 的 3 個月變化（本月正式分數 − 3 個月前正式分數）。")
        st.dataframe(pd.DataFrame([
            ["擴張", "X ≥ 0", "Y ≥ 0"],
            ["惡化", "X ≥ 0", "Y < 0"],
            ["衰退", "X < 0", "Y < 0"],
            ["復甦／築底", "X < 0", "Y ≥ 0"],
        ], columns=["象限", "X 景氣水準", "Y 3M動能"]), hide_index=True, use_container_width=True)

    with tabs[1]:
        st.dataframe(pd.DataFrame([
            ["國發會景氣領先指標", "50%", "1M/3M動能 → rolling Z-score"],
            ["外銷訂單總額", "20%", "1M/3M動能 → rolling Z-score"],
            ["電子產品外銷訂單", "10%", "1M/3M動能 → rolling Z-score"],
            ["資訊通信外銷訂單", "10%", "1M/3M動能 → rolling Z-score"],
            ["資訊電子生產指數", "10%", "1M/3M動能 → rolling Z-score"],
        ], columns=["Macro指標", "Macro內權重", "計算"]), hide_index=True, use_container_width=True)
        st.caption("Macro總分 = 40% × 1M合成分數 + 60% × 3M合成分數；再占正式模型75%。")

    with tabs[2]:
        st.dataframe(pd.DataFrame([
            ["TAIEX 3M動能", "25%"],
            ["TAIEX 6M動能", "25%"],
            ["TAIEX vs 6M均線", "25%"],
            ["TAIEX vs 12M均線", "25%"],
        ], columns=["Market指標", "Market內權重"]), hide_index=True, use_container_width=True)
        st.caption("Market總分占正式模型25%；用途是市場確認，不取代Macro。")

    with tabs[3]:
        st.dataframe(pd.DataFrame([
            ["0~-15%", "🟢 很低", "正常定投，不動用熊市預備金"],
            ["-15~-25%", "🟡 中等", "開始參考Macro，觀察或小額加碼"],
            ["-25~-35%", "🟠 大幅", "價格權重提高，分批加碼"],
            ["≤-35%", "🔴 深熊", "Macro無否決權，但仍分批，不一次All-in"],
        ], columns=["Local Drawdown", "燈號", "意義"]), hide_index=True, use_container_width=True)

    with tabs[4]:
        st.dataframe(pd.DataFrame([
            ["正常定投／觀察", "0%"],
            ["小額加碼", "5%"],
            ["加碼", "10%"],
            ["積極加碼", "15%"],
            ["大幅分批／深熊強制分批", "20%"],
        ], columns=["決策", "占剩餘熊市預備金"]), hide_index=True, use_container_width=True)


def render_sidebar(defaults: dict) -> dict:
    _inject_css()
    st.sidebar.header("Macro Cycle Lab")
    start_date = st.sidebar.date_input(
        "起始日期",
        value=pd.Timestamp(defaults["start_date"]).date(),
        min_value=date(1990, 1, 1),
    )
    end_date = st.sidebar.date_input("結束日期", value=date.today())
    momentum_months = st.sidebar.slider(
        "景氣動能（月）",
        min_value=1,
        max_value=12,
        value=int(defaults["momentum_months"]),
    )
    use_pmi_extension = st.sidebar.toggle(
        "台股 PMI 擴充版",
        value=bool(defaults["use_pmi_extension"]),
        help="僅影響舊版台股研究模型；正式75/25台股決策模型不因此改權重。",
    )
    transaction_cost_bps = st.sidebar.number_input(
        "單向交易成本（bps）",
        min_value=0.0,
        max_value=100.0,
        value=float(defaults["transaction_cost_bps"]),
        step=1.0,
    )
    risk_free_rate = st.sidebar.number_input(
        "現金／短債年化報酬",
        min_value=0.0,
        max_value=0.20,
        value=float(defaults["risk_free_rate"]),
        step=0.005,
        format="%.3f",
    )
    initial_capital = st.sidebar.number_input(
        "回測初始資金",
        min_value=10_000,
        value=int(defaults["initial_capital"]),
        step=100_000,
    )
    uploaded_file = st.sidebar.file_uploader(
        "上傳自訂月資料 CSV",
        type=["csv"],
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
        settings["start_date"], settings["end_date"], spec["benchmark"]
    )
    data = merge_all_data(macro, features, uploaded)
    return data, status


def _weights_for_market(market: str, settings: dict):
    spec = MODEL_SPECS[market]
    weights = dict(spec["weights"])
    directions = dict(spec["directions"])

    if market == "TW" and settings["use_pmi_extension"]:
        pmi = spec["pmi_extension"]
        scale = 1 - pmi["weight"]
        weights = {k: v * scale for k, v in weights.items()}
        weights[pmi["indicator"]] = pmi["weight"]
        directions[pmi["indicator"]] = pmi["direction"]

    return weights, directions


def _run_market(market: str, settings: dict):
    data, status = _load_market_data(market, settings)
    weights, directions = _weights_for_market(market, settings)
    revised = run_model(data, weights, directions, settings["momentum_months"])

    original_spec = ORIGINAL_MODEL_SPECS[market]
    original = run_model(
        data,
        original_spec["weights"],
        MODEL_SPECS[market]["directions"],
        settings["momentum_months"],
    )
    return data, status, revised, original


def render_overview(settings: dict):
    _inject_css()
    st.subheader("投資儀表板總覽")
    st.caption("先看燈號與景氣位置；數值與計算公式放在頁面後段參考。")

    tw_status = None
    macro_path = Path("data/processed/taiwan_macro_inputs.csv")
    ndc_path = Path("data/raw/ndc_business_cycle.xlsx")

    if macro_path.exists() and ndc_path.exists():
        try:
            hist = build_official_history(macro_path, ndc_path)
            sig = latest_signal(hist)
            raw = pd.read_csv(macro_path, parse_dates=["date"]).sort_values("date")
            market = raw.dropna(subset=["taiex_close"])
            local = local_cycle_engine(market[["date", "taiex_close"]])
            dd = float(local.iloc[-1]["local_drawdown_pct"])
            regime = str(sig["macro_regime"])
            action, confidence, rationale = decide_action(dd, regime)
            tw_status = (hist, sig, dd, regime, action, confidence, rationale)
        except Exception:
            tw_status = None

    if tw_status:
        hist, sig, dd, regime, action, confidence, rationale = tw_status
        phase = PHASE_ZH.get(regime, regime)
        dd_icon, dd_label = _drawdown_status(dd)
        macro_icon, macro_label = _status_from_value(float(sig["macro_score"]))
        market_icon, market_label = _status_from_value(float(sig["market_score"]))

        c1, c2 = st.columns([1.35, 1])
        with c1:
            q = hist.dropna(subset=["official_score"]).copy()
            q["quadrant_y"] = q["official_score"].diff(3)
            st.plotly_chart(
                _quadrant_chart(q, "official_score", "quadrant_y", "台灣景氣四象限"),
                use_container_width=True
            )
        with c2:
            st.markdown(
                f"""
                <div class="mcl-card">
                  <div class="mcl-muted">投資決策總結</div>
                  <div class="mcl-big">{ACTION_LABELS[action]}</div>
                  <div>{rationale}</div>
                  <hr>
                  <div>景氣循環（75%）　{macro_icon} <b>{macro_label}</b></div>
                  <div>市場確認（25%）　{market_icon} <b>{market_label}</b></div>
                  <div>回撤程度　　　　 {dd_icon} <b>{dd_label}</b></div>
                  <div>景氣位置　　　　 <b>{phase}</b></div>
                  <hr>
                  <div class="mcl-muted">單次建議動用剩餘熊市預備金</div>
                  <div class="mcl-big">{TRANCHE_MAP[action]}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        lamps = pd.DataFrame([
            _lamp_row("景氣循環（75%）", macro_label, "景氣與企業基本面"),
            _lamp_row("市場確認（25%）", market_label, "股價動能與均線"),
            _lamp_row("回撤程度", dd_label, f"Local Drawdown {dd:.1f}%"),
            _lamp_row("投資環境", "正向" if action == "NORMAL_DCA" else "中性", ACTION_LABELS[action]),
        ])
        st.markdown("### 四大方向燈號")
        st.dataframe(lamps, hide_index=True, use_container_width=True)
    else:
        st.warning("台股正式資料尚未就緒；可先查看美股／台股研究模型。")

    st.markdown("### 使用原則")
    st.info(
        "核心邏輯：景氣決定『環境』、市場確認決定『是否被價格反映』、"
        "Local Drawdown 決定『是否值得動用熊市預備金』。"
    )


def render_market_page(market: str, settings: dict):
    _inject_css()
    label = "美股模型" if market == "US" else "台股模型"
    st.subheader(label)
    st.caption("先看四象限與燈號；詳細分數與權重放到頁面下方。")

    try:
        data, status, revised, original = _run_market(market, settings)
    except Exception as exc:
        st.error(f"資料載入失敗：{exc}")
        return

    frame = revised.frame.dropna(subset=["SCORE"]).copy()
    if frame.empty:
        st.error("目前可用指標不足，無法計算模型。")
        return

    latest = frame.iloc[-1]
    frame["QUAD_Y"] = frame["SCORE"].diff(3)
    latest_y = float(frame["QUAD_Y"].iloc[-1]) if pd.notna(frame["QUAD_Y"].iloc[-1]) else 0.0
    phase, phase_note = _phase_from_xy(float(latest["SCORE"]), latest_y)
    score_icon, score_lamp = _status_from_value(float(latest["SCORE"]))
    mom_icon, mom_lamp = _status_from_value(float(latest["MOMENTUM_3M"]))
    exp = float(latest["EXPOSURE"])

    chart_frame = frame.reset_index()
    if "date" not in chart_frame.columns:
        chart_frame = chart_frame.rename(columns={chart_frame.columns[0]: "date"})

    c1, c2 = st.columns([1.35, 1])
    with c1:
        st.plotly_chart(
            _quadrant_chart(chart_frame, "SCORE", "QUAD_Y", f"{label}｜景氣四象限"),
            use_container_width=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="mcl-card">
              <div class="mcl-muted">目前位置</div>
              <div class="mcl-big">{phase}</div>
              <div>{phase_note}</div>
              <hr>
              <div>景氣水準　{score_icon} <b>{score_lamp}</b></div>
              <div>3M 動能　 {mom_icon} <b>{mom_lamp}</b></div>
              <div>模型建議曝險　<b>{exp:.0%}</b></div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 核心燈號")
    core = pd.DataFrame([
        _lamp_row("景氣水準", score_lamp, "模型綜合水準"),
        _lamp_row("景氣動能", mom_lamp, "3個月變化"),
        _lamp_row("建議曝險", "正向" if exp >= .65 else ("中性" if exp >= .35 else "負向"),
                  f"目前模型曝險 {exp:.0%}"),
    ])
    st.dataframe(core, hide_index=True, use_container_width=True)

    st.markdown("### 關鍵指標燈號")
    rows = []
    for ind, w in sorted(revised.used_weights.items(), key=lambda kv: kv[1], reverse=True):
        if ind in revised.frame.columns:
            s = pd.to_numeric(revised.frame[ind], errors="coerce").dropna()
            if len(s) >= 4:
                change = float(s.iloc[-1] - s.iloc[-4])
                icon, lamp = _status_from_value(change, pos=0, neg=0)
            else:
                icon, lamp = "⚪", "資料不足"
        else:
            icon, lamp = "⚪", "資料不足"
        rows.append([ind, f"{icon} {lamp}", f"{w:.0%}"])

    st.dataframe(
        pd.DataFrame(rows, columns=["指標", "目前燈號", "模型權重"]),
        hide_index=True, use_container_width=True
    )

    with st.expander("查看數值、權重與原模型比較"):
        weight_df = pd.DataFrame({
            "指標": revised.used_weights.keys(),
            "正規化權重": revised.used_weights.values(),
        }).sort_values("正規化權重", ascending=False)
        st.dataframe(weight_df.style.format({"正規化權重": "{:.1%}"}),
                     hide_index=True, use_container_width=True)

        compare = pd.DataFrame({
            "修正版分數": revised.frame["SCORE"],
            "原模型分數": original.frame["SCORE"],
        }).tail(24)
        st.line_chart(compare)

        if revised.missing_indicators:
            st.warning("缺少指標：" + "、".join(revised.missing_indicators))

    with st.expander("燈號判定方式"):
        st.write("模型分數 ≥ +0.10：🟢 正向；-0.10～+0.10：🟡 中性；≤ -0.10：🔴 負向。")
        st.write("四象限：X = 模型景氣水準；Y = X 的3個月變化。")


def render_taiwan_official_page(settings: dict):
    _inject_css()
    st.subheader("台股投資決策｜正式版")
    st.caption("75% Macro / 25% Market｜四象限｜Local Drawdown｜V3.1")

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

    latest_market = market.iloc[-1]
    market_date = pd.Timestamp(latest_market["date"])
    market_price = float(latest_market["taiex_close"])
    dd = float(local.iloc[-1]["local_drawdown_pct"])

    macro_date = pd.Timestamp(sig["date"])
    regime = str(sig["macro_regime"])
    action, confidence, rationale = decide_action(dd, regime)

    macro_icon, macro_lamp = _status_from_value(float(sig["macro_score"]))
    market_icon, market_lamp = _status_from_value(float(sig["market_score"]))
    dd_icon, dd_lamp = _drawdown_status(dd)

    hist_q = hist.dropna(subset=["official_score"]).copy()
    hist_q["Q_Y"] = hist_q["official_score"].diff(3)
    latest_q = hist_q.dropna(subset=["Q_Y"]).iloc[-1]
    quadrant_name, quadrant_note = _phase_from_xy(
        float(latest_q["official_score"]), float(latest_q["Q_Y"])
    )

    c1, c2 = st.columns([1.45, 1])
    with c1:
        st.plotly_chart(
            _quadrant_chart(hist_q, "official_score", "Q_Y", "景氣四象限｜最近12個成熟月份"),
            use_container_width=True
        )
        st.success(f"目前位置：**{quadrant_name}**｜{quadrant_note}")
    with c2:
        st.markdown(
            f"""
            <div class="mcl-card">
              <div class="mcl-muted">投資決策總結</div>
              <div class="mcl-big">{ACTION_LABELS[action]}</div>
              <div>{rationale}</div>
              <hr>
              <div>景氣循環 75%　{macro_icon} <b>{macro_lamp}</b></div>
              <div>市場確認 25%　{market_icon} <b>{market_lamp}</b></div>
              <div>回撤程度　　　{dd_icon} <b>{dd_lamp}</b></div>
              <div>整體景氣位置　<b>{PHASE_ZH.get(regime, regime)}</b></div>
              <hr>
              <div class="mcl-muted">單次建議動用剩餘熊市預備金</div>
              <div class="mcl-big">{TRANCHE_MAP[action]}%</div>
              <div class="mcl-muted">信心：{confidence}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### 四大方向燈號")
    lamps = pd.DataFrame([
        _lamp_row("景氣循環（75%）", macro_lamp, "景氣與企業基本面"),
        _lamp_row("市場確認（25%）", market_lamp, "TAIEX動能與均線"),
        _lamp_row("回撤程度", dd_lamp, f"{dd:.1f}%"),
        _lamp_row("投資環境", "正向" if action == "NORMAL_DCA" else "中性", ACTION_LABELS[action]),
    ])
    st.dataframe(lamps, hide_index=True, use_container_width=True)

    st.markdown("### 景氣領先指標｜Macro 75%")
    latest = hist.dropna(subset=["official_score"]).iloc[-1]

    macro_items = [
        ("國發會景氣領先指標", latest.get("ndc_z3"), "Macro內50%"),
        ("外銷訂單總額", latest.get("orders_total_z3"), "Macro內20%"),
        ("電子產品外銷訂單", latest.get("orders_elec_z3"), "Macro內10%"),
        ("資訊通信外銷訂單", latest.get("orders_ict_z3"), "Macro內10%"),
        ("資訊電子生產指數", latest.get("mfg_elec_z3"), "Macro內10%"),
    ]
    macro_rows = []
    for name, value, w in macro_items:
        icon, lamp = _status_from_value(value, pos=.20, neg=-.20)
        macro_rows.append([name, f"{icon} {lamp}", w])
    st.dataframe(
        pd.DataFrame(macro_rows, columns=["指標", "燈號", "權重"]),
        hide_index=True, use_container_width=True
    )

    st.markdown("### 市場確認指標｜Market 25%")
    market_items = [
        ("TAIEX 3個月動能", latest.get("market_m3_z"), "Market內25%"),
        ("TAIEX 6個月動能", latest.get("market_m6_z"), "Market內25%"),
        ("TAIEX vs 6月均線", latest.get("market_ma6_z"), "Market內25%"),
        ("TAIEX vs 12月均線", latest.get("market_ma12_z"), "Market內25%"),
    ]
    market_rows = []
    for name, value, w in market_items:
        icon, lamp = _status_from_value(value, pos=.20, neg=-.20)
        market_rows.append([name, f"{icon} {lamp}", w])
    st.dataframe(
        pd.DataFrame(market_rows, columns=["指標", "燈號", "權重"]),
        hide_index=True, use_container_width=True
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(_gauge_drawdown(dd, "Local Drawdown"), use_container_width=True)
        st.caption(f"最新市場：{market_date.strftime('%Y-%m')}｜TAIEX {market_price:,.0f}")
    with c2:
        st.markdown("### 資產行動建議")
        if action == "NORMAL_DCA":
            st.success("維持分批定期定額；目前不動用熊市預備金。")
        elif action in ("OBSERVE", "SMALL_ADD"):
            st.warning("保持紀律，等待更深回撤或Macro改善後再提高投入。")
        else:
            st.warning(f"啟動分批加碼；本次建議使用剩餘熊市預備金 {TRANCHE_MAP[action]}%。")
        st.write("**資料時點：**")
        st.write(f"- 市場：{market_date.strftime('%Y-%m')}")
        st.write(f"- Macro成熟資料：{macro_date.strftime('%Y-%m')}")

    with st.expander("數值、燈號、X/Y與權重計算依據（參考）"):
        _reference_tables()


def render_backtest_page(settings: dict):
    _inject_css()
    st.subheader("策略回測")
    st.caption("先看是否改善報酬／回撤，再看詳細數值。")
    market = st.radio("市場", ["US", "TW"], horizontal=True)

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
        price, revised_exposure, settings["initial_capital"],
        settings["transaction_cost_bps"], settings["risk_free_rate"]
    )
    bt_original = run_exposure_backtest(
        price, original_exposure, settings["initial_capital"],
        settings["transaction_cost_bps"], settings["risk_free_rate"]
    )
    bt_staged = run_exposure_backtest(
        price, staged, settings["initial_capital"],
        settings["transaction_cost_bps"], settings["risk_free_rate"]
    )
    bt_hold = run_exposure_backtest(
        price, buy_hold_exp, settings["initial_capital"],
        0, settings["risk_free_rate"]
    )

    chart_data = {
        "修正版動態配置": bt_revised,
        "原模型動態配置": bt_original,
        "85%核心＋15%回撤彈藥": bt_staged,
        "Buy & Hold": bt_hold,
    }

    metrics = {}
    for name, bt in chart_data.items():
        metrics[name] = performance_metrics(
            bt["STRATEGY_EQUITY"], bt["STRATEGY_RETURN"], settings["risk_free_rate"]
        )

    main = metrics["修正版動態配置"]
    hold = metrics["Buy & Hold"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("年化報酬率", f"{main['CAGR']:.1%}",
              delta=f"{main['CAGR']-hold['CAGR']:+.1%} vs Buy&Hold")
    c2.metric("最大回撤", f"{main['最大回撤']:.1%}",
              delta=f"{main['最大回撤']-hold['最大回撤']:+.1%}")
    c3.metric("Sharpe", f"{main['Sharpe']:.2f}",
              delta=f"{main['Sharpe']-hold['Sharpe']:+.2f}")
    c4.metric("整體判定",
              "🟢 改善" if (main["Sharpe"] >= hold["Sharpe"] and main["最大回撤"] >= hold["最大回撤"])
              else "🟡 混合")

    st.plotly_chart(equity_chart(chart_data), use_container_width=True)

    st.markdown("### 回測燈號")
    signals = pd.DataFrame([
        _lamp_row("報酬", "正向" if main["CAGR"] >= hold["CAGR"] else "中性",
                  "修正版CAGR vs Buy&Hold"),
        _lamp_row("回撤控制", "正向" if main["最大回撤"] >= hold["最大回撤"] else "負向",
                  "最大回撤越接近0越好"),
        _lamp_row("風險調整", "正向" if main["Sharpe"] >= hold["Sharpe"] else "中性",
                  "Sharpe比較"),
    ])
    st.dataframe(signals, hide_index=True, use_container_width=True)

    with st.expander("完整回測數值"):
        rows = [{"策略": k, **v} for k, v in metrics.items()]
        metrics_df = pd.DataFrame(rows).set_index("策略")
        st.dataframe(metrics_df.style.format({
            "總報酬": "{:.1%}",
            "CAGR": "{:.1%}",
            "年化波動": "{:.1%}",
            "Sharpe": "{:.2f}",
            "最大回撤": "{:.1%}",
        }), use_container_width=True)

        st.info(
            "回測用於比較規則穩健性，不代表未來報酬。正式投資決策仍以"
            "「景氣四象限 × 市場確認 × Local Drawdown」為主。"
        )


def render_data_page(settings: dict):
    _inject_css()
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

    with st.expander("最新資料日期與有效筆數"):
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

    with st.expander("CSV 欄名參考"):
        st.code(
            "TW_EXPORTS, TW_EXPORT_ORDERS, TW_SEMI_EXPORTS,\n"
            "TW_INDUSTRIAL_PRODUCTION, TW_MONEY_M1B, TW_PMI,\n"
            "US_LEI_PROXY, GLOBAL_SEMI"
        )
