
import streamlit as st
import pandas as pd
from pathlib import Path
from src.taiwan_decision_engine import build_official_history, latest_signal

st.set_page_config(page_title="Macro Cycle Lab｜台灣正式版", layout="wide")

MACRO = Path("data/processed/taiwan_macro_inputs.csv")
NDC = Path("data/raw/ndc_business_cycle.xlsx")

st.title("Macro Cycle Lab｜台灣投資決策正式版")
st.caption("75% Macro / 25% Market｜Local Drawdown｜V3.1 Robustness Rules")

if not MACRO.exists():
    st.error("找不到 data/processed/taiwan_macro_inputs.csv。請先執行 fetch_taiwan_full.py。")
    st.stop()

if not NDC.exists():
    st.error("找不到 data/raw/ndc_business_cycle.xlsx。")
    st.stop()

hist = build_official_history(MACRO, NDC)
sig = latest_signal(hist)

date = pd.Timestamp(sig["date"]).strftime("%Y-%m")
dd = float(sig["local_drawdown_pct"])
score = float(sig["official_score"])
taiex = float(sig["taiex_close"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("資料月份", date)
c2.metric("TAIEX", f"{taiex:,.0f}")
c3.metric("Local Drawdown", f"{dd:.1f}%")
c4.metric("Macro Regime", sig["macro_regime"])

st.divider()

c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    st.subheader("目前投資訊號")
    st.markdown(f"## {sig['action_label_zh']}")
    st.write(sig["rationale"])

with c2:
    st.subheader("信心")
    st.markdown(f"## {sig['confidence']}")
    st.caption("STRONG = V3.1穩健性較高；TENTATIVE/WEAK = 樣本較少或結果較不穩健。")

with c3:
    st.subheader("熊市預備金")
    tranche = int(sig["suggested_tranche_pct_of_remaining_reserve"])
    st.markdown(f"## {tranche}%")
    st.caption("指『剩餘熊市預備金』的單次建議分批比例；不是總資產比例。")
    if bool(sig["new_trigger"]):
        st.success("本月為較高級別的新觸發訊號。")
    else:
        st.info("目前不是新的升級觸發；避免同一訊號每月重複加碼。")

st.divider()

st.subheader("正式決策矩陣")
matrix = pd.DataFrame([
    ["0~-15%", "正常定投", "正常定投", "正常定投", "不動用主要熊市彈藥"],
    ["-15~-20%", "觀察", "小額加碼", "小額加碼", "觀察區"],
    ["-20~-25%", "觀察／保守", "加碼", "積極加碼", "V3.1：Macro辨識價值 STRONG"],
    ["-25~-30%", "開始加碼", "積極加碼", "積極加碼", "V3.1：價格機會 STRONG"],
    ["-30~-35%", "積極分批", "大幅分批", "大幅分批", "樣本較少，避免一次All-in"],
    ["≤-35%", "強制分批", "強力分批", "強力分批", "Macro無否決權；V3.1不支持無腦All-in"],
], columns=["TAIEX回撤", "Macro惡化", "Macro築底", "Macro復甦", "說明"])
st.dataframe(matrix, use_container_width=True, hide_index=True)

st.subheader("最近 24 個月")
show_cols = [
    "date", "taiex_close", "local_drawdown_pct",
    "macro_regime", "official_score",
    "action_label_zh", "confidence",
    "suggested_tranche_pct_of_remaining_reserve",
]
recent = hist[show_cols].dropna(subset=["official_score"]).tail(24).copy()
recent["date"] = recent["date"].dt.strftime("%Y-%m")
st.dataframe(recent, use_container_width=True, hide_index=True)

st.caption(
    "模型定位：景氣狀態與回撤決策輔助，不是短線進出預測。"
    "正式版鎖定75/25，不再以歷史資料持續微調權重。"
)
