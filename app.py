from __future__ import annotations

import streamlit as st

from src.config import APP_TITLE, DEFAULT_SETTINGS
from src.ui import (
    render_sidebar,
    render_overview,
    render_market_page,
    render_taiwan_official_page,
    render_backtest_page,
    render_data_page,
)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Macro Cycle Lab")
st.caption("台股／美股景氣循環評分、3 個月動能、市場確認與資產回測")

settings = render_sidebar(DEFAULT_SETTINGS)

page = st.sidebar.radio(
    "功能",
    ["總覽", "美股模型", "台股模型", "台股投資決策", "策略回測", "資料與診斷"],
)

if page == "總覽":
    render_overview(settings)
elif page == "美股模型":
    render_market_page("US", settings)
elif page == "台股模型":
    render_market_page("TW", settings)
elif page == "台股投資決策":
    render_taiwan_official_page(settings)
elif page == "策略回測":
    render_backtest_page(settings)
else:
    render_data_page(settings)
