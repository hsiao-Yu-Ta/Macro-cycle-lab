# Macro Cycle Lab

Macro Cycle Lab 是一套以 Streamlit 建立的台股／美股景氣循環分析與回測工具。

## 已內建的核心邏輯

### 美股修正版
- 75%：經濟與金融領先指標
- 25%：市場確認
- 股價均線訊號在總分中的目標權重為 7%
- 景氣階段同時使用：
  - 綜合分數
  - 3 個月分數動能
- 支援與「原模型」並列比較

### 台股修正版
- 75%：出口、半導體與全球領先指標
- 25%：市場確認
- 股價均線訊號在總分中的目標權重為 7%
- 2000 年起主回測不強制使用 PMI
- 2012 年後可切換 PMI 擴充版
- 景氣階段同時使用綜合分數與 3 個月動能

## 功能

- 自動抓取 Yahoo Finance 月資料
- 自動抓取 FRED 公開 CSV 資料
- 台灣官方資料可用 CSV 上傳補充
- 指標標準化、方向調整、加權評分
- 3 個月動能判定
- 景氣階段分類
- 股價市場確認
- 原模型與修正版比較
- Buy & Hold、分數配置、分段加碼策略回測
- 最大回撤、CAGR、年化波動、Sharpe 等績效指標
- 可下載月度模型結果 CSV

## 部署方式

### 本機執行

```bash
git clone <你的 GitHub Repository URL>
cd macro-cycle-lab
python -m venv .venv
```

Windows：

```bash
.venv\Scripts\activate
```

macOS / Linux：

```bash
source .venv/bin/activate
```

安裝並啟動：

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Community Cloud

1. 將本資料夾全部上傳至 GitHub。
2. 進入 Streamlit Community Cloud。
3. 選擇 Repository。
4. Main file path 填入 `app.py`。
5. Deploy。

## 資料說明

公開資料來源可能因代碼調整、發布延遲或網路限制而抓取失敗。程式會：

1. 優先抓取網路資料。
2. 使用本機快取。
3. 找不到必要資料時顯示缺漏狀態，不會偷偷填入虛構數據。
4. 支援上傳自訂月資料 CSV。

CSV 格式：

```csv
date,value
2020-01-31,100
2020-02-29,101
```

或寬表：

```csv
date,US_LEI,US_YIELD_CURVE,TW_EXPORTS
2020-01-31,99.1,0.7,25000
```

## 模型限制

本工具是投資研究與資產配置輔助工具，不代表未來報酬保證。總體資料通常具有發布落後與修訂問題；正式回測若需避免前視偏誤，應另外保存 vintage data 或 realtime data。
