"""
Macro Cycle Lab - Taiwan Official Data Fetcher
Sources:
1) MOEA export orders direct CSV
2) MOEA manufacturing production index direct CSV
3) Customs Administration total trade direct CSV
4) TWSE TAIEX monthly historical pages

Output:
data/processed/taiwan_macro_inputs.csv
"""

from io import BytesIO
from pathlib import Path
import time
import pandas as pd
import requests

URLS = {
    "orders_total": "https://service.moea.gov.tw/EE520/opendata/b.csv",
    "orders_electronics": "https://service.moea.gov.tw/EE520/opendata/%E7%B6%93%E6%BF%9F%E9%83%A8%E7%B5%B1%E8%A8%88%E8%99%95_%E5%A4%96%E9%8A%B7%E8%A8%82%E5%96%AE_%E9%9B%BB%E5%AD%90%E7%94%A2%E5%93%81.csv",
    "orders_ict": "https://service.moea.gov.tw/EE520/opendata/%E7%B6%93%E6%BF%9F%E9%83%A8%E7%B5%B1%E8%A8%88%E8%99%95_%E5%A4%96%E9%8A%B7%E8%A8%82%E5%96%AE_%E8%B3%87%E8%A8%8A%E9%80%9A%E8%A8%8A%E7%94%A2%E5%93%81.csv",
    "manufacturing": "https://service.moea.gov.tw/EE520/opendata/%E7%B6%93%E6%BF%9F%E9%83%A8%E7%B5%B1%E8%A8%88%E8%99%95_%E8%A3%BD%E9%80%A0%E6%A5%AD%E7%94%9F%E7%94%A2%E9%87%8F%E6%8C%87%E6%95%B8.csv",
    "customs_trade": "https://opendata.customs.gov.tw/data/6053/csv.csv",
}

TWSE_URL = "https://www.twse.com.tw/indicesReport/MI_5MINS_HIST?date={date}&response=html"

HEADERS = {
    "User-Agent": "Mozilla/5.0 MacroCycleLab/1.0"
}

def read_csv_url(url):
    r = requests.get(url, headers=HEADERS, timeout=45)
    r.raise_for_status()
    return pd.read_csv(BytesIO(r.content), encoding="utf-8-sig")

def roc_ym_to_date(v):
    s = str(int(v))
    return pd.Timestamp(int(s[:-2]) + 1911, int(s[-2:]), 1)

def add_momentum(df, raw_cols):
    df = df.sort_values("date").copy()
    for col in raw_cols:
        root = col
        for suffix in ["_usd_mn", "_index", "_ntd_thousand", "_close"]:
            root = root.replace(suffix, "")
        df[f"{root}_yoy_pct"] = df[col].pct_change(12) * 100
        df[f"{root}_mom_3m_pct"] = df[col].pct_change(3) * 100
        df[f"{root}_mom_6m_pct"] = df[col].pct_change(6) * 100
    return df

def fetch_export_orders():
    total = read_csv_url(URLS["orders_total"])
    elec = read_csv_url(URLS["orders_electronics"])
    ict = read_csv_url(URLS["orders_ict"])

    a = pd.DataFrame({
        "date": total["資料期(民國年)"].map(roc_ym_to_date),
        "export_orders_total_usd_mn": pd.to_numeric(total["統計值(美元)"], errors="coerce"),
    })
    b = pd.DataFrame({
        "date": elec["資料期(民國年)"].map(roc_ym_to_date),
        "export_orders_electronics_usd_mn": pd.to_numeric(elec["統計值(金額)"], errors="coerce"),
    })
    c = pd.DataFrame({
        "date": ict["資料期(民國年)"].map(roc_ym_to_date),
        "export_orders_ict_usd_mn": pd.to_numeric(ict["統計值(金額)"], errors="coerce"),
    })

    df = a.merge(b,on="date",how="outer").merge(c,on="date",how="outer")
    return add_momentum(df, [x for x in df.columns if x != "date"])

def fetch_manufacturing():
    raw = read_csv_url(URLS["manufacturing"])
    raw["date"] = raw["資料期(民國年)"].map(roc_ym_to_date)
    raw["value"] = pd.to_numeric(raw["統計值(指數)"], errors="coerce")

    df = (
        raw.pivot_table(index="date", columns="行業別", values="value", aggfunc="first")
           .reset_index()
           .rename(columns={
               "金屬機電工業":"mfg_metal_machinery_index",
               "資訊電子工業":"mfg_info_electronics_index",
               "化學工業":"mfg_chemical_index",
               "民生工業":"mfg_consumer_index",
           })
    )
    return add_momentum(df, [x for x in df.columns if x != "date"])

def fetch_customs_trade():
    raw = read_csv_url(URLS["customs_trade"])
    raw["month_num"] = raw["月份"].astype(str).str.extract(r"(\d+)", expand=False)
    raw["date"] = pd.to_datetime(
        (raw["年度"] + 1911).astype(str) + "-" + raw["month_num"] + "-01",
        errors="coerce"
    )

    df = pd.DataFrame({
        "date": raw["date"],
        "exports_ntd_thousand": pd.to_numeric(raw["出口總值(新臺幣千元)"], errors="coerce"),
        "imports_ntd_thousand": pd.to_numeric(raw["進口總值(新臺幣千元)"], errors="coerce"),
        "trade_balance_ntd_thousand": pd.to_numeric(raw["出入超(新臺幣千元)"], errors="coerce"),
    }).dropna(subset=["date"]).sort_values("date")

    df = add_momentum(df, ["exports_ntd_thousand","imports_ntd_thousand"])
    return df

def _roc_date_to_ts(v):
    s = str(v).strip()
    parts = s.split("/")
    if len(parts) != 3:
        return pd.NaT
    return pd.Timestamp(int(parts[0]) + 1911, int(parts[1]), int(parts[2]))

def fetch_taiex_month(year, month):
    date_arg = f"{year:04d}{month:02d}01"
    url = TWSE_URL.format(date=date_arg)

    # read_html parses the official TWSE table directly
    tables = pd.read_html(url)
    target = None
    for t in tables:
        cols = [str(c) for c in t.columns]
        if any("日期" in c for c in cols) and any("收盤指數" in c for c in cols):
            target = t
            break
    if target is None:
        return pd.DataFrame()

    target.columns = [str(c) for c in target.columns]
    date_col = next(c for c in target.columns if "日期" in c)
    close_col = next(c for c in target.columns if "收盤指數" in c)

    out = pd.DataFrame({
        "trade_date": target[date_col].map(_roc_date_to_ts),
        "taiex_close": pd.to_numeric(
            target[close_col].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )
    }).dropna()
    return out

def fetch_taiex(start="1999-01-01", sleep_seconds=0.25):
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp.today().normalize()

    months = pd.period_range(start_ts, end_ts, freq="M")
    parts = []

    for i, p in enumerate(months, start=1):
        try:
            d = fetch_taiex_month(p.year, p.month)
            if not d.empty:
                parts.append(d)
        except Exception as e:
            print(f"TWSE warning {p}: {e}")
        if i < len(months):
            time.sleep(sleep_seconds)

    if not parts:
        raise RuntimeError("No TAIEX data returned from TWSE.")

    daily = pd.concat(parts, ignore_index=True).drop_duplicates("trade_date").sort_values("trade_date")
    monthly = (
        daily.set_index("trade_date")["taiex_close"]
             .resample("MS")
             .last()
             .dropna()
             .rename_axis("date")
             .reset_index()
    )

    monthly["taiex_return_1m_pct"] = monthly["taiex_close"].pct_change(1) * 100
    monthly["taiex_return_3m_pct"] = monthly["taiex_close"].pct_change(3) * 100
    monthly["taiex_return_6m_pct"] = monthly["taiex_close"].pct_change(6) * 100
    monthly["taiex_return_12m_pct"] = monthly["taiex_close"].pct_change(12) * 100

    monthly["taiex_ma_3m"] = monthly["taiex_close"].rolling(3).mean()
    monthly["taiex_ma_6m"] = monthly["taiex_close"].rolling(6).mean()
    monthly["taiex_ma_12m"] = monthly["taiex_close"].rolling(12).mean()

    monthly["taiex_vs_ma_3m_pct"] = (monthly["taiex_close"] / monthly["taiex_ma_3m"] - 1) * 100
    monthly["taiex_vs_ma_6m_pct"] = (monthly["taiex_close"] / monthly["taiex_ma_6m"] - 1) * 100
    monthly["taiex_vs_ma_12m_pct"] = (monthly["taiex_close"] / monthly["taiex_ma_12m"] - 1) * 100

    rolling_high = monthly["taiex_close"].cummax()
    monthly["taiex_drawdown_from_peak_pct"] = (monthly["taiex_close"] / rolling_high - 1) * 100

    return monthly

def main():
    outdir = Path("data/processed")
    outdir.mkdir(parents=True, exist_ok=True)

    print("1/4 Fetching export orders...")
    orders = fetch_export_orders()

    print("2/4 Fetching manufacturing index...")
    manufacturing = fetch_manufacturing()

    print("3/4 Fetching Customs trade...")
    trade = fetch_customs_trade()

    print("4/4 Fetching TAIEX from TWSE...")
    taiex = fetch_taiex()

    master = (
        orders.merge(manufacturing, on="date", how="outer")
              .merge(trade, on="date", how="outer")
              .merge(taiex, on="date", how="outer")
              .sort_values("date")
    )

    outfile = outdir / "taiwan_macro_inputs.csv"
    master.to_csv(outfile, index=False, encoding="utf-8-sig")

    print(f"Saved: {outfile}")
    print(f"Range: {master['date'].min().date()} -> {master['date'].max().date()}")
    print(f"Rows: {len(master)}")

if __name__ == "__main__":
    main()
