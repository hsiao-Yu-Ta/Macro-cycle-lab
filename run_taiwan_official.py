
from pathlib import Path
import pandas as pd
from src.taiwan_decision_engine import build_official_history, latest_signal

MACRO = Path("data/processed/taiwan_macro_inputs.csv")
NDC = Path("data/raw/ndc_business_cycle.xlsx")
OUTDIR = Path("data/processed")

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    hist = build_official_history(MACRO, NDC)
    sig = latest_signal(hist)

    hist.to_csv(OUTDIR / "taiwan_official_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([sig]).to_csv(
        OUTDIR / "taiwan_official_signal.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("===== TAIWAN MACRO CYCLE LAB OFFICIAL SIGNAL =====")
    print("Date:", sig["date"])
    print("TAIEX:", round(float(sig["taiex_close"]), 2))
    print("Local drawdown %:", round(float(sig["local_drawdown_pct"]), 2))
    print("Regime:", sig["macro_regime"])
    print("Official score:", round(float(sig["official_score"]), 3))
    print("Action:", sig["action_label_zh"])
    print("Confidence:", sig["confidence"])
    print("Suggested tranche % of remaining reserve:",
          sig["suggested_tranche_pct_of_remaining_reserve"])
    print("New trigger:", bool(sig["new_trigger"]))
    print("Rationale:", sig["rationale"])

if __name__ == "__main__":
    main()
