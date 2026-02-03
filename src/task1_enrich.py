from __future__ import annotations

import os
import re
import sys
import pandas as pd

RAW_PATH = "data/raw/ethiopia_fi_unified_data.csv"
OUT_PATH = "data/processed/ethiopia_fi_unified_data_enriched.csv"

COLLECTED_BY = "AbbyGailErm"
COLLECTION_DATE = "2026-02-03"

ID_COL = "record_id"
DATE_COL = "observation_date"

def ensure_dirs() -> None:
    os.makedirs("data/processed", exist_ok=True)

def mk_base_row(cols: list[str]) -> dict:
    return {c: pd.NA for c in cols}

def next_id(existing_ids: pd.Series, prefix: str) -> str:
    """
    Generate next ID of form PREFIX_0001 based on existing record_id values.
    Works for REC_, EVT_, LNK_.
    """
    ids = existing_ids.dropna().astype(str).tolist()
    nums: list[int] = []
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    for x in ids:
        m = pat.match(x)
        if m:
            try:
                nums.append(int(m.group(1)))
            except ValueError:
                pass
    nxt = (max(nums) + 1) if nums else 1
    return f"{prefix}_{nxt:04d}"

def fix_collector_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    In your sample, collected_by contains a date and collection_date is null.
    This corrects rows where collected_by looks like YYYY-MM-DD and collection_date is missing.
    """
    out = df.copy()
    if "collected_by" not in out.columns or "collection_date" not in out.columns:
        return out

    mask = (
        out["collected_by"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
        & out["collection_date"].isna()
    )
    out.loc[mask, "collection_date"] = out.loc[mask, "collected_by"]
    out.loc[mask, "collected_by"] = "Example_Trainee"
    return out

def add_event(
    df: pd.DataFrame,
    cols: list[str],
    *,
    indicator: str,
    indicator_code: str,
    category: str,
    event_date: str,
    source_name: str,
    source_type: str,
    source_url: str,
    original_text: str,
    confidence: str = "medium",
    notes: str = "",
) -> tuple[pd.DataFrame, dict]:
    row = mk_base_row(cols)
    row[ID_COL] = next_id(df[ID_COL], "EVT")
    row["record_type"] = "event"
    row["category"] = category
    row["pillar"] = pd.NA  # events should not be assigned to pillars
    row["indicator"] = indicator
    row["indicator_code"] = indicator_code

    row["value_text"] = "Occurred"
    row["value_type"] = "categorical"
    row[DATE_COL] = event_date
    row["fiscal_year"] = str(pd.to_datetime(event_date).year)

    row["gender"] = "all"
    row["location"] = "national"
    row["region"] = pd.NA

    row["source_name"] = source_name
    row["source_type"] = source_type
    row["source_url"] = source_url
    row["confidence"] = confidence

    row["collected_by"] = COLLECTED_BY
    row["collection_date"] = COLLECTION_DATE
    row["original_text"] = original_text
    row["notes"] = notes

    df2 = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df2, row

def add_impact_link(
    df: pd.DataFrame,
    cols: list[str],
    *,
    parent_id: str,
    pillar: str,
    related_indicator: str,
    impact_direction: str,
    impact_magnitude: float,
    lag_months: int,
    evidence_basis: str,
    relationship_type: str = "intervention",
    comparable_country: str = "Ethiopia",
    confidence: str = "medium",
    notes: str = "",
) -> tuple[pd.DataFrame, dict]:
    row = mk_base_row(cols)
    row[ID_COL] = next_id(df[ID_COL], "LNK")
    row["record_type"] = "impact_link"
    row["parent_id"] = parent_id

    row["pillar"] = pillar
    row["related_indicator"] = related_indicator
    row["relationship_type"] = relationship_type
    row["impact_direction"] = impact_direction
    row["impact_magnitude"] = float(impact_magnitude)
    row["lag_months"] = int(lag_months)
    row["evidence_basis"] = evidence_basis
    row["comparable_country"] = comparable_country

    row["confidence"] = confidence
    row["collected_by"] = COLLECTED_BY
    row["collection_date"] = COLLECTION_DATE
    row["notes"] = notes or "Impact link added for event-augmented modeling; calibrate in Task 3."

    df2 = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df2, row

def main() -> int:
    ensure_dirs()

    if not os.path.exists(RAW_PATH):
        print(f"[ERROR] Raw dataset not found at: {RAW_PATH}", file=sys.stderr)
        return 2

    df = pd.read_csv(RAW_PATH)
    cols = df.columns.tolist()

    for required in [ID_COL, "record_type", "indicator_code", DATE_COL]:
        if required not in df.columns:
            print(f"[ERROR] Missing required column: {required}", file=sys.stderr)
            return 2

    df = fix_collector_fields(df)

    # Build event lookup by indicator_code
    events = df[df["record_type"].eq("event")].copy()
    event_code_to_id = (
        events[[ID_COL, "indicator_code"]]
        .dropna(subset=["indicator_code"])
        .set_index("indicator_code")[ID_COL]
        .to_dict()
    )

    # 1) Add optional missing events (only if absent)
    if "EVT_INTEROP_CROSSOVER" not in event_code_to_id:
        df, evt = add_event(
            df, cols,
            indicator="Interoperable P2P transfers surpass ATM cash withdrawals",
            indicator_code="EVT_INTEROP_CROSSOVER",
            category="infrastructure",
            event_date="2024-01-01",
            source_name="(Add official source)",
            source_type="news",
            source_url="(ADD_URL)",
            original_text="Interoperable P2P digital transfers have surpassed ATM cash withdrawals.",
            confidence="low",
            notes="Replace (ADD_URL) and date with an official publication/press release."
        )
        event_code_to_id["EVT_INTEROP_CROSSOVER"] = evt[ID_COL]

    if "EVT_FAYDA_ROLLOUT" not in event_code_to_id:
        df, evt = add_event(
            df, cols,
            indicator="Fayda Digital ID rollout milestone",
            indicator_code="EVT_FAYDA_ROLLOUT",
            category="policy",
            event_date="2023-01-01",
            source_name="Fayda/NIDP",
            source_type="policy",
            source_url="https://www.id.gov.et/",
            original_text="Fayda digital ID program rollout (milestone date).",
            confidence="medium",
            notes="Replace with a verified milestone date/quote from id.gov.et once located."
        )
        event_code_to_id["EVT_FAYDA_ROLLOUT"] = evt[ID_COL]

    # Refresh lookup
    events = df[df["record_type"].eq("event")].copy()
    event_code_to_id = (
        events[[ID_COL, "indicator_code"]]
        .dropna(subset=["indicator_code"])
        .set_index("indicator_code")[ID_COL]
        .to_dict()
    )

    # 2) Add impact links if missing
    existing_links = df[df["record_type"].eq("impact_link")].copy()
    existing_link_keys = set(
        zip(
            existing_links.get("parent_id", pd.Series(dtype=str)).astype(str),
            existing_links.get("related_indicator", pd.Series(dtype=str)).astype(str),
        )
    )

    def maybe_add_link(
        parent_event_code: str,
        pillar: str,
        related_indicator: str,
        impact_direction: str,
        impact_magnitude: float,
        lag_months: int,
        evidence_basis: str,
    ) -> None:
        nonlocal df
        if parent_event_code not in event_code_to_id:
            return
        pid = event_code_to_id[parent_event_code]
        key = (str(pid), str(related_indicator))
        if key in existing_link_keys:
            return
        df, _ = add_impact_link(
            df, cols,
            parent_id=pid,
            pillar=pillar,
            related_indicator=related_indicator,
            impact_direction=impact_direction,
            impact_magnitude=impact_magnitude,
            lag_months=lag_months,
            evidence_basis=evidence_basis,
        )

    maybe_add_link(
        "EVT_TELEBIRR", "ACCESS", "ACC_MM_ACCOUNT",
        "positive", 2.0, 6,
        "Telebirr launch expected to raise mobile money account ownership after onboarding/agent rollout lag."
    )
    maybe_add_link(
        "EVT_TELEBIRR", "USAGE", "USG_DIGITAL_PAYMENT",
        "positive", 3.0, 6,
        "Mobile money launch expands payment capability; lag reflects time for adoption and ecosystem growth."
    )
    maybe_add_link(
        "EVT_SAFARICOM", "USAGE", "USG_MPESA_USERS",
        "positive", 1.0, 6,
        "New operator entry increases distribution and product competition; effects realized after rollout."
    )
    maybe_add_link(
        "EVT_INTEROP_CROSSOVER", "USAGE", "USG_DIGITAL_PAYMENT",
        "positive", 2.0, 0,
        "Interoperability milestone indicates mainstreaming of digital P2P vs cash withdrawals."
    )
    maybe_add_link(
        "EVT_FAYDA_ROLLOUT", "ACCESS", "ACC_OWNERSHIP",
        "positive", 1.0, 12,
        "Digital ID reduces KYC barriers; expected gradual effect on formal account ownership."
    )

    df.to_csv(OUT_PATH, index=False)

    # Print summary (so you always see something)
    print(f"Wrote: {OUT_PATH}")
    print(df["record_type"].value_counts(dropna=False).to_string())

    return 0

if __name__ == "__main__":
    raise SystemExit(main())