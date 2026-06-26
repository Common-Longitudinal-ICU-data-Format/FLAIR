"""Code metadata generation and codes.parquet output (vendored from CLIF_MEDS).

Adds ``build_codes_table`` so the codes registry can be derived from an in-memory
events frame (the FE-meds path writes a single MEDS.parquet rather than per-domain
files, so the original glob-based scanner is supplemented).
"""
from pathlib import Path

import polars as pl

DOMAIN_LABELS = {
    "VITAL": "Vital sign", "LAB": "Laboratory", "MED_CON": "Continuous medication",
    "MED_INT": "Intermittent medication", "RESP": "Respiratory support",
    "PA": "Patient assessment", "CODE_STATUS": "Code status",
    "HOSP": "Hospitalization", "PATIENT": "Patient", "ADT": "ADT",
    "POS": "Position", "CRRT": "CRRT", "ECMO_MCS": "ECMO/MCS",
    "PROC": "Procedure", "HOSP_DX": "Hospital diagnosis",
}

CODES_SCHEMA = {
    "code": pl.Utf8,
    "description": pl.Utf8,
    "parent_codes": pl.List(pl.Utf8),
    "concept_version": pl.Utf8,
    "event_count": pl.Int64,
    "is_numeric_value": pl.Int8,
    "is_text_value": pl.Int8,
}


def generate_description(code: str) -> str:
    """Auto-generate a human-readable description from an ELF code string."""
    if code == "MEDS_BIRTH":
        return "Patient birth event"
    if code == "MEDS_DEATH":
        return "Patient death event"

    parts = code.split("//")
    domain = parts[0]
    label = DOMAIN_LABELS.get(domain, domain)
    concept = parts[1].replace("_", " ") if len(parts) > 1 else ""
    qualifiers = [p for p in parts[2:] if p not in ("NA", "UNK")]

    desc = f"{label}: {concept}"
    if qualifiers:
        desc += f" ({', '.join(qualifiers)})"
    return desc


def code_to_domain(code: str):
    """Extract the domain prefix from an ELF code for version lookup."""
    if code.startswith("MEDS_"):
        return "PATIENT"
    parts = code.split("//")
    return parts[0] if parts[0] in DOMAIN_LABELS else None


def build_codes_table(events: pl.DataFrame, domain_versions: dict[str, str]) -> pl.DataFrame:
    """Build the codes registry from an in-memory MEDS events frame.

    Mirrors ``write_codes_parquet`` semantics: event counts < 10 are floored to 10
    (PHI protection), and numeric/text presence flags are derived per code.
    """
    if events.height == 0:
        return pl.DataFrame(schema=CODES_SCHEMA)

    code_stats = (
        events.lazy()
        .select("code", "numeric_value", "text_value")
        .filter(pl.col("code").is_not_null() & (pl.col("code") != ""))
        .group_by("code")
        .agg(
            pl.len().alias("len"),
            pl.col("numeric_value").is_not_null().any().alias("has_numeric"),
            pl.col("text_value").is_not_null().any().alias("has_text"),
        )
        .collect()
        .with_columns(
            pl.when(pl.col("len") < 10).then(10).otherwise(pl.col("len")).cast(pl.Int64).alias("event_count"),
            pl.col("has_numeric").cast(pl.Int8).alias("is_numeric_value"),
            pl.col("has_text").cast(pl.Int8).alias("is_text_value"),
        )
        .drop("len", "has_numeric", "has_text")
        .sort("code")
    )

    rows = []
    for s in code_stats.iter_rows(named=True):
        domain = code_to_domain(s["code"])
        rows.append({
            "code": s["code"],
            "description": generate_description(s["code"]),
            "parent_codes": None,
            "concept_version": domain_versions.get(domain, "0.0.0") if domain else "0.0.0",
            "event_count": s["event_count"],
            "is_numeric_value": s["is_numeric_value"],
            "is_text_value": s["is_text_value"],
        })
    return pl.DataFrame(rows, schema=CODES_SCHEMA)


def write_codes_parquet(output_dir: Path, domain_versions: dict[str, str]):
    """Scan output parquet files under data/ and write metadata/codes.parquet."""
    data_dir = output_dir / "data"
    frames = [pl.read_parquet(p) for p in data_dir.glob("*.parquet")]
    events = pl.concat(frames, how="diagonal") if frames else pl.DataFrame(
        schema={"code": pl.Utf8, "numeric_value": pl.Float32, "text_value": pl.Utf8})
    codes_df = build_codes_table(events, domain_versions)
    out_path = output_dir / "metadata" / "codes.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    codes_df.write_parquet(out_path)
    print(f"  codes.parquet -> {out_path} ({len(codes_df)} codes)")
    return codes_df
