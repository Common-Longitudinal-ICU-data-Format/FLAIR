"""DataFrame transform helpers (vendored from CLIF_MEDS)."""
import polars as pl


def strip_tz(df: pl.DataFrame) -> pl.DataFrame:
    """Strip timezone info from all datetime columns, preserving local wall-clock time.

    clifpy's loader has already converted UTC → site-local once; dropping the tz
    here (``replace_time_zone(None)``) keeps that wall-clock and guarantees no
    second conversion ever happens downstream.
    """
    for col_name in df.columns:
        dtype = df[col_name].dtype
        if dtype == pl.Datetime or (hasattr(dtype, "time_zone") and dtype.time_zone is not None):
            df = df.with_columns(
                pl.col(col_name).dt.replace_time_zone(None).alias(col_name)
            )
    return df


def normalize_categories(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize all *_category columns: trim, lowercase, spaces/hyphens → underscores."""
    cat_cols = [c for c in df.columns if c.endswith("_category") and df[c].dtype == pl.Utf8]
    if not cat_cols:
        return df
    return df.with_columns(
        pl.col(c)
          .str.strip_chars()
          .str.to_lowercase()
          .str.replace_all(" ", "_")
          .str.replace_all("-", "_")
        for c in cat_cols
    )
