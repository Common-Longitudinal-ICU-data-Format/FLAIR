"""ELF code and time resolution from config specs (vendored from CLIF_MEDS)."""


def resolve_code(code_spec, row: dict) -> str:
    """Build an ELF code string from a config code spec and a data row.

    code_spec is either:
      - a string (literal code, e.g. "MEDS_BIRTH")
      - a list like ["PATIENT", "sex", "col(sex_category)"]
        where "col(X)" means pull value from row[X], others are literals

    Returns: ELF code string joined with "//"
    """
    if isinstance(code_spec, str):
        return code_spec
    parts = []
    for part in code_spec:
        if isinstance(part, dict):
            # Dict element: {col(X): {raw_val: mapped_val, ...}}
            col_expr = next(iter(part))
            value_map = part[col_expr]
            col_name = col_expr[4:-1] if col_expr.startswith("col(") and col_expr.endswith(")") else col_expr
            val = row.get(col_name)
            raw = str(val) if val is not None else "UNK"
            parts.append(str(value_map.get(raw, raw)))
        elif isinstance(part, str) and part.startswith("col(") and part.endswith(")"):
            col_name = part[4:-1]
            val = row.get(col_name)
            parts.append(str(val) if val is not None else "UNK")
        else:
            parts.append(str(part))
    return "//".join(parts)


def resolve_time(time_spec, row: dict):
    """Extract raw time value from a row given the config spec.

    time_spec is either None → None, or "col(column_name)" → row[column_name].
    """
    if time_spec is None:
        return None
    if isinstance(time_spec, str) and time_spec.startswith("col("):
        col_name = time_spec[4:-1]
        return row.get(col_name)
    return None
