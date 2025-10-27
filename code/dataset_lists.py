# pip install ucimlrepo
from ucimlrepo import fetch_ucirepo

# ---- configuration of type keywords (heuristics, based on UCI metadata) ----
REAL_KEYS       = {"real", "continuous", "float", "numeric", "double", "decimal"}
INTEGER_KEYS    = {"integer", "int", "discrete", "count"}
CATEG_KEYS      = {"categorical", "nominal", "ordinal", "binary", "bool", "boolean", "string", "text", "char"}

def _normalise_types(strings):
    """Lower-case, strip, and split composite labels; return a list of tokens."""
    out = []
    for s in strings:
        if not s:
            continue
        s = str(s).lower().strip()
        # Split on common separators
        parts = [p.strip() for p in s.replace("/", ",").replace(";", ",").split(",") if p.strip()]
        out.extend(parts if parts else [s])
    return out or []

def _is_classification(meta):
    tasks = [t.lower() for t in (meta.task or [])]
    return any("classification" in t for t in tasks)

def _feature_type_strings(ds):
    """
    Prefer the per-variable schema (ds.variables) for rows where role == 'Feature'.
    Fallback to metadata.feature_types if variables are missing.
    Returns a list of type strings for features.
    """
    vars_df = getattr(ds, "variables", None)
    if vars_df is not None and {"role", "type"}.issubset(vars_df.columns):
        feats = vars_df[vars_df["role"].astype(str).str.lower() == "feature"]
        if not feats.empty:
            return [str(t) for t in feats["type"].tolist()]
    # Fallback (coarser): dataset-level feature types
    meta_types = getattr(ds.metadata, "feature_types", None) or []
    return [str(t) for t in meta_types]

def _bucket_feature_types(type_strings):
    """
    Decide whether the dataset has any real/continuous, integer/discrete, and/or categorical features.
    Returns (has_real, has_integer, has_categ).
    """
    types = _normalise_types(type_strings)
    has_real    = any(any(k in t for k in REAL_KEYS)    for t in types)
    has_integer = any(any(k in t for k in INTEGER_KEYS) for t in types)
    has_categ   = any(any(k in t for k in CATEG_KEYS)   for t in types)
    return has_real, has_integer, has_categ

# ---- main enumeration over UCI dataset IDs ----
all_classification = []
only_continuous    = []  # ONLY real/continuous; no integer, no categorical
only_categorical   = []  # ONLY categorical; no real, no integer
both_cont_and_cat  = []  # at least one categorical AND at least one numeric (real OR integer)

for ds_id in range(1, 900):  # covers current catalogue; unused IDs are skipped
    try:
        ds = fetch_ucirepo(id=ds_id)
    except Exception:
        continue  # non-existent ID

    meta = ds.metadata
    name = (meta.name or f"dataset_{ds_id}").strip()

    if not _is_classification(meta):
        continue

    tstrings = _feature_type_strings(ds)
    if not tstrings:  # no schema; skip
        continue

    has_real, has_integer, has_categ = _bucket_feature_types(tstrings)

    # Keep the name in the master list
    all_classification.append(name)

    # Bucket membership
    if has_categ and (has_real or has_integer):
        both_cont_and_cat.append(name)
    elif has_categ and not has_real and not has_integer:
        only_categorical.append(name)
    elif has_real and not has_integer and not has_categ:
        only_continuous.append(name)
    # (Else: numeric but not "continuous-only" → e.g., pure integer datasets; not requested.)

# Optional: sort for reproducible order
all_classification = sorted(set(all_classification))
only_continuous    = sorted(set(only_continuous))
only_categorical   = sorted(set(only_categorical))
both_cont_and_cat  = sorted(set(both_cont_and_cat))

# Peek at counts (you can remove these prints if you just need the lists)
print("All classification:", len(all_classification))
print("Only continuous  :", len(only_continuous))
print("Only categorical :", len(only_categorical))
print("Both cont+cat    :", len(both_cont_and_cat))

# The four lists you asked for are now populated:
#   all_classification
#   only_continuous
#   only_categorical
#   both_cont_and_cat
