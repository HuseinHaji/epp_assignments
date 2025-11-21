from pathlib import Path
import pandas as pd
import numpy as np

pd.options.mode.copy_on_write = True


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def _to_pandas_missing(s: pd.Series) -> pd.Series:
    """
    Replace numeric missing codes (e.g. -7) with pandas NA.
    """
    return s.replace(-7, pd.NA)


def _harmonize_bpi_category(s: pd.Series) -> pd.Series:
    """
    Convert BPI answers to an ordered categorical:
    not true < sometimes true < often true.
    Works with strings like 'NOT TRUE', 'SOMETIMES TRUE', 'OFTEN TRUE'.
    """
    s = _to_pandas_missing(s)

    def _normalize_value(v):
        if pd.isna(v):
            return pd.NA
        v = str(v).strip().upper()
        if v == "NOT TRUE":
            return "not true"
        if v == "SOMETIMES TRUE":
            return "sometimes true"
        if v == "OFTEN TRUE":
            return "often true"
        return pd.NA

    s = s.map(_normalize_value)

    cat_type = pd.CategoricalDtype(
        categories=["not true", "sometimes true", "often true"],
        ordered=True,
    )
    return s.astype(cat_type)


def _categorical_to_binary(s: pd.Series) -> pd.Series:
    """
    Map ordered BPI categories to 0/1 for scoring.
    not true -> 0
    sometimes true / often true -> 1
    """
    mapping = {
        "not true": 0.0,
        "sometimes true": 1.0,
        "often true": 1.0,
    }
    return s.map(mapping)


def _get_subscale(readable_name: str) -> str:
    """
    Infer subscale name from readable_name prefix.
    Example: 'anxiety_mood' -> 'anxiety'.
    """
    return readable_name.split("_", 1)[0]


# =====================================================================
# MAIN FUNCTION FOR A SINGLE WAVE
# =====================================================================

def _clean_one_wave(raw_nlsy: pd.DataFrame, year: int, bpi_info: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the NLSY BPI data for a single survey year and return a DataFrame
    indexed by (childid, year).

    bpi_variable_info.csv must contain:
    - nlsy_name: raw variable name
    - readable_name: cleaned final name
    - survey_year: year or 'invariant'
    """
    df = raw_nlsy.copy()

    # --- 1. Split metadata ---
    id_rows = bpi_info[bpi_info["survey_year"] == "invariant"].copy()
    year_rows = bpi_info[bpi_info["survey_year"].astype(str) == str(year)].copy()

    id_raw = id_rows["nlsy_name"].tolist()
    id_clean = id_rows["readable_name"].tolist()

    item_raw = year_rows["nlsy_name"].tolist()
    item_clean = year_rows["readable_name"].tolist()

    # --- 2. Add year column ---
    df["year"] = year

    # --- 3. Subset to necessary columns ---
    cols_needed = id_raw + item_raw
    cols_needed = [c for c in cols_needed if c in df.columns]
    df = df[cols_needed + ["year"]]

    # --- 4. Rename columns ---
    rename_map = dict(zip(id_raw, id_clean)) | dict(zip(item_raw, item_clean))
    df = df.rename(columns=rename_map)

    # --- 5. Clean BPI items ---
    item_cols_cat = []
    for col in item_clean:
        if col not in df.columns:
            continue
        df[col] = _harmonize_bpi_category(df[col])
        item_cols_cat.append(col)

    # --- 6. Numeric scoring (0/1) ---
    df_num = pd.DataFrame(index=df.index)
    for col in item_cols_cat:
        df_num[col] = _categorical_to_binary(df[col])

    # --- 7. Build subscale scores ---
    df_scores = pd.DataFrame(index=df.index)
    item_to_subscale = {col: _get_subscale(col) for col in item_cols_cat}
    subscales = sorted(set(item_to_subscale.values()))

    for subscale in subscales:
        sub_items = [c for c, sc in item_to_subscale.items() if sc == subscale]
        if not sub_items:
            continue
        df_scores[f"bpi_{subscale}"] = df_num[sub_items].mean(axis=1)

    # --- 8. Correct dtypes + index ---
    if "childid" not in df.columns:
        raise ValueError("childid missing (check bpi_variable_info.csv).")

    df["childid"] = df["childid"].astype("Int64")
    df["year"] = df["year"].astype("Int64")

    if "momid" in df.columns:
        df["momid"] = df["momid"].astype("Int64")
    if "birth_order" in df.columns:
        df["birth_order"] = df["birth_order"].astype("Int64")

    out = df.join(df_scores)
    out = out.set_index(["childid", "year"])

    return out


# =====================================================================
# TASK 5: MANAGE ALL WAVES
# =====================================================================

def manage_nlsy_data(raw_nlsy: pd.DataFrame, bpi_info: pd.DataFrame) -> pd.DataFrame:
    """
    Clean all NLSY BPI waves (1986–2010 even years) and return long dataset.
    """
    waves = []
    for year in range(1986, 2012, 2):
        wave_df = _clean_one_wave(raw_nlsy, year, bpi_info)
        waves.append(wave_df)

    df_long = pd.concat(waves).sort_index()
    return df_long


# =====================================================================
# MAIN EXECUTION
# =====================================================================

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    bld_dir = project_root / "bld"

    # Load data extracted by unzip.py
    nlsy_path = bld_dir / "BEHAVIOR_PROBLEMS_INDEX.dta"
    info_path = bld_dir / "bpi_variable_info.csv"

    raw_nlsy = pd.read_stata(nlsy_path)
    bpi_info = pd.read_csv(info_path)

    clean_nlsy = manage_nlsy_data(raw_nlsy, bpi_info)

    out_path = bld_dir / "nlsy_clean.parquet"
    clean_nlsy.to_parquet(out_path)

    print("NLSY cleaning complete.")
