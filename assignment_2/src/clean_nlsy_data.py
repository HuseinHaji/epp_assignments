from pathlib import Path
import pandas as pd
import numpy as np

pd.options.mode.copy_on_write = True


def _to_pandas_missing(s: pd.Series) -> pd.Series:
    """
    Convert NLSY-style missing codes to pandas missing.
    """
    # Example: negative values are missings. Adjust to your data!
    return s.where(~(s < 0), other=pd.NA)


def _harmonize_bpi_category(s: pd.Series) -> pd.Series:
    """
    Convert BPI answers to an ordered categorical: not true < sometimes true < often true.
    """
    # Depending on the raw values (codes or strings), you adapt this mapping:
    mapping = {
        0: "not true",
        1: "sometimes true",
        2: "often true",
        "NOT TRUE": "not true",
        "SOMETIMES TRUE": "sometimes true",
        "OFTEN TRUE": "often true",
    }

    s = s.map(mapping).astype("string")
    cat_type = pd.CategoricalDtype(
        categories=["not true", "sometimes true", "often true"], ordered=True
    )
    return s.astype(cat_type)


def _categorical_to_binary(s: pd.Series) -> pd.Series:
    """
    Map ordered BPI categories to 0/1 for scoring.
    """
    mapping = {
        "not true": 0.0,
        "sometimes true": 1.0,
        "often true": 1.0,
    }
    return s.map(mapping)


def _clean_one_wave(
    raw_nlsy: pd.DataFrame, year: int, bpi_info: pd.DataFrame
) -> pd.DataFrame:
    """
    Clean the NLSY BPI data for a single wave (year) and return a DataFrame
    indexed by (childid, year).
    """
    # Filter metadata for this wave
    info_year = bpi_info.query("year == @year").copy()

    # Suppose bpi_info has columns: raw_name, clean_name, subscale
    # You must check the actual CSV columns!
    raw_names = info_year["raw_name"].tolist()
    clean_names = info_year["clean_name"].tolist()
    subscales = info_year["subscale"].tolist()

    df = raw_nlsy.copy()

    # Ensure year column exists in this wide NLSY dataset or create it if needed
    df["year"] = year

    # Keep ID variables + BPI items
    id_cols = ["childid", "momid", "year"]  # adjust to actual columns
    df = df[id_cols + raw_names]

    # Clean each item variable
    item_cols_cat = []
    for raw_name, clean_name in zip(raw_names, clean_names):
        s = df[raw_name]
        s = _to_pandas_missing(s)
        s = _harmonize_bpi_category(s)
        df[clean_name] = s
        item_cols_cat.append(clean_name)

    # Compute numeric version for scoring
    df_num = pd.DataFrame(index=df.index)
    for col in item_cols_cat:
        df_num[col] = _categorical_to_binary(df[col])

    # Compute subscale scores by averaging
    df_scores = pd.DataFrame(index=df.index)
    info_year = info_year.set_index("clean_name")
    for clean_name in item_cols_cat:
        subscale = info_year.loc[clean_name, "subscale"]  # e.g. "antisocial"
        score_col = f"bpi_{subscale}"
        if score_col not in df_scores:
            df_scores[score_col] = df_num[clean_name]
        else:
            df_scores[score_col] = df_scores[score_col].add(df_num[clean_name], fill_value=0)

    # Divide each subscale by number of items
    for subscale in info_year["subscale"].unique():
        score_col = f"bpi_{subscale}"
        n_items = (info_year["subscale"] == subscale).sum()
        df_scores[score_col] = df_scores[score_col] / n_items

    # Combine ID + categorical item vars + scores
    out = df[id_cols + item_cols_cat].join(df_scores)

    # Set index to (childid, year)
    out["childid"] = out["childid"].astype("Int64")
    out["year"] = out["year"].astype("Int64")
    out = out.set_index(["childid", "year"])

    return out
