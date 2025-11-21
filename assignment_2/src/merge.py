from pathlib import Path
import pandas as pd

pd.options.mode.copy_on_write = True


def merge_chs_nlsy(chs: pd.DataFrame, nlsy: pd.DataFrame) -> pd.DataFrame:
    """
    Merge cleaned CHS and NLSY data and restrict to ages 5–13.
    """
    # Ensure indices are aligned
    if not isinstance(chs.index, pd.MultiIndex):
        raise ValueError("chs must have a MultiIndex (childid, year)")
    if not isinstance(nlsy.index, pd.MultiIndex):
        raise ValueError("nlsy must have a MultiIndex (childid, year)")

    # Avoid overlapping column names
    chs_cols = set(chs.columns)
    nlsy_cols = set(nlsy.columns)
    overlap = chs_cols & nlsy_cols
    if overlap:
        chs = chs.rename(columns={c: f"{c}_chs" for c in overlap})
        nlsy = nlsy.rename(columns={c: f"{c}_nlsy" for c in overlap})

    merged = chs.join(nlsy, how="inner")

    # Age might be in CHS or NLSY – adjust accordingly
    if "age" in merged.columns:
        age_col = "age"
    elif "age_chs" in merged.columns:
        age_col = "age_chs"
    else:
        raise ValueError("No age column found in merged data")

    merged = merged.loc[(merged[age_col] >= 5) & (merged[age_col] <= 13)]

    return merged


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    bld_dir = project_root / "bld"

    chs_path = bld_dir / "chs_clean.parquet"
    nlsy_path = bld_dir / "nlsy_clean.parquet"

    chs = pd.read_parquet(chs_path)
    nlsy = pd.read_parquet(nlsy_path)

    merged = merge_chs_nlsy(chs, nlsy)

    out_path = bld_dir / "merged_chs_nlsy.parquet"
    merged.to_parquet(out_path)
