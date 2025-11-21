from pathlib import Path
import pandas as pd
import plotly.express as px

pd.options.mode.copy_on_write = True


def make_score_plot(df: pd.DataFrame, score: str, out_path: Path) -> None:
    """
    Create a facet grid scatter plot comparing NLSY bpi_* vs CHS bpi_*_chs
    for each age (5 subplots).
    """

    x_col = f"bpi_{score}"         # NLSY score
    y_col = f"bpi_{score}_chs"     # CHS score

    if x_col not in df.columns:
        raise ValueError(f"{x_col} missing in dataset.")

    if y_col not in df.columns:
        raise ValueError(f"{y_col} missing in dataset.")

    # Plot
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        facet_col="age",
        facet_col_wrap=5,
        trendline="ols",
        title=f"NLSY vs CHS {score.capitalize()} Score by Age",
        labels={x_col: f"NLSY {score}", y_col: f"CHS {score}"},
    )

    # Save PNG or HTML
    if out_path.suffix == ".png":
        fig.write_image(out_path)
    else:
        fig.write_html(out_path)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    bld_dir = project_root / "bld"

    merged_path = bld_dir / "merged_chs_nlsy.parquet"
    df = pd.read_parquet(merged_path)

    scores = ["antisocial", "anxiety", "headstrong", "hyperactive", "peer"]

    for score in scores:
        out_file = bld_dir / f"plot_{score}.png"
        make_score_plot(df, score, out_file)

    print("All plots generated.")
