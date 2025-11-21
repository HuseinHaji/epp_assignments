from pathlib import Path
import zipfile
import pandas as pd  # not strictly needed here, but often in the project

# modern pandas settings (example)
pd.options.mode.copy_on_write = True


def unzip_original_data(zip_path: Path, target_dir: Path) -> None:
    """
    Unzip the original data into the target directory.
    """
    # No side effects on inputs (we don't modify zip_path / target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    zip_path = project_root / "src" / "original_data" / "original_data.zip"
    bld_dir = project_root / "bld"

    unzip_original_data(zip_path, bld_dir)
