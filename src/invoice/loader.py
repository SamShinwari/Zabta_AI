from pathlib import Path

import pandas as pd


def load_invoice(file_path: str | Path) -> pd.DataFrame:
    """
    Load an invoice CSV file.

    Parameters
    ----------
    file_path:
        Path to the invoice CSV.

    Returns
    -------
    pandas.DataFrame
        Loaded invoice data.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Invoice file not found: {file_path}"
        )

    if file_path.suffix.lower() != ".csv":
        raise ValueError(
            "Currently Zabta supports CSV invoice files only."
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(
            "Invoice file is empty."
        )

    return df