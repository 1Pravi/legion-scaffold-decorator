"""I/O utilities for loading and saving molecular data."""

import os
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd


def load_smiles_from_csv(
    csv_path: str,
    column_name: str = "smiles",
) -> List[str]:
    """Load SMILES strings from a CSV file.

    Args:
        csv_path: Path to the CSV file.
        column_name: Name of the column containing SMILES strings.
                     Defaults to "smiles".
    Returns:
        List of SMILES strings from the CSV file.

    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
        KeyError: If the specified column doesn't exist in the CSV.
        pd.errors.EmptyDataError: If the CSV file is empty.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if column_name not in df.columns:
        raise KeyError(
            f"Column '{column_name}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    smiles_list = df[column_name].dropna().tolist()

    return smiles_list


def save_results_chunk(
    results: List[Tuple[str, Tuple[str, str], Optional[str]]],
    output_path: str,
    start_idx: int,
) -> None:
    """Save a chunk of results to a CSV file.

    Args:
        results: List of tuples containing (scaffold, decorations, result_smiles).
        output_path: Path where the CSV file should be saved.
        start_idx: Starting index in the results list for this chunk.

    Raises:
        IOError: If the file cannot be written.
    """
    chunk_results = results[start_idx:]

    df = pd.DataFrame(
        {
            "smiles": [res[2] for res in chunk_results],
            "scaffold": [res[0] for res in chunk_results],
            "left_decoration": [res[1][0] for res in chunk_results],
            "right_decoration": [res[1][1] for res in chunk_results],
        }
    )

    df.to_csv(output_path, index=False)
