import pandas as pd
import pytest
import logging
from scaffold_decorator.generator import generate_decorated_molecules
from scaffold_decorator.generator import generate_random_molecules
from scaffold_decorator.utils.io import load_smiles_from_csv
from scaffold_decorator.utils.io import save_results_chunk


@pytest.fixture
def logger():
    """Create a logger for testing."""
    return logging.getLogger(__name__)


@pytest.fixture
def temp_csv_dir(tmp_path):
    """Create a temporary directory for CSV files."""
    return tmp_path


@pytest.fixture
def sample_scaffolds_csv(temp_csv_dir):
    """Create a sample scaffolds CSV file."""
    csv_path = temp_csv_dir / "scaffolds.csv"
    df = pd.DataFrame({"smiles": ["c1ccc([*:1])cc1", "[*:1]c1ccccc1", "CC([*:1])C"]})
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_decorations_csv(temp_csv_dir):
    """Create a sample decorations CSV file."""
    csv_path = temp_csv_dir / "decorations.csv"
    df = pd.DataFrame({"smiles": ["[*:1]C", "[*:1]O", "[*:1]N", "[*:1]CC"]})
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def empty_csv(temp_csv_dir):
    """Create an empty CSV file."""
    csv_path = temp_csv_dir / "empty.csv"
    df = pd.DataFrame({"smiles": []})
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def csv_wrong_column(temp_csv_dir):
    """Create a CSV file with wrong column name."""
    csv_path = temp_csv_dir / "wrong_column.csv"
    df = pd.DataFrame({"wrong_name": ["c1ccccc1"]})
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_results():
    """Create sample results for testing."""
    return [
        ("scaffold1", ("[*:1]C", "[*:2]O"), "Cc1ccccc1O"),
        ("scaffold2", ("[*:1]N", "[*:2]C"), "Nc1ccccc1C"),
        ("scaffold3", ("[*:1]O", "[*:2]N"), None),
    ]


# Tests for load_smiles_from_csv


def test_load_smiles_from_csv_success(sample_scaffolds_csv):
    """Test loading SMILES from a valid CSV file."""
    smiles = load_smiles_from_csv(sample_scaffolds_csv)
    assert len(smiles) == 3
    assert "c1ccc([*:1])cc1" in smiles


def test_load_smiles_from_csv_with_na_values(temp_csv_dir):
    """Test loading CSV with NA values."""
    csv_path = temp_csv_dir / "with_na.csv"
    df = pd.DataFrame({"smiles": ["c1ccccc1", None, "CC"]})
    df.to_csv(csv_path, index=False)

    smiles = load_smiles_from_csv(str(csv_path))
    assert len(smiles) == 2
    assert None not in smiles


def test_load_smiles_from_csv_file_not_found():
    """Test loading from non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_smiles_from_csv("nonexistent_file.csv")


def test_load_smiles_from_csv_wrong_column(csv_wrong_column):
    """Test loading CSV with wrong column name."""
    with pytest.raises(KeyError):
        load_smiles_from_csv(csv_wrong_column)


def test_load_smiles_from_csv_custom_column(temp_csv_dir):
    """Test loading CSV with custom column name."""
    csv_path = temp_csv_dir / "custom_column.csv"
    df = pd.DataFrame({"my_smiles": ["c1ccccc1", "CC"]})
    df.to_csv(csv_path, index=False)

    smiles = load_smiles_from_csv(str(csv_path), column_name="my_smiles")
    assert len(smiles) == 2


# Tests for save_results_chunk


def test_save_results_chunk_success(sample_results, temp_csv_dir):
    """Test saving results chunk to CSV."""
    output_path = temp_csv_dir / "results.csv"
    save_results_chunk(sample_results, str(output_path), 0)

    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert len(df) == 3
    assert "smiles" in df.columns
    assert "scaffold" in df.columns
    assert "left_decoration" in df.columns
    assert "right_decoration" in df.columns


def test_save_results_chunk_partial(sample_results, temp_csv_dir):
    """Test saving partial chunk starting from an index."""
    output_path = temp_csv_dir / "partial_results.csv"
    save_results_chunk(sample_results, str(output_path), 1)

    df = pd.read_csv(output_path)
    assert len(df) == 2  # Only last 2 results


def test_save_results_chunk_with_none_values(sample_results, temp_csv_dir):
    """Test saving results that contain None values."""
    output_path = temp_csv_dir / "with_none.csv"
    save_results_chunk(sample_results, str(output_path), 0)

    df = pd.read_csv(output_path)
    # Check that None values are preserved as NaN
    assert df.iloc[2]["smiles"] != df.iloc[2]["smiles"]  # NaN != NaN


# Tests for generate_decorated_molecules


def test_generate_decorated_molecules_basic():
    """Test basic molecule generation."""
    scaffolds = ["c1ccc([*:1])cc1"]
    left_decorations = ["[*:1]C"]
    right_decorations = ["[*:1]O"]
    sample_size = 1

    results = generate_decorated_molecules(
        scaffolds, left_decorations, right_decorations, sample_size
    )

    assert len(results) == 1
    assert results[0][0] == "c1ccc([*:1])cc1"
    assert len(results[0][1]) == 2  # Two decorations


def test_generate_decorated_molecules_multiple_scaffolds():
    """Test generation with multiple scaffolds."""
    scaffolds = ["c1ccc([*:1])cc1", "[*:1]c1ccccc1"]
    left_decorations = ["[*:1]C", "[*:1]O"]
    right_decorations = ["[*:1]N"]
    sample_size = 2

    results = generate_decorated_molecules(
        scaffolds, left_decorations, right_decorations, sample_size
    )

    # 2 scaffolds * 2 left * 1 right = 4 results
    assert len(results) == 4


def test_generate_decorated_molecules_sample_size_larger_than_available():
    """Test when sample size is larger than available decorations."""
    scaffolds = ["c1ccc([*:1])cc1"]
    left_decorations = ["[*:1]C"]
    right_decorations = ["[*:1]O"]
    sample_size = 10  # Much larger than available

    results = generate_decorated_molecules(
        scaffolds, left_decorations, right_decorations, sample_size
    )

    # Should only generate 1 * 1 = 1 result
    assert len(results) == 1


def test_generate_decorated_molecules_with_invalid_scaffold():
    """Test generation with invalid scaffold."""
    scaffolds = ["invalid_smiles"]
    left_decorations = ["[*:1]C"]
    right_decorations = ["[*:1]O"]
    sample_size = 1

    results = generate_decorated_molecules(
        scaffolds, left_decorations, right_decorations, sample_size
    )

    # Should have result but with None SMILES
    assert len(results) == 1
    assert results[0][2] is None


# Tests for generate_random_molecules


def test_generate_random_molecules_basic():
    """Test basic random molecule generation."""
    scaffolds = ["c1ccc([*:1])cc1"]
    left_decorations = ["[*:1]C", "[*:1]O"]
    right_decorations = ["[*:1]N", "[*:1]CC"]
    n_molecules = 5

    results = generate_random_molecules(
        scaffolds, left_decorations, right_decorations, n_molecules
    )

    assert len(results) == 5
    # Each result should have correct structure
    for scaffold, decorations, result_smiles in results:
        assert scaffold in scaffolds
        assert decorations[0] in left_decorations
        assert decorations[1] in right_decorations


def test_generate_random_molecules_with_randomization():
    """Test random generation with full randomization."""
    scaffolds = ["c1ccc([*:1])cc1", "[*:1]c1ccccc1"]
    left_decorations = ["[*:1]C", "[*:1]O"]
    right_decorations = ["[*:1]N"]
    n_molecules = 10

    results = generate_random_molecules(
        scaffolds=scaffolds,
        left_decorations=left_decorations,
        right_decorations=right_decorations,
        n_molecules=n_molecules,
        randomize_scaffolds=True,
        randomize_decorations=True,
    )

    assert len(results) == 10


def test_generate_random_molecules_no_randomization():
    """Test systematic generation without randomization."""
    scaffolds = ["c1ccc([*:1])cc1", "[*:1]c1ccccc1"]
    left_decorations = ["[*:1]C", "[*:1]O"]
    right_decorations = ["[*:1]N"]
    n_molecules = 4

    results = generate_random_molecules(
        scaffolds=scaffolds,
        left_decorations=left_decorations,
        right_decorations=right_decorations,
        n_molecules=n_molecules,
        randomize_scaffolds=False,
        randomize_decorations=False,
    )

    assert len(results) == 4
    # Should cycle through systematically
    assert results[0][0] == scaffolds[0]
    assert results[1][0] == scaffolds[1]
    assert results[2][0] == scaffolds[0]  # Cycles back
    assert results[3][0] == scaffolds[1]


def test_generate_random_molecules_large_n():
    """Test generating more molecules than scaffolds."""
    scaffolds = ["c1ccc([*:1])cc1"]
    left_decorations = ["[*:1]C"]
    right_decorations = ["[*:1]O"]
    n_molecules = 100

    results = generate_random_molecules(
        scaffolds, left_decorations, right_decorations, n_molecules
    )

    assert len(results) == 100
