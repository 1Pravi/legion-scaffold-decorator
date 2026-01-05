import pandas as pd
import pytest
import logging
from scaffold_decorator.generator import generate_decorated_molecules
from scaffold_decorator.generator import generate_molecules
from scaffold_decorator.utils.io import load_smiles_from_csv
from scaffold_decorator.utils.io import save_results_chunk


@pytest.fixture
def logger():
    """Create a logger for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("TEST")
    return logger


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


# Integration tests


def test_full_workflow(sample_scaffolds_csv, sample_decorations_csv, temp_csv_dir):
    """Test a complete workflow from loading to saving.

    This is an integration test that verifies the entire workflow
    works together: loading CSV files, generating molecules, and saving results.
    """
    # Load data
    scaffolds = load_smiles_from_csv(sample_scaffolds_csv)
    decorations = load_smiles_from_csv(sample_decorations_csv)

    # Generate molecules
    results = generate_decorated_molecules(
        scaffolds=scaffolds,
        left_decorations=decorations,
        right_decorations=decorations,
        decorations_sample_size=2,
    )

    # Should have results
    assert len(results) > 0

    # Save results
    output_path = temp_csv_dir / "workflow_results.csv"
    save_results_chunk(results, str(output_path), 0)

    # Verify saved file
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert len(df) == len(results)


def test_generate_molecules_basic(
    sample_scaffolds_csv, sample_decorations_csv, temp_csv_dir, logger
):
    """Test generate_molecules function with valid inputs."""
    output_dir = temp_csv_dir / "output"

    generate_molecules(
        scaffolds_csv=sample_scaffolds_csv,
        left_decorations_csv=sample_decorations_csv,
        right_decorations_csv=sample_decorations_csv,
        output_dir=str(output_dir),
        decorations_sample_size=2,
        save_every=2,
        logger=logger,
    )

    # Check that output files were created
    assert output_dir.exists()
    csv_files = list(output_dir.glob("*.csv"))
    assert len(csv_files) > 0

    # Verify content of output files
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        assert "smiles" in df.columns
        assert "scaffold" in df.columns
        assert "left_decoration" in df.columns
        assert "right_decoration" in df.columns


def test_generate_molecules_with_invalid_sample_size(
    sample_scaffolds_csv, sample_decorations_csv, temp_csv_dir, logger
):
    """Test main function with invalid sample size."""
    with pytest.raises(ValueError, match="sample_size must be positive"):
        generate_molecules(
            scaffolds_csv=sample_scaffolds_csv,
            left_decorations_csv=sample_decorations_csv,
            right_decorations_csv=sample_decorations_csv,
            output_dir=str(temp_csv_dir),
            decorations_sample_size=0,
            save_every=10,
            logger=logger,
        )


def test_generate_molecules_with_invalid_save_every(
    sample_scaffolds_csv, sample_decorations_csv, temp_csv_dir, logger
):
    """Test main function with invalid save_every."""
    with pytest.raises(ValueError, match="save_every must be positive"):
        generate_molecules(
            scaffolds_csv=sample_scaffolds_csv,
            left_decorations_csv=sample_decorations_csv,
            right_decorations_csv=sample_decorations_csv,
            output_dir=str(temp_csv_dir),
            decorations_sample_size=2,
            save_every=-1,
            logger=logger,
        )


def test_generate_molecules_with_nonexistent_file(
    sample_decorations_csv, temp_csv_dir, logger
):
    """Test main function with nonexistent file."""
    with pytest.raises(FileNotFoundError):
        generate_molecules(
            scaffolds_csv="nonexistent.csv",
            left_decorations_csv=sample_decorations_csv,
            right_decorations_csv=sample_decorations_csv,
            output_dir=str(temp_csv_dir),
            decorations_sample_size=2,
            save_every=10,
            logger=logger,
        )


def test_generate_molecules_with_empty_scaffolds(
    empty_csv, sample_decorations_csv, temp_csv_dir, logger
):
    """Test main function with empty scaffolds file."""
    with pytest.raises(ValueError, match="No scaffolds found"):
        generate_molecules(
            scaffolds_csv=empty_csv,
            left_decorations_csv=sample_decorations_csv,
            right_decorations_csv=sample_decorations_csv,
            output_dir=str(temp_csv_dir),
            decorations_sample_size=2,
            save_every=10,
            logger=logger,
        )


def test_generate_molecules_saves_final_chunk(
    sample_scaffolds_csv, sample_decorations_csv, temp_csv_dir, logger
):
    """Test that main saves the final chunk even if not at save_every interval."""
    output_dir = temp_csv_dir / "output"

    # Use save_every larger than number of scaffolds
    generate_molecules(
        scaffolds_csv=sample_scaffolds_csv,
        left_decorations_csv=sample_decorations_csv,
        right_decorations_csv=sample_decorations_csv,
        output_dir=str(output_dir),
        decorations_sample_size=1,
        save_every=100,  # Larger than number of scaffolds
        logger=logger,
    )

    # Should still create at least one output file
    csv_files = list(output_dir.glob("*.csv"))
    assert len(csv_files) >= 1


def test_generate_molecules_with_multiple_chunks(
    sample_scaffolds_csv, sample_decorations_csv, temp_csv_dir, logger
):
    """Test that main creates multiple chunks when save_every is small."""
    output_dir = temp_csv_dir / "output"

    # Use small save_every to force multiple chunks
    generate_molecules(
        scaffolds_csv=sample_scaffolds_csv,
        left_decorations_csv=sample_decorations_csv,
        right_decorations_csv=sample_decorations_csv,
        output_dir=str(output_dir),
        decorations_sample_size=1,
        save_every=1,  # Save after each scaffold
        logger=logger,
    )

    # Should create multiple chunks
    csv_files = list(output_dir.glob("*.csv"))
    assert len(csv_files) >= 2
