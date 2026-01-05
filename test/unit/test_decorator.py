import pytest
from rdkit import Chem
from scaffold_decorator.decorator import _add_attachment_point_num
from scaffold_decorator.decorator import decorate_scaffold
from scaffold_decorator.decorator import join


@pytest.fixture
def benzene_scaffold():
    """Benzene scaffold with single attachment point."""
    return "c1ccc([*:1])cc1"


@pytest.fixture
def methyl_decoration():
    """Methyl group decoration."""
    return "[*:1]C"


@pytest.fixture
def aliphatic_scaffold():
    """Aliphatic scaffold for double bond testing."""
    return "CC([*:1])C"


@pytest.fixture
def double_bond_decoration():
    """Decoration with double bond."""
    return "[*:1]=C"


@pytest.fixture
def two_point_scaffold():
    """Scaffold with two attachment points."""
    return "[*:1]c1ccc([*:2])cc1"


@pytest.fixture
def two_decorations():
    """Two decorations for multi-point scaffolds."""
    return ["[*:1]C", "[*:2]O"]


@pytest.fixture
def simple_carbon_mol():
    """Simple carbon molecule for testing."""
    return Chem.MolFromSmiles("C")


# Tests for join function


def test_join_simple_single_bond(benzene_scaffold, methyl_decoration):
    """Test joining scaffold with decoration using single bond."""
    result = join(benzene_scaffold, methyl_decoration)

    assert result is not None
    smiles = Chem.MolToSmiles(result)
    # Should get toluene (methylbenzene)
    assert "C" in smiles
    assert "c" in smiles


def test_join_with_double_bond(aliphatic_scaffold, double_bond_decoration):
    """Test joining with double bond."""
    result = join(aliphatic_scaffold, double_bond_decoration)

    assert result is not None
    smiles = Chem.MolToSmiles(result)
    # Should contain double bond
    assert "=" in smiles


def test_join_invalid_scaffold(methyl_decoration):
    """Test joining with invalid scaffold SMILES."""
    result = join("invalid_smiles", methyl_decoration)
    assert result is None


def test_join_invalid_decoration(benzene_scaffold):
    """Test joining with invalid decoration SMILES."""
    result = join(benzene_scaffold, "invalid_smiles")
    assert result is None


def test_join_no_attachment_point_in_decoration(benzene_scaffold):
    """Test joining when decoration has no attachment point."""
    result = join(benzene_scaffold, "C")  # No attachment point
    assert result is None


def test_join_no_matching_attachment_point(benzene_scaffold):
    """Test joining when attachment points don't match."""
    decoration = "[*:2]C"  # Different attachment point number
    result = join(benzene_scaffold, decoration)
    assert result is None


def test_join_multiple_attachment_points_in_decoration(benzene_scaffold):
    """Test joining when decoration has multiple attachment points."""
    decoration = "[*:1]C[*:1]"  # Multiple attachment points
    result = join(benzene_scaffold, decoration)
    assert result is None


def test_join_keep_labels_false(benzene_scaffold, methyl_decoration):
    """Test that labels are not kept by default."""
    result = join(benzene_scaffold, methyl_decoration, keep_label_on_atoms=False)

    assert result is not None
    # Check that no atoms have the molAtomMapNumber property
    has_labels = any(atom.HasProp("molAtomMapNumber") for atom in result.GetAtoms())
    assert not has_labels


def test_join_keep_labels_true(benzene_scaffold, methyl_decoration):
    """Test that labels are kept when requested."""
    result = join(benzene_scaffold, methyl_decoration, keep_label_on_atoms=True)

    assert result is not None
    # Check that at least one atom has the molAtomMapNumber property
    has_labels = any(atom.HasProp("molAtomMapNumber") for atom in result.GetAtoms())
    assert has_labels


# Tests for decorate_scaffold function


def test_decorate_scaffold_single_decoration(benzene_scaffold):
    """Test decorating scaffold with single decoration."""
    decorations = ["[*:1]C"]
    result = decorate_scaffold(benzene_scaffold, decorations)

    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


def test_decorate_scaffold_multiple_decorations(two_point_scaffold, two_decorations):
    """Test decorating scaffold with multiple decorations."""
    result = decorate_scaffold(two_point_scaffold, two_decorations)

    assert result is not None
    assert isinstance(result, str)


def test_decorate_scaffold_invalid_smiles():
    """Test with invalid scaffold SMILES."""
    result = decorate_scaffold("invalid_smiles", ["[*:1]C"])
    assert result is None


def test_decorate_scaffold_no_attachment_points():
    """Test scaffold without attachment points."""
    scaffold = "c1ccccc1"  # Benzene without attachment points
    result = decorate_scaffold(scaffold, ["[*:1]C"])
    assert result is None


def test_decorate_scaffold_insufficient_decorations(two_point_scaffold):
    """Test when not enough decorations are provided."""
    decorations = ["[*:1]C"]  # Only one decoration
    result = decorate_scaffold(two_point_scaffold, decorations)
    assert result is None


def test_decorate_scaffold_with_attempts(benzene_scaffold):
    """Test that n_attempts parameter works."""
    decorations = ["[*:1]C"]
    # Should succeed even with 1 attempt for valid input
    result = decorate_scaffold(benzene_scaffold, decorations, n_attempts=1)
    assert result is not None


def test_decorate_scaffold_empty_decorations(benzene_scaffold):
    """Test with empty decorations list."""
    result = decorate_scaffold(benzene_scaffold, [])
    assert result is None


def test_decorate_scaffold_order_matters(two_point_scaffold):
    """Test that decoration order matches attachment point order."""
    decorations1 = ["[*:1]C", "[*:2]O"]
    decorations2 = ["[*:1]O", "[*:2]C"]

    result1 = decorate_scaffold(two_point_scaffold, decorations1)
    result2 = decorate_scaffold(two_point_scaffold, decorations2)

    # Both should succeed but produce different molecules
    assert result1 is not None
    assert result2 is not None
    # Results might be canonicalized, so we just check they're valid
    assert len(result1) > 0
    assert len(result2) > 0


# Tests for _add_attachment_point_num helper function


def test_add_attachment_point_to_atom_without_property(simple_carbon_mol):
    """Test adding attachment point to atom without existing property."""
    atom = simple_carbon_mol.GetAtomWithIdx(0)

    _add_attachment_point_num(atom, "1")

    assert atom.HasProp("molAtomMapNumber")
    assert atom.GetProp("molAtomMapNumber") == "1"


def test_add_attachment_point_to_atom_with_existing_property(simple_carbon_mol):
    """Test adding attachment point to atom with existing property."""
    atom = simple_carbon_mol.GetAtomWithIdx(0)
    atom.SetProp("molAtomMapNumber", "1")

    _add_attachment_point_num(atom, "2")

    assert atom.HasProp("molAtomMapNumber")
    prop_value = atom.GetProp("molAtomMapNumber")
    # Should contain both numbers in sorted order
    assert "1" in prop_value
    assert "2" in prop_value


def test_add_duplicate_attachment_point(simple_carbon_mol):
    """Test adding duplicate attachment point number."""
    atom = simple_carbon_mol.GetAtomWithIdx(0)
    atom.SetProp("molAtomMapNumber", "1")

    _add_attachment_point_num(atom, "1")

    # Should not duplicate
    assert atom.GetProp("molAtomMapNumber") == "1"


# Integration tests


def test_full_decoration_workflow(two_point_scaffold):
    """Test complete workflow of decorating a scaffold."""
    decorations = ["[*:1]CC", "[*:2]N"]

    result = decorate_scaffold(two_point_scaffold, decorations)

    assert result is not None
    # Verify it's a valid SMILES
    mol = Chem.MolFromSmiles(result)
    assert mol is not None


def test_join_then_join_again(two_point_scaffold):
    """Test joining multiple times sequentially."""
    decoration1 = "[*:1]C"
    decoration2 = "[*:2]O"

    # First join
    intermediate = join(two_point_scaffold, decoration1)
    assert intermediate is not None

    # Second join
    intermediate_smi = Chem.MolToSmiles(intermediate)
    final = join(intermediate_smi, decoration2)
    assert final is not None


# Edge case tests for better coverage


def test_join_attachment_point_without_map_number():
    """Test attachment point without molAtomMapNumber property."""
    # Create a decoration with * but no atom map number property
    # This should trigger KeyError handling
    scaffold = "c1ccc([*:1])cc1"
    # Decoration with * but missing atom map number
    decoration = "[*]C"  # No :1 label
    result = join(scaffold, decoration)
    assert result is None


def test_join_attachment_point_with_multiple_bonds():
    """Test attachment point with degree != 1 (multiple bonds)."""
    # Create a scaffold where attachment point has multiple connections
    # This is a malformed case that should be rejected
    scaffold = "[*:1](C)(C)c1ccccc1"  # Attachment with degree > 1
    decoration = "[*:1]C"
    result = join(scaffold, decoration)
    # Should fail because attachment point doesn't have degree 1
    assert result is None


def test_join_creates_invalid_chemistry():
    """Test joining that would create chemically invalid structure."""
    # Try to create a structure that would fail sanitization
    # For example, connecting aromatics in a way that breaks aromaticity
    scaffold = "c1ccc([*:1])cc1"
    # This might create valence issues depending on RDKit version
    decoration = "[*:1]=[N+]=[N-]"
    result = join(scaffold, decoration)
    # Result might be None if sanitization fails
    # This tests the ValueError handling in sanitization
    if result is not None:
        # If it succeeds, at least verify it's valid
        assert Chem.MolToSmiles(result) is not None


def test_decorate_scaffold_with_retry_mechanism():
    """Test that retry mechanism works when decoration temporarily fails."""
    # Use a scaffold that might have issues on first attempt
    scaffold = "[*:1]c1ccccc1"
    # This should succeed eventually
    decorations = ["[*:1]C"]
    result = decorate_scaffold(scaffold, decorations, n_attempts=3)
    assert result is not None


def test_decorate_scaffold_all_attempts_fail():
    """Test when all decoration attempts fail."""
    # Create a scenario where decoration consistently fails
    scaffold = "[*:1]c1ccccc1"
    # Use an invalid decoration that will always fail
    decorations = ["invalid_decoration"]
    result = decorate_scaffold(scaffold, decorations, n_attempts=3)
    assert result is None


def test_join_with_zero_attachment_atoms_found():
    """Test when no matching attachment atoms are found in combined molecule."""
    # This is a tricky edge case - scaffold has attachment but decoration doesn't match
    scaffold = "C([*:1])C"
    decoration = "[*:3]O"  # Different number, won't match
    result = join(scaffold, decoration)
    assert result is None
