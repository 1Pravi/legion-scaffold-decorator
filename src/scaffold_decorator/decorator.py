"""Module for scaffold decoration using RDKit.

This module provides functions to join molecular scaffolds with decorations
and construct complete molecules from scaffolded SMILES strings.
"""

from typing import List
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Mol

ATTACHMENT_POINT_TOKEN = "*"  # nosec B105 - not a password


def _add_attachment_point_num(atom: Chem.Atom, idx: str) -> None:
    """Add attachment point numbers to atoms.

    Args:
        atom: RDKit Atom object to add attachment point to.
        idx: Attachment point index as a string.

    Note:
        This function modifies the atom in place by setting the
        'molAtomMapNumber' property.
    """
    idxs = []
    if atom.HasProp("molAtomMapNumber"):
        idxs = atom.GetProp("molAtomMapNumber").split(",")
    idxs.append(str(idx))
    idxs = sorted(list(set(idxs)))
    atom.SetProp("molAtomMapNumber", ",".join(idxs))


def join(
    scaffold_smi: str, decoration_smi: str, keep_label_on_atoms: bool = False
) -> Optional[Mol]:
    """Join a SMILES scaffold with a decoration.

    Both the scaffold and decoration must contain labeled attachment points
    using the atom map number property. The function finds matching attachment
    points and connects them, removing the attachment point markers.

    Args:
        scaffold_smi: SMILES string of the scaffold with labeled attachment points.
        decoration_smi: SMILES string of the decoration with labeled attachment points.
        keep_label_on_atoms: If True, preserve labels on atoms after attachment.
            Useful for debugging but may cause issues. Defaults to False.

    Returns:
        RDKit Mol object of the joined scaffold, or None if the operation fails.

    Note:
        The function expects exactly one attachment point in the decoration that
        matches an attachment point in the scaffold. Both attachment points must
        have degree 1 (single connection).
    """
    scaffold = Chem.MolFromSmiles(scaffold_smi)
    decoration = Chem.MolFromSmiles(decoration_smi)

    if not scaffold or not decoration:
        return None

    # Find the attachment point in the decoration
    try:
        attachment_points = [
            atom.GetProp("molAtomMapNumber")
            for atom in decoration.GetAtoms()
            if atom.GetSymbol() == ATTACHMENT_POINT_TOKEN
        ]
        if len(attachment_points) != 1:
            return None  # Must have exactly one attachment point
        attachment_point_id = attachment_points[0]
    except KeyError:
        return None

    # Combine molecules and find matching attachment points
    combined_mol = Chem.RWMol(Chem.CombineMols(decoration, scaffold))
    attachment_atoms = [
        atom
        for atom in combined_mol.GetAtoms()
        if (
            atom.GetSymbol() == ATTACHMENT_POINT_TOKEN
            and atom.HasProp("molAtomMapNumber")
            and atom.GetProp("molAtomMapNumber") == attachment_point_id
        )
    ]

    if len(attachment_atoms) != 2:
        return None  # Must have exactly two matching attachment points

    # Get neighbors of attachment points and validate
    neighbors = []
    for attachment_atom in attachment_atoms:
        if attachment_atom.GetDegree() != 1:
            return None  # Attachment points must have exactly one bond
        neighbors.append(attachment_atom.GetNeighbors()[0])

    # Determine bond type from existing attachment bonds
    attachment_bonds = [atom.GetBonds()[0] for atom in attachment_atoms]
    bond_type = Chem.BondType.SINGLE
    if any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in attachment_bonds):
        bond_type = Chem.BondType.DOUBLE

    # Create new bond between neighbors and remove attachment points
    combined_mol.AddBond(neighbors[0].GetIdx(), neighbors[1].GetIdx(), bond_type)
    # Remove atoms in reverse order to avoid index shifting issues
    idx1, idx2 = attachment_atoms[0].GetIdx(), attachment_atoms[1].GetIdx()
    combined_mol.RemoveAtom(max(idx1, idx2))
    combined_mol.RemoveAtom(min(idx1, idx2))

    # Optionally keep labels for debugging
    if keep_label_on_atoms:
        for neighbor in neighbors:
            _add_attachment_point_num(neighbor, attachment_point_id)

    # Sanitize the final molecule
    result_mol = combined_mol.GetMol()
    try:
        Chem.SanitizeMol(result_mol)
    except ValueError:
        return None  # Sanitization failed

    return result_mol


def decorate_scaffold(
    scaffold_smi: str, decorations: List[str], n_attempts: int = 3
) -> Optional[str]:
    """Decorate a scaffold with multiple decorations.

    This function takes a scaffold SMILES string with labeled attachment points
    and attaches multiple decorations to it. The decorations are applied in order
    based on the sorted attachment point labels in the scaffold.

    Args:
        scaffold_smi: SMILES string of the scaffold with labeled attachment points.
        decorations: List of SMILES strings for decorations to attach. The order
            should match the sorted attachment point labels.
        n_attempts: Maximum number of attempts to construct the molecule if
            a decoration fails to attach. Defaults to 3.

    Returns:
        SMILES string of the fully decorated molecule, or None if construction fails.

    Note:
        The function will retry the entire decoration process up to n_attempts times
        if any single decoration fails to attach. Each retry starts from the original
        scaffold.
    """
    mol = Chem.MolFromSmiles(scaffold_smi)
    if not mol:
        return None

    # Extract and sort attachment point labels from scaffold
    labels = []
    for atom in mol.GetAtoms():
        smarts = atom.GetSmarts()
        if ATTACHMENT_POINT_TOKEN in smarts:
            labels.append(smarts)

    if not labels:
        return None  # No attachment points found

    labels.sort()

    # Attempt to construct the molecule
    for attempt in range(n_attempts):
        current_smi = scaffold_smi
        construction_failed = False

        for i, label in enumerate(labels):
            if i >= len(decorations):
                return None  # Not enough decorations provided

            decorated_mol = join(current_smi, decorations[i])
            if decorated_mol:
                current_smi = Chem.MolToSmiles(decorated_mol)
            else:
                construction_failed = True
                break  # Try again from scratch

        if not construction_failed:
            return current_smi

    return None  # All attempts failed
