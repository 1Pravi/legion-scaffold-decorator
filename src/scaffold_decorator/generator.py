"""Module for generating molecules by combining scaffolds with decorations.

This module provides functions to generate molecules by systematically or randomly
combining scaffolds with decorations.
"""

import logging
import os
from itertools import product
from random import choice
from random import sample
from random import shuffle
from typing import List
from typing import Optional
from typing import Tuple

from .adaptive_sampler import BanditSampler
from .decorator import decorate_scaffold
from .utils.io import load_smiles_from_csv
from .utils.io import save_results_chunk


def generate_decorated_molecules(
    scaffolds: List[str],
    left_decorations: List[str],
    right_decorations: List[str],
    decorations_sample_size: int,
) -> List[Tuple[str, Tuple[str, str], Optional[str]]]:
    """Generate decorated molecules for a list of scaffolds.

    Args:
        scaffolds: List of scaffold SMILES strings.
        left_decorations: List of left decoration SMILES strings.
        right_decorations: List of right decoration SMILES strings.
        decorations_sample_size: Number of decorations to sample for each position.
    Returns:
        List of tuples containing (scaffold, decorations, result_smiles).
        The result_smiles can be None if decoration fails.
    """
    results = []

    for scaffold in scaffolds:
        # Sample decorations for this scaffold
        sampled_left = sample(  # nosec B311 - not cryptographic use
            left_decorations, min(decorations_sample_size, len(left_decorations))
        )
        sampled_right = sample(  # nosec B311 - not cryptographic use
            right_decorations, min(decorations_sample_size, len(right_decorations))
        )

        # Generate all combinations
        for left_dec, right_dec in product(sampled_left, sampled_right):
            decorations = (left_dec, right_dec)
            result_smiles = decorate_scaffold(scaffold, list(decorations))
            results.append((scaffold, decorations, result_smiles))

    return results


def generate_random_molecules(
    scaffolds: List[str],
    left_decorations: List[str],
    right_decorations: List[str],
    n_molecules: int,
    randomize_scaffolds: bool = True,
    randomize_decorations: bool = True,
    strategy: str = "uniform",
    usage_penalty: float = 0.0,
    seed: Optional[int] = None,
) -> List[Tuple[str, Tuple[str, str], Optional[str]]]:
    """Generate N random decorated molecules using adaptive sampling.

    Args:
        scaffolds: List of scaffold SMILES strings.
        left_decorations: List of left decoration SMILES strings.
        right_decorations: List of right decoration SMILES strings.
        n_molecules: Number of random molecules to generate.
        randomize_scaffolds: If True, randomly sample scaffolds.
                             If False, use scaffolds in order (cycling if needed).
        randomize_decorations: If True, randomly sample decorations each time.
                               If False, use decorations systematically.
        strategy: Sampling strategy ('uniform', 'thompson', 'ucb').
        usage_penalty: Penalty factor for usage count to promote diversity.
        seed: Random seed for reproducibility.

    Returns:
        List of tuples containing (scaffold, decorations, result_smiles).
    """
    results = []

    # Initialize samplers
    # Note: seed is set once here if provided. BanditSampler uses global random state.
    if seed is not None:
        import random
        random.seed(seed)

    scaffold_sampler = BanditSampler(strategy, usage_penalty)
    left_sampler = BanditSampler(strategy, usage_penalty)
    right_sampler = BanditSampler(strategy, usage_penalty)

    for i in range(n_molecules):
        # Select scaffold
        if randomize_scaffolds:
            scaffold = scaffold_sampler.sample(scaffolds)
        else:
            scaffold = scaffolds[i % len(scaffolds)]

        # Select decorations
        if randomize_decorations:
            left_dec = left_sampler.sample(left_decorations)
            right_dec = right_sampler.sample(right_decorations)
        else:
            left_dec = left_decorations[i % len(left_decorations)]
            right_dec = right_decorations[i % len(right_decorations)]

        decorations = (left_dec, right_dec)
        result_smiles = decorate_scaffold(scaffold, list(decorations))

        # Feedback to samplers
        is_success = result_smiles is not None

        if randomize_scaffolds:
            scaffold_sampler.update(scaffold, is_success)

        if randomize_decorations:
            left_sampler.update(left_dec, is_success)
            right_sampler.update(right_dec, is_success)

        results.append((scaffold, decorations, result_smiles))

    return results


def generate_molecules(
    scaffolds_csv: str,
    left_decorations_csv: str,
    right_decorations_csv: str,
    output_dir: str,
    decorations_sample_size: int,
    save_every: int,
    logger: logging.Logger,
    shuffle_scaffolds: bool = True,
) -> None:
    """Generate molecules by systematically combining scaffolds with decorations.

    This function reads scaffolds and decorations from CSV files, samples
    decorations for each scaffold, generates all combinations, and saves
    the results in chunks.

    Args:
        scaffolds_csv: Path to CSV file containing scaffolds.
        left_decorations_csv: Path to CSV file containing left decorations.
        right_decorations_csv: Path to CSV file containing right decorations.
        output_dir: Directory to save output CSV chunks.
        decorations_sample_size: Number of decorations to sample for each position.
        save_every: Save results every N scaffolds.
        logger: Logger instance for logging.
        shuffle_scaffolds: If True, shuffle scaffolds before processing.

    Raises:
        FileNotFoundError: If any of the input CSV files don't exist.
        ValueError: If decorations_sample_size or save_every are invalid.
    """
    logger.info("Starting systematic molecule generation process")

    # Validate parameters
    if decorations_sample_size <= 0:
        raise ValueError(
            f"decorations_sample_size must be positive, got {decorations_sample_size}"
        )
    if save_every <= 0:
        raise ValueError(f"save_every must be positive, got {save_every}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load data
    try:
        scaffolds = load_smiles_from_csv(scaffolds_csv)
        left_decorations = load_smiles_from_csv(left_decorations_csv)
        right_decorations = load_smiles_from_csv(right_decorations_csv)
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        raise

    # Validate loaded data
    if not scaffolds:
        raise ValueError(f"No scaffolds found in {scaffolds_csv}")
    if not left_decorations:
        raise ValueError(f"No decorations found in {left_decorations_csv}")
    if not right_decorations:
        raise ValueError(f"No decorations found in {right_decorations_csv}")

    # Log statistics
    logger.info(f"Loaded {len(scaffolds)} scaffolds")
    logger.info(f"Loaded {len(left_decorations)} left decorations")
    logger.info(f"Loaded {len(right_decorations)} right decorations")

    actual_sample_size = min(
        decorations_sample_size, len(left_decorations), len(right_decorations)
    )
    if actual_sample_size < decorations_sample_size:
        logger.warning(
            f"Sample size reduced to {actual_sample_size} due to "
            f"insufficient decorations"
        )

    estimated_total = len(scaffolds) * actual_sample_size**2
    logger.info(f"Estimated molecules to generate: {estimated_total:,}")

    # Shuffle scaffolds if requested
    if shuffle_scaffolds:
        shuffle(scaffolds)

    # Process scaffolds in batches
    all_results = []
    chunk_start_idx = 0
    chunk_counter = 0

    for i, scaffold in enumerate(scaffolds, 1):
        # Generate molecules for this scaffold
        scaffold_results = generate_decorated_molecules(
            scaffolds=[scaffold],
            left_decorations=left_decorations,
            right_decorations=right_decorations,
            decorations_sample_size=actual_sample_size,
        )
        all_results.extend(scaffold_results)

        # Save chunk if needed
        if i % save_every == 0 or i == len(scaffolds):
            logger.info(
                f"Progress: {i}/{len(scaffolds)} scaffolds "
                f"({i / len(scaffolds) * 100:.1f}%)"
            )

            filename = (
                f"chunk_{chunk_counter:04d}_"
                f"{chunk_start_idx}-{len(all_results)}.csv"
            )
            output_path = os.path.join(output_dir, filename)

            try:
                save_results_chunk(all_results, output_path, chunk_start_idx)
                chunk_start_idx = len(all_results)
                chunk_counter += 1
            except Exception as e:
                logger.error(f"Failed to save chunk: {e}")
                raise

    # Log final statistics
    successful = sum(1 for res in all_results if res[2] is not None)
    failed = len(all_results) - successful
    logger.info(
        f"Generation complete: {successful:,} successful, "
        f"{failed:,} failed out of {len(all_results):,} total"
    )
    logger.info(f"Success rate: {successful / len(all_results) * 100:.2f}%")
