"""CLI for generating random decorated molecules."""

import argparse
import logging
import os
import time

from scaffold_decorator.generator import generate_random_molecules
from scaffold_decorator.utils.io import load_smiles_from_csv
from scaffold_decorator.utils.io import save_results_chunk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for random molecule generation CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate N random molecules by combining scaffolds with decorations"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--scaffolds-csv",
        type=str,
        default="input/unique_scaffolds.csv",
        help="Path to CSV file containing scaffolds",
    )
    parser.add_argument(
        "--left-decorations-csv",
        type=str,
        default="input/unique_left_decorations.csv",
        help="Path to CSV file containing left decorations",
    )
    parser.add_argument(
        "--right-decorations-csv",
        type=str,
        default="input/unique_right_decorations.csv",
        help="Path to CSV file containing right decorations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/random_molecules.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "-n",
        "--n-molecules",
        type=int,
        required=True,
        help="Number of random molecules to generate",
    )
    parser.add_argument(
        "--no-randomize-scaffolds",
        action="store_true",
        help="Don't randomize scaffold selection (use systematically)",
    )
    parser.add_argument(
        "--no-randomize-decorations",
        action="store_true",
        help="Don't randomize decoration selection (use systematically)",
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="uniform",
        choices=["uniform", "thompson", "ucb"],
        help="Sampling strategy: 'uniform', 'thompson', or 'ucb' (bandit algorithms)",
    )
    parser.add_argument(
        "--usage-penalty",
        type=float,
        default=0.0,
        help="Penalty factor for repeated item usage (to promote diversity)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Load data
    logger.info("Loading data...")
    scaffolds = load_smiles_from_csv(args.scaffolds_csv)
    left_decorations = load_smiles_from_csv(args.left_decorations_csv)
    right_decorations = load_smiles_from_csv(args.right_decorations_csv)

    # Validate
    if not scaffolds:
        raise ValueError(f"No scaffolds found in {args.scaffolds_csv}")
    if not left_decorations:
        raise ValueError(f"No decorations found in {args.left_decorations_csv}")
    if not right_decorations:
        raise ValueError(f"No decorations found in {args.right_decorations_csv}")

    logger.info(f"Loaded {len(scaffolds)} scaffolds")
    logger.info(f"Loaded {len(left_decorations)} left decorations")
    logger.info(f"Loaded {len(right_decorations)} right decorations")

    # Generate random molecules
    logger.info(f"Generating {args.n_molecules} random molecules using {args.sampling} sampling...")
    start_time = time.time()
    results = generate_random_molecules(
        scaffolds=scaffolds,
        left_decorations=left_decorations,
        right_decorations=right_decorations,
        n_molecules=args.n_molecules,
        randomize_scaffolds=not args.no_randomize_scaffolds,
        randomize_decorations=not args.no_randomize_decorations,
        strategy=args.sampling,
        usage_penalty=args.usage_penalty,
        seed=args.seed,
    )
    end_time = time.time()
    duration = end_time - start_time
    throughput = len(results) / duration if duration > 0 else 0

    # Save results
    logger.info(f"Saving results to {args.output}...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_results_chunk(results, args.output, 0)

    # Log statistics
    successful = sum(1 for res in results if res[2] is not None)
    failed = len(results) - successful
    logger.info(
        f"Generation complete: {successful:,} successful, "
        f"{failed:,} failed out of {len(results):,} total in {duration:.2f}s"
    )
    logger.info(f"Success rate: {successful / len(results) * 100:.2f}%")
    logger.info(f"Throughput: {throughput:.2f} molecules/sec")


if __name__ == "__main__":
    main()
