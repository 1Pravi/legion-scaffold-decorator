"""CLI for systematic molecule generation."""

import argparse
import logging
import os

from scaffold_decorator.generator import generate_molecules

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for systematic molecule generation CLI."""
    parser = argparse.ArgumentParser(
        description="Generate molecules by combining scaffolds with decorations",
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
        "--output-dir",
        type=str,
        default="output/",
        help="Directory to save output CSV chunks",
    )
    parser.add_argument(
        "--decorations-sample-size",
        type=int,
        default=64,
        help="Number of decorations to sample for each position",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save results every N scaffolds",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle scaffolds before processing",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate molecules
    generate_molecules(
        scaffolds_csv=args.scaffolds_csv,
        left_decorations_csv=args.left_decorations_csv,
        right_decorations_csv=args.right_decorations_csv,
        output_dir=args.output_dir,
        decorations_sample_size=args.decorations_sample_size,
        save_every=args.save_every,
        shuffle_scaffolds=not args.no_shuffle,
        logger=logger,
    )


if __name__ == "__main__":
    main()
