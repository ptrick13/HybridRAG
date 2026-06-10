"""Main data loading orchestrator for Software Development Analytics.

Loads generated sample data into all three data stores (PostgreSQL, Neo4j, Qdrant).
Run ``python -m data.generate_sample`` first to create the JSON files.

Usage:
    # Load all stores
    python -m data.load_data

    # Load only specific stores
    python -m data.load_data --stores postgres neo4j

    # Skip confirmation prompt
    python -m data.load_data --yes
"""

import argparse
import asyncio
import logging
from pathlib import Path

from data import ingest_neo4j, ingest_postgres, ingest_qdrant

logger = logging.getLogger(__name__)

_SAMPLE_DIR = Path(__file__).parent / "sample"


async def load_sample(stores: list[str]) -> None:
    """Load sample data into all specified stores.

    Args:
        stores: Which stores to populate: ``["postgres", "neo4j", "qdrant"]``.
    """
    logger.info("Loading sample dataset from %s", _SAMPLE_DIR)

    if "postgres" in stores:
        logger.info("--- PostgreSQL ---")
        ingest_postgres.ingest_from_files(_SAMPLE_DIR)

    if "neo4j" in stores:
        logger.info("--- Neo4j ---")
        ingest_neo4j.ingest_from_files(_SAMPLE_DIR)

    if "qdrant" in stores:
        logger.info("--- Qdrant ---")
        await ingest_qdrant.ingest_from_files(_SAMPLE_DIR)

    logger.info("Sample data loading complete.")


def _confirm(message: str, skip: bool) -> bool:
    if skip:
        return True
    response = input(f"{message} [y/N] ").strip().lower()
    return response == "y"


def main() -> None:
    """CLI entry point for the data loading orchestrator."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=(
            "Load Software Development Analytics sample data into "
            "PostgreSQL, Neo4j, and Qdrant."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stores",
        nargs="+",
        choices=["postgres", "neo4j", "qdrant"],
        default=["postgres", "neo4j", "qdrant"],
        help="Which data stores to populate (default: all three).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompts.",
    )
    args = parser.parse_args()

    if not _confirm(
        f"This will load sample data into: {', '.join(args.stores)}. Continue?",
        skip=args.yes,
    ):
        logger.info("Aborted.")
        return

    asyncio.run(load_sample(args.stores))


if __name__ == "__main__":
    main()
