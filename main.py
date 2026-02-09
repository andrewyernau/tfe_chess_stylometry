#!/usr/bin/env python3
"""Command-line interface for Chess Stylometry utilities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from labs.src.embedding_exporter import EmbeddingExporter


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return ivalue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chess Stylometry CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed_parser = subparsers.add_parser(
        "generate-embeddings",
        help="Convert board + heatmap PNGs into vector embeddings",
    )
    embed_parser.add_argument(
        "--event-dir",
        type=Path,
        required=True,
        help="Event directory produced by pipeline_stylometry_blocks.py",
    )
    embed_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory for embeddings (default: <event-dir>/embeddings)",
    )
    embed_parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device identifier (e.g., cpu, cuda, cuda:0)",
    )
    embed_parser.add_argument(
        "--image-size",
        type=_positive_int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(224, 224),
        help="Resize target applied before inference (pixels)",
    )
    embed_parser.add_argument(
        "--players",
        nargs="*",
        help="Optional list of player names to include (use quotes when names contain spaces)",
    )
    embed_parser.add_argument(
        "--limit-per-player",
        type=_positive_int,
        help="Maximum number of games to embed per player",
    )
    embed_parser.add_argument(
        "--weights",
        choices=["imagenet", "none"],
        default="imagenet",
        help="Backbone initialization mode",
    )

    return parser


def run_generate_embeddings(args: argparse.Namespace) -> None:
    exporter = EmbeddingExporter(
        event_dir=args.event_dir,
        output_dir=args.output_dir,
        image_size=(args.image_size[0], args.image_size[1]),
        device=args.device,
        weights=None if args.weights == "none" else args.weights,
        player_filter=args.players,
        limit_per_player=args.limit_per_player,
    )
    manifest_path = exporter.run()
    print(f"Embedding manifest saved at: {manifest_path}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate-embeddings":
        run_generate_embeddings(args)
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
