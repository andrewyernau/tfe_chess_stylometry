#!/usr/bin/env python3
"""Utilities to export chess stylometry embeddings from board and heatmap images."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


BLOCK_PATTERN = re.compile(r"game_(?P<game>[\w-]+)_block_(?P<block>\d+)\.png", re.IGNORECASE)
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)


@dataclass
class GameSample:
    """File paths that belong to a single (player, game) pair."""

    player: str
    game_id: str
    board_paths: List[Path]
    heat_paths: List[Path]


def _sanitize_player_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def _index_blocked_images(image_dir: Path) -> Dict[str, Dict[str, Path]]:
    """Builds a nested mapping: game_id -> block_id -> path."""

    index: Dict[str, Dict[str, Path]] = defaultdict(dict)
    for file_path in sorted(image_dir.glob("game_*_block_*.png")):
        match = BLOCK_PATTERN.match(file_path.name)
        if not match:
            continue
        game_id = match.group("game")
        block_id = match.group("block")
        index[game_id][block_id] = file_path
    return index


def discover_samples(
    event_dir: Path,
    player_filter: Optional[Iterable[str]] = None,
    limit_per_player: Optional[int] = None,
) -> Iterator[GameSample]:
    """Yield every valid (board, heat) pair discovered inside an event directory."""

    board_root = event_dir / "board_images"
    heat_root = event_dir / "heatmap_images"

    if not board_root.exists() or not heat_root.exists():
        raise FileNotFoundError(
            f"Missing board or heatmap directories inside {event_dir}. "
            "Run the image pipeline before exporting embeddings."
        )

    allowed_players = None
    if player_filter:
        allowed_players = {_sanitize_player_name(name) for name in player_filter}

    for player_dir in sorted(p for p in board_root.iterdir() if p.is_dir()):
        if allowed_players and player_dir.name not in allowed_players:
            continue

        heat_dir = heat_root / player_dir.name
        if not heat_dir.exists():
            print(f"[warning] Skipping {player_dir.name}: heatmap directory not found.")
            continue

        board_index = _index_blocked_images(player_dir)
        heat_index = _index_blocked_images(heat_dir)
        shared_games = sorted(set(board_index.keys()) & set(heat_index.keys()))

        produced = 0
        for game_id in shared_games:
            blocks = sorted(set(board_index[game_id].keys()) & set(heat_index[game_id].keys()))
            if not blocks:
                continue

            board_paths = [board_index[game_id][block] for block in blocks]
            heat_paths = [heat_index[game_id][block] for block in blocks]

            yield GameSample(
                player=player_dir.name,
                game_id=game_id,
                board_paths=board_paths,
                heat_paths=heat_paths,
            )
            produced += 1

            if limit_per_player and produced >= limit_per_player:
                break


class ResNetFeatureExtractor:
    """Thin wrapper around ResNet18 to obtain L2-normalized embeddings."""

    def __init__(
        self,
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        device: str = "cpu",
        weights: Optional[str] = "imagenet",
    ) -> None:
        self.image_size = image_size
        requested_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        if device != "cpu" and requested_device.type == "cpu":
            print("[warning] CUDA requested but not available. Falling back to CPU.")
        self.device = requested_device
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.model = self._build_model(weights)

    def _build_model(self, weights: Optional[str]):
        weight_enum = None
        if weights and weights.lower() == "imagenet":
            try:
                weight_enum = ResNet18_Weights.IMAGENET1K_V1
            except Exception as exc:  # pragma: no cover - fallback for offline setups
                print(f"[warning] Could not load ImageNet weights ({exc}). Using random init.")
        model = resnet18(weights=weight_enum)
        model.fc = torch.nn.Identity()
        model.eval()
        return model.to(self.device)

    def _prepare_tensor(self, image_path: Path) -> torch.Tensor:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).permute(2, 0, 1)
        tensor = (tensor - self.mean) / self.std
        return tensor.unsqueeze(0).to(self.device)

    def embed_image(self, image_path: Path) -> torch.Tensor:
        with torch.no_grad():
            batch = self._prepare_tensor(image_path)
            features = self.model(batch)
        return features.squeeze(0).cpu()


class EmbeddingExporter:
    """Exports per-game embeddings together with per-player centroids."""

    def __init__(
        self,
        event_dir: Path,
        output_dir: Optional[Path] = None,
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        device: str = "cpu",
        weights: Optional[str] = "imagenet",
        player_filter: Optional[Sequence[str]] = None,
        limit_per_player: Optional[int] = None,
    ) -> None:
        self.event_dir = Path(event_dir)
        if not self.event_dir.exists():
            raise FileNotFoundError(f"Event directory not found: {self.event_dir}")
        self.output_dir = Path(output_dir) if output_dir else self.event_dir / "embeddings"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = ResNetFeatureExtractor(image_size=image_size, device=device, weights=weights)
        self.player_filter = list(player_filter) if player_filter else None
        self.limit_per_player = limit_per_player

    def run(self) -> Path:
        """Generate embeddings and return the manifest path."""

        player_vectors: Dict[str, List[np.ndarray]] = defaultdict(list)
        sample_counts: Dict[str, int] = defaultdict(int)
        total_samples = 0

        for sample in discover_samples(
            self.event_dir,
            player_filter=self.player_filter,
            limit_per_player=self.limit_per_player,
        ):
            vector = self._embed_sample(sample)
            self._save_vector(sample, vector)
            player_vectors[sample.player].append(vector)
            sample_counts[sample.player] += 1
            total_samples += 1

            if total_samples % 25 == 0:
                print(f"Processed {total_samples} samples so far...")

        if total_samples == 0:
            raise RuntimeError("No samples found. Verify that board and heatmap images exist.")

        manifest_path = self._write_manifest(player_vectors, sample_counts)
        print(f"Embeddings saved to {self.output_dir}")
        print(f"Manifest written to {manifest_path}")
        return manifest_path

    def _embed_sample(self, sample: GameSample) -> np.ndarray:
        block_vectors: List[torch.Tensor] = []
        for board_path, heat_path in zip(sample.board_paths, sample.heat_paths):
            board_vec = self.extractor.embed_image(board_path)
            heat_vec = self.extractor.embed_image(heat_path)
            block_vectors.append(torch.cat([board_vec, heat_vec], dim=0))

        stacked = torch.stack(block_vectors)
        pooled = stacked.mean(dim=0)
        normalized = F.normalize(pooled.unsqueeze(0), p=2, dim=1).squeeze(0)
        return normalized.numpy().astype(np.float32)

    def _save_vector(self, sample: GameSample, vector: np.ndarray) -> Path:
        player_dir = self.output_dir / sample.player
        player_dir.mkdir(parents=True, exist_ok=True)
        sample_path = player_dir / f"{sample.game_id}.npy"
        np.save(sample_path, vector)
        return sample_path

    def _write_manifest(
        self,
        player_vectors: Dict[str, List[np.ndarray]],
        sample_counts: Dict[str, int],
    ) -> Path:
        manifest = {
            "event": self.event_dir.name,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "players": {},
            "summary": {
                "player_count": len(player_vectors),
                "sample_count": int(sum(sample_counts.values())),
            },
        }

        for player, vectors in sorted(player_vectors.items()):
            stacked = np.stack(vectors, axis=0)
            centroid = stacked.mean(axis=0).astype(np.float32)
            centroid_path = self.output_dir / player / "centroid.npy"
            np.save(centroid_path, centroid)
            manifest["players"][player] = {
                "samples": int(stacked.shape[0]),
                "embedding_dim": int(stacked.shape[1]),
                "vector_dir": str((self.output_dir / player).resolve()),
                "centroid_path": str(centroid_path.resolve()),
            }

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as handler:
            json.dump(manifest, handler, indent=2)
        return manifest_path


__all__ = [
    "EmbeddingExporter",
    "GameSample",
    "ResNetFeatureExtractor",
    "discover_samples",
]
