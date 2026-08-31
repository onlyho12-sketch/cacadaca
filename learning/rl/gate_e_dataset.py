"""Isaac-free Gate E dataset schema and deterministic seed splitting."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np

from learning.rl.env.gate_d_observation import (
    BASE14,
    GLOBAL20,
    SPATIAL120,
    observation_dim,
)


SPLIT_NAMES = ("train", "validation", "test")
SPLIT_TO_ID = {name: index for index, name in enumerate(SPLIT_NAMES)}


def split_for_seed(profile_seed: int) -> str:
    """Stable 70/15/15 assignment shared across profiles and paths."""
    digest = hashlib.sha256(f"gate-e-seed:{int(profile_seed)}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def split_id_for_seed(profile_seed: int) -> int:
    return SPLIT_TO_ID[split_for_seed(profile_seed)]


@dataclass(frozen=True)
class GateEShard:
    observations120: np.ndarray
    actions: np.ndarray
    profile_seeds: np.ndarray
    sequence_ids: np.ndarray
    split_ids: np.ndarray

    def validate(self) -> None:
        n = len(self.observations120)
        expected = {
            "observations120": (n, observation_dim(SPATIAL120)),
            "actions": (n, 2),
            "profile_seeds": (n,),
            "sequence_ids": (n,),
            "split_ids": (n,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} shape={value.shape}, expected={shape}")
        if n == 0:
            raise ValueError("Gate E shard cannot be empty")
        if not np.isfinite(self.observations120).all():
            raise ValueError("observations contain NaN/Inf")
        if not np.isfinite(self.actions).all():
            raise ValueError("actions contain NaN/Inf")
        if np.any(np.abs(self.actions) > 1.000001):
            raise ValueError("actions exceed [-1, 1]")
        if not set(np.unique(self.split_ids)).issubset(set(range(len(SPLIT_NAMES)))):
            raise ValueError("invalid split id")
        expected_splits = np.asarray(
            [split_id_for_seed(seed) for seed in self.profile_seeds], dtype=np.uint8)
        if not np.array_equal(self.split_ids, expected_splits):
            raise ValueError("split ids do not match deterministic profile-seed split")

    def observations(self, mode: str) -> np.ndarray:
        dim = observation_dim(mode)
        if mode not in (BASE14, GLOBAL20, SPATIAL120):
            raise ValueError(f"unknown observation mode {mode!r}")
        return self.observations120[:, :dim]


def save_shard(path: str, shard: GateEShard, metadata: dict) -> None:
    shard.validate()
    if os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite dataset shard: {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        observations120=shard.observations120.astype(np.float32, copy=False),
        actions=shard.actions.astype(np.float32, copy=False),
        profile_seeds=shard.profile_seeds.astype(np.int64, copy=False),
        sequence_ids=shard.sequence_ids.astype(np.int32, copy=False),
        split_ids=shard.split_ids.astype(np.uint8, copy=False),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def load_shard(path: str) -> tuple[GateEShard, dict]:
    with np.load(path, allow_pickle=False) as data:
        shard = GateEShard(
            observations120=np.asarray(data["observations120"], dtype=np.float32),
            actions=np.asarray(data["actions"], dtype=np.float32),
            profile_seeds=np.asarray(data["profile_seeds"], dtype=np.int64),
            sequence_ids=np.asarray(data["sequence_ids"], dtype=np.int32),
            split_ids=np.asarray(data["split_ids"], dtype=np.uint8),
        )
        metadata = json.loads(str(data["metadata_json"].item()))
    shard.validate()
    return shard, metadata


def combine_shards(paths: list[str]) -> tuple[GateEShard, list[dict]]:
    if not paths:
        raise ValueError("at least one shard is required")
    loaded = [load_shard(path) for path in paths]
    shards = [item[0] for item in loaded]
    combined = GateEShard(
        observations120=np.concatenate([item.observations120 for item in shards]),
        actions=np.concatenate([item.actions for item in shards]),
        profile_seeds=np.concatenate([item.profile_seeds for item in shards]),
        sequence_ids=np.concatenate([
            item.sequence_ids.astype(np.int64) + index * 1_000_000
            for index, item in enumerate(shards)
        ]).astype(np.int32),
        split_ids=np.concatenate([item.split_ids for item in shards]),
    )
    combined.validate()
    return combined, [item[1] for item in loaded]

