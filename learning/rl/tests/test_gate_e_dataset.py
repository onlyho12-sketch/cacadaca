"""Isaac-free tests for Gate E dataset persistence and split invariants."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, _REPO_ROOT)

from learning.rl.env.gate_d_observation import BASE14, GLOBAL20, SPATIAL120  # noqa: E402
from learning.rl.gate_e_dataset import (  # noqa: E402
    GateEShard,
    load_shard,
    save_shard,
    split_id_for_seed,
)


def main() -> None:
    seeds = np.asarray([42, 42, 139, 236, 333, 430], dtype=np.int64)
    observations = np.arange(6 * 120, dtype=np.float32).reshape(6, 120) / 100.0
    actions = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(6, 2)
    shard = GateEShard(
        observations120=observations,
        actions=actions,
        profile_seeds=seeds,
        sequence_ids=np.asarray([0, 0, 1, 2, 3, 4], dtype=np.int32),
        split_ids=np.asarray([split_id_for_seed(seed) for seed in seeds], dtype=np.uint8),
    )
    shard.validate()
    assert shard.split_ids[0] == shard.split_ids[1]
    assert shard.observations(BASE14).shape == (6, 14)
    assert shard.observations(GLOBAL20).shape == (6, 20)
    assert shard.observations(SPATIAL120).shape == (6, 120)
    with tempfile.TemporaryDirectory(prefix="gate_e_dataset_") as directory:
        path = os.path.join(directory, "shard.npz")
        save_shard(path, shard, {"training_performed": False})
        restored, metadata = load_shard(path)
        np.testing.assert_array_equal(restored.observations120, observations)
        np.testing.assert_array_equal(restored.actions, actions)
        assert metadata["training_performed"] is False
    print("Gate E dataset: schema/split/prefix/roundtrip tests PASS")


if __name__ == "__main__":
    main()
