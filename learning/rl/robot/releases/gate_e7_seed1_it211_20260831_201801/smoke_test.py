"""Non-PhysX release smoke for the packaged Gate E7 policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

import torch
from rsl_rl.models.mlp_model import MLPModel
from tensordict import TensorDict


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    checkpoint = os.path.abspath(args.checkpoint)
    source = os.path.abspath(args.source_checkpoint)
    out = os.path.abspath(args.out)
    if os.path.exists(out):
        raise FileExistsError(f"refusing to overwrite {out}")

    packaged_hash = sha256(checkpoint)
    source_hash = sha256(source)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter", "infos"}
    state = ck["actor_state_dict"]
    obs_dim = int(state["mlp.0.weight"].shape[1])
    action_dim = int(state["mlp.4.weight"].shape[0])
    tensors_finite = all(
        bool(torch.isfinite(value).all())
        for value in state.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    )

    dummy = TensorDict({"policy": torch.zeros(1, obs_dim)}, batch_size=[1])
    actor = MLPModel(
        dummy,
        {"actor": ["policy"]},
        "actor",
        action_dim,
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.3,
            "std_type": "scalar",
        },
    ).cpu()
    actor.load_state_dict(state, strict=True)
    actor.eval()

    observations = torch.stack(
        (torch.zeros(obs_dim), torch.ones(obs_dim), -torch.ones(obs_dim),
         torch.linspace(-1.0, 1.0, obs_dim))
    )
    td = TensorDict({"policy": observations}, batch_size=[len(observations)])
    with torch.no_grad():
        torch.manual_seed(20260831)
        actions_a = actor(td).clamp(-1.0, 1.0)
        torch.manual_seed(20260831)
        actions_b = actor(td).clamp(-1.0, 1.0)

    checks = {
        "source_and_packaged_sha256_match": packaged_hash == source_hash,
        "required_checkpoint_keys_present": required.issubset(ck),
        "saved_iteration_is_211": int(ck["iter"]) == 211,
        "observation_dimension_is_14": obs_dim == 14,
        "action_dimension_is_2": action_dim == 2,
        "actor_tensors_are_finite": tensors_finite,
        "strict_actor_state_load": True,
        "action_shape_is_4_by_2": list(actions_a.shape) == [4, 2],
        "actions_are_finite": bool(torch.isfinite(actions_a).all()),
        "actions_are_clamped": bool((actions_a >= -1.0).all() and (actions_a <= 1.0).all()),
        "same_seed_actions_are_exact": bool(torch.equal(actions_a, actions_b)),
    }
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_type": "non-destructive CPU checkpoint load and inference smoke",
        "checkpoint": checkpoint,
        "source_checkpoint": source,
        "checkpoint_sha256": packaged_hash,
        "source_checkpoint_sha256": source_hash,
        "saved_iteration": int(ck["iter"]),
        "observation_dim": obs_dim,
        "action_dim": action_dim,
        "input_batch_shape": list(observations.shape),
        "output_batch_shape": list(actions_a.shape),
        "action_min": float(actions_a.min()),
        "action_max": float(actions_a.max()),
        "checks": checks,
        "overall_pass": all(checks.values()),
        "physx_executed": False,
        "training_executed": False,
    }
    with open(out, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["overall_pass"]:
        raise RuntimeError("release smoke failed")


if __name__ == "__main__":
    main()
