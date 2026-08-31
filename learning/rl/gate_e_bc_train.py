"""Offline behavior cloning for Gate E2 observation ablation.

All modes consume the exact same successful-sequence shards and deterministic
seed split.  The critic is schema-compatible but deliberately untrained; PPO is
outside Gate E2.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from rsl_rl.models.mlp_model import MLPModel
from tensordict import TensorDict

from learning.rl.env.gate_d_observation import (
    BASE14,
    GLOBAL20,
    SPATIAL120,
    observation_dim,
    observation_feature_names,
)
from learning.rl.gate_e_dataset import SPLIT_NAMES, combine_shards


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model(obs_dim: int, output_dim: int, obs_set: str, device: torch.device):
    dummy = TensorDict(
        {"policy": torch.zeros(1, obs_dim, device=device)}, batch_size=[1])
    kwargs = dict(
        hidden_dims=[128, 128], activation="elu", obs_normalization=True)
    if obs_set == "actor":
        kwargs["distribution_cfg"] = {
            "class_name": "GaussianDistribution", "init_std": 0.3,
            "std_type": "scalar",
        }
    return MLPModel(
        dummy, {obs_set: ["policy"]}, obs_set, output_dim, **kwargs).to(device)


def _predict(actor, values: torch.Tensor, batch_size: int = 16384) -> torch.Tensor:
    outputs = []
    actor.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = values[start:start + batch_size]
            td = TensorDict({"policy": batch}, batch_size=[len(batch)])
            outputs.append(actor(td).clamp(-1.0, 1.0))
    return torch.cat(outputs) if outputs else torch.empty((0, 2), device=values.device)


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    error = prediction - target
    return {
        "n": int(len(target)),
        "mse": float(error.square().mean()),
        "mae": float(error.abs().mean()),
        "force_mse": float(error[:, 0].square().mean()),
        "feed_mse": float(error[:, 1].square().mean()),
        "force_mae": float(error[:, 0].abs().mean()),
        "feed_mae": float(error[:, 1].abs().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--mode", choices=(BASE14, GLOBAL20, SPATIAL120), required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, patience, and batch_size must be positive")
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    shard_paths = [os.path.abspath(path) for path in args.shards]
    dataset, shard_metadata = combine_shards(shard_paths)
    dim = observation_dim(args.mode)
    X = torch.as_tensor(dataset.observations(args.mode), device=device)
    Y = torch.as_tensor(dataset.actions, device=device)
    split_ids = torch.as_tensor(dataset.split_ids.astype(np.int64), device=device)
    indices = {
        name: (split_ids == split_id).nonzero(as_tuple=False).squeeze(-1)
        for split_id, name in enumerate(SPLIT_NAMES)
    }
    if any(len(value) == 0 for value in indices.values()):
        raise RuntimeError(
            f"all splits must be non-empty: { {name: len(value) for name, value in indices.items()} }")

    actor = _model(dim, 2, "actor", device)
    critic = _model(dim, 1, "critic", device)
    train_td = TensorDict(
        {"policy": X[indices["train"]]}, batch_size=[len(indices["train"])])
    actor.update_normalization(train_td)
    critic.update_normalization(train_td)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=args.learning_rate)

    history = []
    best_val = float("inf")
    best_epoch = -1
    best_state = None
    stale = 0
    train_index = indices["train"]
    for epoch in range(args.epochs):
        actor.train()
        permutation = train_index[torch.randperm(len(train_index), device=device)]
        train_loss_sum = 0.0
        for start in range(0, len(permutation), args.batch_size):
            selected = permutation[start:start + args.batch_size]
            td = TensorDict({"policy": X[selected]}, batch_size=[len(selected)])
            prediction = actor(td)
            loss = torch.nn.functional.mse_loss(prediction, Y[selected])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss_sum += float(loss.detach()) * len(selected)
        validation_prediction = _predict(actor, X[indices["validation"]])
        validation = _metrics(validation_prediction, Y[indices["validation"]])
        row = {
            "epoch": epoch,
            "train_mse": train_loss_sum / len(train_index),
            "validation_mse": validation["mse"],
            "validation_mae": validation["mae"],
        }
        history.append(row)
        print(
            f"[Gate E2 train] mode={args.mode} epoch={epoch:02d} "
            f"train_mse={row['train_mse']:.7f} val_mse={row['validation_mse']:.7f}",
            flush=True)
        if validation["mse"] < best_val - 1e-8:
            best_val = validation["mse"]
            best_epoch = epoch
            best_state = copy.deepcopy(actor.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    actor.load_state_dict(best_state)
    actor.eval()

    metrics_rows = []
    for name in SPLIT_NAMES:
        selected = indices[name]
        metrics_rows.append({
            "observation_mode": args.mode,
            "observation_dim": dim,
            "split": name,
            **_metrics(_predict(actor, X[selected]), Y[selected]),
        })
    with open(os.path.join(out_dir, "epoch_metrics.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    with open(os.path.join(out_dir, "split_metrics.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0]))
        writer.writeheader(); writer.writerows(metrics_rows)

    checkpoint_path = os.path.join(out_dir, f"model_bc_gate_e2_{args.mode}.pt")
    info = {
        "source": "gate_e2_factory_success_bc_pilot",
        "observation_mode": args.mode,
        "observation_dim": dim,
        "feature_names": observation_feature_names(args.mode),
        "dataset_samples": len(dataset.observations120),
        "split_counts": {name: int(len(index)) for name, index in indices.items()},
        "dataset_shards": [
            {"path": path, "sha256": _sha256(path)} for path in shard_paths],
        "teacher": "frozen_champion_pass1",
        "factory_only_training": True,
        "legacy_training_samples": 0,
        "critic_training_performed": False,
        "ppo_performed": False,
        "best_epoch": best_epoch,
        "best_validation_mse": best_val,
        "seed": args.seed,
    }
    torch.save({
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter": 0,
        "infos": info,
    }, checkpoint_path)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **info,
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "epochs_completed": len(history),
        "early_stopped": len(history) < args.epochs,
        "training_performed": True,
        "shard_metadata": shard_metadata,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(
        f"[Gate E2 train] saved {checkpoint_path}; best_epoch={best_epoch} "
        f"val_mse={best_val:.7f}")


if __name__ == "__main__":
    main()
