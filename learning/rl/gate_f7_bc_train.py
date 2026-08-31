"""Same-data parent-initialized BC fine-tune for one Gate F7 observation arm."""
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

from learning.rl.gate_f7_observation import (
    BASE14, NORMAL17, NORMAL_CURVATURE20, observation_dim, observation_feature_names)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model(dim: int, output: int, group: str, device: torch.device):
    dummy = TensorDict({"policy": torch.zeros(1, dim, device=device)}, batch_size=[1])
    kwargs = dict(hidden_dims=[128, 128], activation="elu", obs_normalization=True)
    if group == "actor":
        kwargs["distribution_cfg"] = {
            "class_name": "GaussianDistribution", "init_std": 0.3, "std_type": "scalar"}
    return MLPModel(dummy, {group: ["policy"]}, group, output, **kwargs).to(device)


def _expand_parent(module, parent_state: dict, dim: int, extra_values: torch.Tensor) -> None:
    state = module.state_dict()
    for name in state:
        if name in parent_state and state[name].shape == parent_state[name].shape:
            state[name] = parent_state[name].to(state[name].device).clone()
    state["mlp.0.weight"].zero_()
    state["mlp.0.weight"][:, :14] = parent_state["mlp.0.weight"].to(state["mlp.0.weight"].device)
    for name in ("_mean", "_var", "_std"):
        key = f"obs_normalizer.{name}"
        state[key][:, :14] = parent_state[key].to(state[key].device)
    if dim > 14:
        mean = extra_values.mean(dim=0, keepdim=True)
        var = extra_values.var(dim=0, correction=0, keepdim=True).clamp_min(1.0e-6)
        state["obs_normalizer._mean"][:, 14:] = mean
        state["obs_normalizer._var"][:, 14:] = var
        state["obs_normalizer._std"][:, 14:] = var.sqrt()
    module.load_state_dict(state, strict=True)


def _mean_action(actor, values: torch.Tensor) -> torch.Tensor:
    return actor.mlp(actor.obs_normalizer(values)).clamp(-1.0, 1.0)


def _metrics(actor, x, y, indices) -> dict:
    actor.eval()
    with torch.no_grad():
        error = _mean_action(actor, x[indices]) - y[indices]
    return {
        "n": int(len(indices)), "mse": float(error.square().mean()),
        "mae": float(error.abs().mean()),
        "force_mae": float(error[:, 0].abs().mean()),
        "feed_mae": float(error[:, 1].abs().mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--mode", choices=(BASE14, NORMAL17, NORMAL_CURVATURE20), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    if os.path.exists(out_dir):
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    os.makedirs(out_dir)
    torch.manual_seed(20260901)
    np.random.seed(20260901)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arrays = [np.load(path) for path in args.shards]
    all_x = np.concatenate([item["observations20"] for item in arrays]).astype(np.float32)
    all_y = np.concatenate([item["teacher_actions"] for item in arrays]).astype(np.float32)
    # Stable sample-level split, shared exactly by all three arms.
    rng = np.random.default_rng(20260901)
    order = rng.permutation(len(all_x))
    n_train, n_val = int(0.70 * len(order)), int(0.15 * len(order))
    splits = {"train": order[:n_train], "validation": order[n_train:n_train + n_val],
              "test": order[n_train + n_val:]}
    dim = observation_dim(args.mode)
    x = torch.as_tensor(all_x[:, :dim], device=device)
    y = torch.as_tensor(all_y, device=device)
    ids = {name: torch.as_tensor(value, device=device) for name, value in splits.items()}
    parent = torch.load(args.checkpoint, map_location=device, weights_only=False)
    actor = _model(dim, 2, "actor", device)
    critic = _model(dim, 1, "critic", device)
    extras = x[ids["train"], 14:] if dim > 14 else torch.empty((len(ids["train"]), 0), device=device)
    _expand_parent(actor, parent["actor_state_dict"], dim, extras)
    _expand_parent(critic, parent["critic_state_dict"], dim, extras)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
    best_state, best_val = None, float("inf")
    history = []
    for epoch in range(args.epochs):
        actor.train()
        permutation = ids["train"][torch.randperm(len(ids["train"]), device=device)]
        total = 0.0
        for start in range(0, len(permutation), args.batch_size):
            selected = permutation[start:start + args.batch_size]
            prediction = _mean_action(actor, x[selected])
            loss = torch.nn.functional.mse_loss(prediction, y[selected])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(selected)
        validation = _metrics(actor, x, y, ids["validation"])
        row = {"epoch": epoch, "train_mse": total / len(ids["train"]), **validation}
        history.append(row)
        if validation["mse"] < best_val:
            best_val = validation["mse"]
            best_state = copy.deepcopy(actor.state_dict())
        print(f"[Gate F7 BC] {args.mode} epoch={epoch:02d} val={validation['mse']:.7f}", flush=True)
    actor.load_state_dict(best_state, strict=True)
    metrics = [{"mode": args.mode, "split": name, **_metrics(actor, x, y, index)}
               for name, index in ids.items()]
    with open(os.path.join(out_dir, "epoch_metrics.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    with open(os.path.join(out_dir, "split_metrics.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    info = {
        "gate": "F7_SMALL_BC_ABLATION", "mode": args.mode, "observation_dim": dim,
        "feature_names": observation_feature_names(args.mode), "parent": os.path.abspath(args.checkpoint),
        "parent_sha256": _sha256(args.checkpoint), "samples": len(all_x),
        "split_counts": {name: len(value) for name, value in splits.items()},
        "shards": [{"path": os.path.abspath(path), "sha256": _sha256(path)} for path in args.shards],
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "parent_weight_initialized": True, "critic_training_performed": False,
        "ppo_performed": False, "promotion_allowed": False,
    }
    checkpoint_path = os.path.join(out_dir, f"model_bc_gate_f7_{args.mode}.pt")
    torch.save({"actor_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "iter": 0, "infos": info}, checkpoint_path)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), **info,
                "checkpoint": checkpoint_path, "checkpoint_sha256": _sha256(checkpoint_path),
                "metrics": metrics}
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
