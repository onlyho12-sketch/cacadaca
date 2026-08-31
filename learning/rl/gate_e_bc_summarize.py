"""Aggregate immutable Gate E2 one-pass evaluation runs."""
from __future__ import annotations

import argparse, csv, hashlib, json, os
from collections import defaultdict
from datetime import datetime, timezone


def read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    if not rows: return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def truth(value): return value.lower() == "true"
def mean(rows, key): return sum(float(row[key]) for row in rows) / len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    out = os.path.abspath(args.out_dir)
    if os.path.exists(out): raise FileExistsError(out)
    os.makedirs(out)
    rows = []
    for directory in args.inputs:
        rows += read(os.path.join(directory, "sequences.csv"))
    if len(rows) != 256: raise RuntimeError(f"expected 256 rows, got {len(rows)}")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["policy_label"], row["direction_mode"], row["surface_profile"])].append(row)
    summary = []
    for (policy, direction, profile), group in sorted(grouped.items()):
        summary.append({
            "policy_label": policy, "direction_mode": direction,
            "surface_profile": profile, "n": len(group),
            "success_n": sum(r["outcome"] == "success" for r in group),
            "quality_ok_n": sum(truth(r["quality_ok"]) for r in group),
            "safety_ok_n": sum(truth(r["safety_ok"]) for r in group),
            "sensor_fault_steps": sum(int(r["sensor_fault_steps"]) for r in group),
            "gu_final_mean": mean(group, "gu_final"),
            "ra_final_um_mean": mean(group, "ra_final_um"),
            "rz_final_um_mean": mean(group, "rz_final_um"),
            "scratch_final_um_mean": mean(group, "scratch_final_um"),
            "clearcoat_min_um_min": min(float(r["clearcoat_min_um"]) for r in group),
        })
    by_key = defaultdict(dict)
    for row in rows:
        key = (row["direction_mode"], row["surface_profile"], int(row["profile_seed"]))
        by_key[key][row["policy_label"]] = row
    paired = []
    for (direction, profile, seed), policies in sorted(by_key.items()):
        champion = policies["frozen_champion14"]
        for policy in ("bc_base14", "bc_global20", "bc_spatial120"):
            row = policies[policy]
            paired.append({
                "policy_label": policy, "direction_mode": direction,
                "surface_profile": profile, "profile_seed": seed,
                "policy_outcome": row["outcome"], "champion_outcome": champion["outcome"],
                "policy_safety_ok": row["safety_ok"],
                "champion_safety_ok": champion["safety_ok"],
                "delta_gu": float(row["gu_final"]) - float(champion["gu_final"]),
                "delta_ra_um": float(row["ra_final_um"]) - float(champion["ra_final_um"]),
                "delta_rz_um": float(row["rz_final_um"]) - float(champion["rz_final_um"]),
                "delta_scratch_um": float(row["scratch_final_um"]) - float(champion["scratch_final_um"]),
                "delta_clearcoat_um": float(row["clearcoat_min_um"]) - float(champion["clearcoat_min_um"]),
            })
    write(os.path.join(out, "all_sequences.csv"), rows)
    write(os.path.join(out, "policy_path_profile_summary.csv"), summary)
    write(os.path.join(out, "paired_vs_champion.csv"), paired)
    with open(os.path.join(out, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump({"created_utc": datetime.now(timezone.utc).isoformat(),
                   "input_dirs": [os.path.abspath(x) for x in args.inputs],
                   "executions": len(rows), "paired_rows": len(paired),
                   "max_passes": 1, "ppo_performed": False,
                   "selection": "KEEP_FROZEN_CHAMPION_REJECT_NEW_BC_PROMOTION"},
                  handle, indent=2, sort_keys=True)
    print(f"Gate E2 summary: {len(rows)} executions, {len(paired)} paired rows -> {out}")


if __name__ == "__main__": main()
