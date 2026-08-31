import pytest

from learning.rl.gate_e3_ppo_smoke_summarize import pair_rows


def _row(label: str, seed: int = 7):
    return {
        "direction_mode": "same_xx",
        "surface_profile": "factory_prepolish",
        "profile_seed": str(seed),
        "outcome": label,
        "quality_ok": "True",
        "safety_ok": "True",
        "sensor_fault_steps": "0",
        "gu_final": "71.0",
        "ra_final_um": "0.1",
        "rz_final_um": "1.0",
        "scratch_final_um": "0.2",
        "clearcoat_min_um": "29.0",
        "temperature_peak_c": "30.0",
    }


def test_pair_rows_matches_and_computes_delta():
    champion = _row("champion")
    smoke = _row("smoke")
    smoke["gu_final"] = "72.5"
    paired = pair_rows([champion], [smoke])
    assert len(paired) == 1
    assert paired[0]["delta_gu_final"] == pytest.approx(1.5)
    assert paired[0]["smoke_safety_ok"] == 1


def test_pair_rows_rejects_mismatched_keys():
    champion = _row("champion", seed=7)
    smoke = _row("smoke", seed=8)
    with pytest.raises(ValueError, match="keys differ"):
        pair_rows([champion], [smoke])
