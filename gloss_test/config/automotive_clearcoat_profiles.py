"""Versioned automotive clearcoat profiles used by the RTX gloss tests.

Literature measurements and renderer design parameters are intentionally kept
separate.  The profile below combines two different published reference cases;
it does not claim that the 88.8 GU vehicle was the Toyota specimen.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AutomotiveClearcoatProfile:
    profile_id: str
    display_name: str
    BASE_COLOR: tuple[float, float, float]
    BASE_ROUGHNESS: float
    CLEARCOAT_WEIGHT: float
    IOR: float
    ROUGHNESS_VALUES: tuple[float, ...]
    pristine_ra_um: float
    pristine_ra_source_doi: str
    high_gloss_anchor_20deg_gu: float
    high_gloss_source_doi: str
    actual_gloss_meter_calibrated: bool = False

    def metadata(self):
        data = asdict(self)
        data.update({
            "evidence_scope": {
                "pristine_ra_um": (
                    "2016 white Toyota body specimens before outdoor exposure; "
                    "reported group means were 0.078 and 0.083 um"
                ),
                "high_gloss_anchor_20deg_gu": (
                    "whole-vehicle mean for an acrylic white delivery vehicle; "
                    "manufacturer was not identified as Toyota"
                ),
                "renderer_parameters": (
                    "BASE_COLOR, BASE_ROUGHNESS, CLEARCOAT_WEIGHT and IOR are "
                    "simulation design values, not values directly measured in either paper"
                ),
            },
            "is_literature_composite": True,
        })
        return data


WHITE_AUTOMOTIVE_LITERATURE_COMPOSITE_V1 = AutomotiveClearcoatProfile(
    profile_id="white_automotive_literature_composite_v1",
    display_name="White automotive clearcoat — literature composite v1",
    # Near-white non-metallic base coat.  Kept below 1.0 to retain highlight detail.
    BASE_COLOR=(0.78, 0.80, 0.82),
    BASE_ROUGHNESS=0.32,
    CLEARCOAT_WEIGHT=1.0,
    IOR=1.5,
    ROUGHNESS_VALUES=(0.02, 0.05, 0.10, 0.20, 0.40),
    pristine_ra_um=0.0805,
    pristine_ra_source_doi="10.4236/msa.2017.87036",
    high_gloss_anchor_20deg_gu=88.8,
    high_gloss_source_doi="10.3390/coatings11111320",
)


PROFILES = {
    WHITE_AUTOMOTIVE_LITERATURE_COMPOSITE_V1.profile_id:
        WHITE_AUTOMOTIVE_LITERATURE_COMPOSITE_V1,
}


def get_clearcoat_profile(profile_id):
    try:
        return PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown clearcoat profile {profile_id!r}; choices={sorted(PROFILES)}"
        ) from exc

