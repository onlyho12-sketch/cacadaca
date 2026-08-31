"""Robot-only Gate B2 configuration; existing configs remain untouched."""
from __future__ import annotations

from isaaclab.utils.configclass import configclass

from learning.polytwin.surface_profiles import NEW_CAR_MILD

from .planar_roi_polish_env_cfg import PlanarRoiPolishEnvCfg


@configclass
class ProfiledPlanarRoiPolishEnvCfg(PlanarRoiPolishEnvCfg):
    surface_profile: str = NEW_CAR_MILD
