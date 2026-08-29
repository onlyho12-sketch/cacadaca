"""Small compatibility adapters for Isaac Sim API changes."""

import numpy as np


try:
    # Isaac Sim 6.0+: contact sensor authoring/runtime were moved to the
    # experimental physics API and no longer use initialize/get_current_frame.
    from isaacsim.sensors.experimental.physics import Contact as _Contact
    from isaacsim.sensors.experimental.physics import ContactSensor as _ContactSensor
except ImportError:
    # Isaac Sim 4.5 fallback.  Keeping this import means the project remains
    # runnable from the preserved 4.5 installation while migration is verified.
    from isaacsim.sensors.physics import ContactSensor  # noqa: F401
else:
    class ContactSensor:
        """Expose the Isaac Sim 4.5 ContactSensor surface on Isaac Sim 6.0."""

        def __init__(
            self,
            prim_path,
            name=None,
            frequency=60,
            translation=None,
            **_kwargs,
        ):
            del name, frequency
            translations = None
            if translation is not None:
                translations = np.asarray(translation, dtype=float).reshape(1, 3)
            contact = _Contact.create(
                prim_path,
                translations=translations,
                min_threshold=0.0,
                max_threshold=100000.0,
                radius=-1.0,
            )
            self._sensor = _ContactSensor(contact)

        def initialize(self):
            """The 6.0 runtime registers itself when constructed."""

        def get_current_frame(self):
            return self._sensor.get_data()
