from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

CIVILIAN_CLASSES = frozenset(
    {
        "change",
        "no_change",
        "water",
        "pond",
        "vegetation",
        "building",
        "vehicle",
        "person",
        "boat",
        "other",
    }
)

BLOCKED_CLASS_SUBSTR = (
    "tank",
    "warship",
    "missile",
    "launcher",
    "radar",
    "submarine",
    "target",
    "enemy",
)

REQUIRED_COLUMNS = ("chip_id", "scene_id", "label")


@dataclass(frozen=True)
class ChipRecord:
    chip_id: str
    scene_id: str
    label: str
    flight_id: Optional[str] = None
    aoi_id: Optional[str] = None
    path_t0: Optional[str] = None
    path_t1: Optional[str] = None
    path_mask: Optional[str] = None
    date_t0: Optional[str] = None
    date_t1: Optional[str] = None
    gsd_m: Optional[float] = None
    sensor: Optional[str] = None
    license: Optional[str] = None

    @property
    def is_change(self) -> bool:
        return str(self.label).lower() in {"change", "1", "true"}

    @property
    def split_key(self) -> str:
        if self.flight_id:
            return f"flight:{self.flight_id}"
        return f"scene:{self.scene_id}"
