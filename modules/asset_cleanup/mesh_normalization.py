"""
Mesh normalization config — Sprint 6.

Defines what the Blender script should do to the mesh before GLB export.
All options default to safe / non-destructive values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class NormalizationConfig:
    # Origin / pivot alignment
    align_to_origin: bool = True          # move object origin to world origin
    align_ground_to_z_zero: bool = True   # translate so lowest point of bbox is z=0
    apply_scale: bool = True              # apply scale transforms (Blender apply)

    # Rotation correction
    forward_axis: str = "-Z"              # Blender forward axis for GLB export
    up_axis: str = "Y"                    # Blender up axis for GLB export

    # Decimation (disabled by default — destructive)
    decimate_enabled: bool = False
    decimate_ratio: float = 0.5           # 0.0–1.0; only used when decimate_enabled=True
    decimate_min_faces: int = 5000        # skip decimation below this face count

    # Material / texture
    keep_materials: bool = True
    image_format: str = "PNG"             # for embedded textures

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NormalizationConfig":
        cfg = cls()
        for k, v in d.items():
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, type(getattr(cfg, k))(v))
                except Exception:
                    pass
        return cfg
