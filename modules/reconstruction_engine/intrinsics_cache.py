"""
Intrinsics cache — share calibration across captures from the same device/sensor.

Goal: when a 2nd capture from the same device hits the pipeline, we can pass
COLMAP a prior `--ImageReader.camera_params` instead of solving from scratch.
Doesn't replace COLMAP's BA — just gives it a head start (and a ground truth
to drift from when feature matches are sparse).

Cache key: device + WxH + focal-length-bin + camera_id.  Coarse on purpose:
the same iPhone 15 Pro main lens at 1080p produces near-identical intrinsics.

Cache file: `<data_root>/intrinsics_cache.json`.  Atomic via tmp + rename.
File schema:
    {
      "schema_version": 1,
      "entries": {
        "<key>": {
          "fx": float, "fy": float, "cx": float, "cy": float,
          "k1": float, "k2": float,
          "source": "metadata|estimated|default",
          "first_seen": ISO-8601, "last_used": ISO-8601, "use_count": int
        },
        ...
      }
    }
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

CACHE_SCHEMA_VERSION = 1


@dataclass
class IntrinsicsRecord:
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    source: str = "default"
    first_seen: str = ""
    last_used: str = ""
    use_count: int = 0
    cache_key: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CacheLookupResult:
    status: str            # "hit" | "miss" | "disabled"
    cache_key: str
    record: Optional[IntrinsicsRecord] = None
    source: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "status": self.status,
            "cache_key": self.cache_key,
            "source": self.source,
        }
        if self.record:
            d["record"] = self.record.to_dict()
        return d


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_cache_key(
    width: int,
    height: int,
    device_model: Optional[str] = None,
    focal_mm: Optional[float] = None,
    camera_id: Optional[str] = None,
) -> str:
    """
    Coarse, deterministic key.  Focal binned to 0.5 mm to forgive EXIF jitter.
    Empty/None pieces become "any" so partial signals still group.
    """
    dm = (device_model or "any").strip().lower().replace(" ", "_") or "any"
    cam = (camera_id or "any").strip().lower().replace(" ", "_") or "any"
    if focal_mm and focal_mm > 0:
        binned = round(float(focal_mm) * 2) / 2.0  # 0.5 mm bin
        focal_str = f"f{binned:.1f}"
    else:
        focal_str = "fany"
    return f"{dm}|{int(width)}x{int(height)}|{focal_str}|{cam}"


def _default_intrinsics(width: int, height: int, focal_mm: Optional[float] = None) -> IntrinsicsRecord:
    """
    Reasonable default — assume 35mm-equivalent focal of 26mm if unknown.
    fx ≈ fy ≈ width * (focal_equiv / 36mm sensor) / 1.0  (pixels)
    Actually: fx_pixels = focal_mm_equiv * width / sensor_width_mm.
    We approximate sensor width as 36mm (full-frame ref); good enough for prior.
    """
    f_equiv = float(focal_mm) if focal_mm and focal_mm > 0 else 26.0
    sensor_width_mm = 36.0
    fx = f_equiv * width / sensor_width_mm
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return IntrinsicsRecord(
        fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
        source="default",
        first_seen=_now_iso(),
        last_used=_now_iso(),
        use_count=0,
    )


class IntrinsicsCache:
    """File-backed cache.  Safe under sequential writes; not multi-process safe."""

    def __init__(self, cache_path: Path):
        self.path = Path(cache_path)
        self._data: Dict[str, Any] = {"schema_version": CACHE_SCHEMA_VERSION, "entries": {}}
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict) and "entries" in raw:
                    self._data = raw
                    if self._data.get("schema_version") != CACHE_SCHEMA_VERSION:
                        # Migrate trivial cases by ignoring; preserve entries
                        self._data["schema_version"] = CACHE_SCHEMA_VERSION
            except Exception:
                # Corrupt cache → start fresh, don't blow up
                self._data = {"schema_version": CACHE_SCHEMA_VERSION, "entries": {}}
        self._loaded = True

    def lookup(
        self,
        width: int,
        height: int,
        device_model: Optional[str] = None,
        focal_mm: Optional[float] = None,
        camera_id: Optional[str] = None,
    ) -> CacheLookupResult:
        self._ensure_loaded()
        key = build_cache_key(width, height, device_model, focal_mm, camera_id)
        entries = self._data.get("entries", {})
        if key in entries:
            try:
                rec_d = entries[key]
                rec = IntrinsicsRecord(
                    fx=rec_d["fx"], fy=rec_d["fy"], cx=rec_d["cx"], cy=rec_d["cy"],
                    k1=rec_d.get("k1", 0.0), k2=rec_d.get("k2", 0.0),
                    source=rec_d.get("source", "estimated"),
                    first_seen=rec_d.get("first_seen", _now_iso()),
                    last_used=rec_d.get("last_used", _now_iso()),
                    use_count=int(rec_d.get("use_count", 0)),
                    cache_key=key,
                )
                # Touch
                rec.use_count += 1
                rec.last_used = _now_iso()
                entries[key] = rec.to_dict()
                self._save_atomic()
                return CacheLookupResult(status="hit", cache_key=key, record=rec, source=rec.source)
            except Exception:
                pass
        # Miss → seed default + persist
        rec = _default_intrinsics(width, height, focal_mm)
        rec.cache_key = key
        rec.source = "default"
        entries[key] = rec.to_dict()
        self._data["entries"] = entries
        self._save_atomic()
        return CacheLookupResult(status="miss", cache_key=key, record=rec, source="default")

    def insert_metadata(
        self,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        device_model: Optional[str] = None,
        focal_mm: Optional[float] = None,
        camera_id: Optional[str] = None,
        source: str = "metadata",
    ) -> CacheLookupResult:
        """Force-insert a record (e.g. post-COLMAP BA result feedback)."""
        self._ensure_loaded()
        key = build_cache_key(width, height, device_model, focal_mm, camera_id)
        rec = IntrinsicsRecord(
            fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
            source=source,
            first_seen=_now_iso(),
            last_used=_now_iso(),
            use_count=1,
            cache_key=key,
        )
        self._data.setdefault("entries", {})[key] = rec.to_dict()
        self._save_atomic()
        return CacheLookupResult(status="hit", cache_key=key, record=rec, source=source)

    def _save_atomic(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
        try:
            os.replace(tmp, self.path)
        except Exception:
            tmp.replace(self.path)


def disabled_lookup(width: int, height: int, **kwargs) -> CacheLookupResult:
    """Helper for callers when intrinsics_cache_enabled=false."""
    rec = _default_intrinsics(width, height, kwargs.get("focal_mm"))
    rec.cache_key = "disabled"
    return CacheLookupResult(status="disabled", cache_key="disabled", record=rec, source="default")
