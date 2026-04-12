from pathlib import Path
from typing import Optional, Tuple

def resolve_mask_path(frame_path: Path) -> Tuple[Optional[Path], str]:
    """
    Standardizes mask discovery with a stem-first, legacy-fallback policy.
    Returns (resolved_path, match_mode).
    match_mode standard values: 'stem', 'legacy', 'none'.
    """
    masks_dir = frame_path.parent / "masks"
    if not masks_dir.exists():
        return None, "none"

    # 1. Try stem-based (frame_0000.png) - PREFERRED
    stem_path = masks_dir / f"{frame_path.stem}.png"
    if stem_path.exists():
        return stem_path, "stem"

    # 2. Try legacy double-extension (frame_0000.jpg.png) - FALLBACK
    legacy_path = masks_dir / f"{frame_path.name}.png"
    if legacy_path.exists():
        return legacy_path, "legacy"

    return None, "none"

def resolve_meta_path(frame_path: Path) -> Tuple[Optional[Path], str]:
    """
    Standardizes metadata discovery with a stem-first, legacy-fallback policy.
    Returns (resolved_path, match_mode).
    match_mode standard values: 'stem', 'legacy', 'none'.
    """
    masks_dir = frame_path.parent / "masks"
    if not masks_dir.exists():
        return None, "none"

    # 1. Try stem-based (frame_0000.json) - PREFERRED
    stem_path = masks_dir / f"{frame_path.stem}.json"
    if stem_path.exists():
        return stem_path, "stem"

    # 2. Try legacy double-extension (frame_0000.jpg.json) - FALLBACK
    legacy_path = masks_dir / f"{frame_path.name}.json"
    if legacy_path.exists():
        return legacy_path, "legacy"

    return None, "none"
