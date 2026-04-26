import os
import re
from pathlib import Path
from typing import Dict, Optional

def get_mesh_stats_cheaply(mesh_path: str) -> Dict[str, int]:
    """
    Parses mesh headers (PLY or OBJ) to get face/vertex counts without loading the whole mesh.
    For PLY, it reads the header. For OBJ, it might need to scan the file (still faster than trimesh load).
    """
    path = Path(mesh_path)
    if not path.exists():
        return {"face_count": 0, "vertex_count": 0}

    ext = path.suffix.lower()
    if ext == ".ply":
        return _get_ply_stats(path)
    elif ext == ".obj":
        return _get_obj_stats(path)
    
    return {"face_count": 0, "vertex_count": 0}

def _get_ply_stats(path: Path) -> Dict[str, int]:
    stats = {"face_count": 0, "vertex_count": 0}
    try:
        with open(path, "rb") as f:
            header_found = False
            for line in f:
                line_str = line.decode("ascii", errors="ignore").strip()
                if line_str == "end_header":
                    header_found = True
                    break
                
                if line_str.startswith("element vertex"):
                    parts = line_str.split()
                    if len(parts) >= 3:
                        stats["vertex_count"] = int(parts[2])
                elif line_str.startswith("element face"):
                    parts = line_str.split()
                    if len(parts) >= 3:
                        stats["face_count"] = int(parts[2])
            
            if not header_found:
                # If we didn't find end_header, maybe it's not a valid PLY
                pass
    except Exception:
        pass
    return stats

def _get_obj_stats(path: Path) -> Dict[str, int]:
    stats = {"face_count": 0, "vertex_count": 0}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    stats["vertex_count"] += 1
                elif line.startswith("f "):
                    stats["face_count"] += 1
    except Exception:
        pass
    return stats
