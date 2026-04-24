import trimesh
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from modules.operations.logging_config import get_component_logger

logger = get_component_logger("mesh_selector")


class MeshSelector:
    """
    Scores reconstruction mesh candidates and selects the most product-like one.
    Selection is based on:
    - face count
    - compactness
    - centrality
    - flatness penalty
    - fragmentation penalty
    """

    def __init__(self):
        pass

    def _load_mesh(self, mesh_path: str) -> Optional[trimesh.Trimesh]:
        path = Path(mesh_path)
        if not path.exists():
            return None

        try:
            mesh = trimesh.load(str(path))
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            return mesh
        except Exception as e:
            logger.error(f"Failed to load mesh {mesh_path}: {e}")
            return None

    def _safe_compactness(self, mesh: trimesh.Trimesh) -> float:
        try:
            bbox = mesh.bounds[1] - mesh.bounds[0]
            bbox = np.maximum(bbox, 1e-8)
            bbox_volume = float(np.prod(bbox))
            if bbox_volume <= 1e-8:
                return 0.0

            if mesh.is_watertight:
                mesh_volume = float(abs(mesh.volume))
            else:
                mesh_volume = float(abs(mesh.convex_hull.volume))

            return float(mesh_volume / bbox_volume) if bbox_volume > 0 else 0.0
        except Exception:
            return 0.0

    def _flatness_ratio(self, mesh: trimesh.Trimesh) -> float:
        try:
            bbox = mesh.bounds[1] - mesh.bounds[0]
            bbox = np.maximum(bbox, 1e-8)
            return float(np.min(bbox) / np.max(bbox))
        except Exception:
            return 0.0

    def score_mesh(self, mesh: trimesh.Trimesh) -> Dict[str, float]:
        if mesh is None or len(mesh.vertices) == 0:
            return {
                "total_score": 0.0,
                "face_count": 0,
                "component_count": 0,
                "compactness": 0.0,
                "flatness_ratio": 0.0,
                "centrality_score": 0.0,
                "fragment_penalty": 0.0,
            }

        face_count = int(len(mesh.faces)) if hasattr(mesh, "faces") else 0
        if face_count <= 0:
            return {
                "total_score": 0.0,
                "face_count": 0,
                "component_count": 0,
                "compactness": 0.0,
                "flatness_ratio": 0.0,
                "centrality_score": 0.0,
                "fragment_penalty": 0.0,
            }

        # Face score: logarithmic
        face_score = min(1.0, np.log10(face_count + 1) / 5.0)

        # Compactness
        compactness = self._safe_compactness(mesh)
        compactness_score = min(1.0, compactness * 4.0)

        # Centrality
        try:
            center_dist = float(np.linalg.norm(mesh.centroid))
        except Exception:
            center_dist = 999.0
        centrality_score = float(np.exp(-center_dist / 10.0))

        # Flatness penalty
        flatness_ratio = self._flatness_ratio(mesh)
        flatness_penalty = 1.0 if flatness_ratio >= 0.10 else (flatness_ratio / 0.10)

        # Fragmentation
        try:
            comp_count = len(mesh.split(only_watertight=False))
        except Exception:
            comp_count = 1
        fragment_penalty = float(np.exp(-(comp_count - 1) / 4.0))

        total_score = (
            face_score * 0.30
            + compactness_score * 0.20
            + centrality_score * 0.15
            + flatness_penalty * 0.25
            + fragment_penalty * 0.10
        )

        return {
            "total_score": float(total_score),
            "face_score": float(face_score),
            "compactness": float(compactness),
            "compactness_score": float(compactness_score),
            "centrality_score": float(centrality_score),
            "flatness_ratio": float(flatness_ratio),
            "flatness_penalty": float(flatness_penalty),
            "fragment_penalty": float(fragment_penalty),
            "face_count": int(face_count),
            "component_count": int(comp_count),
        }

    def select_best_mesh(self, mesh_paths: List[str]) -> Optional[str]:
        best_path = None
        best_score = -1.0

        for path_str in mesh_paths:
            mesh = self._load_mesh(path_str)
            if mesh is None:
                continue

            scores = self.score_mesh(mesh)
            logger.info(
                f"MeshSelector :: {Path(path_str).name} "
                f"score={scores['total_score']:.4f} "
                f"faces={scores['face_count']} "
                f"components={scores['component_count']} "
                f"flatness={scores['flatness_ratio']:.4f} "
                f"compactness={scores['compactness']:.4f}"
            )

            if scores["total_score"] > best_score:
                best_score = scores["total_score"]
                best_path = path_str

        return best_path