import fast_simplification
import numpy as np
import trimesh

from .profiles import CleanupProfile


class Remesher:
    def __init__(self):
        pass

    def _has_uv(self, mesh: trimesh.Trimesh) -> bool:
        try:
            return (
                hasattr(mesh.visual, "uv")
                and mesh.visual.uv is not None
                and len(mesh.visual.uv) > 0
            )
        except Exception:
            return False

    def process(self, input_path: str, output_path: str, profile: CleanupProfile) -> int:
        """
        Simplify geometry while trying not to destroy UV/material data.
        """
        mesh = trimesh.load(input_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        if len(mesh.faces) == 0:
            mesh.export(output_path)
            return 0

        mesh.fill_holes()
        mesh.process(validate=True)

        current_faces = len(mesh.faces)
        target_faces = int(profile.target_polycount)
        has_uv = self._has_uv(mesh)

        if current_faces > target_faces:
            if has_uv:
                # Textured assets are decimated conservatively to avoid dropping visuals.
                if current_faces > target_faces * 4:
                    try:
                        candidate = mesh.simplify_quadric_decimation(max(target_faces * 2, target_faces))
                        if self._has_uv(candidate):
                            mesh = candidate
                    except Exception:
                        pass
            else:
                try:
                    points = mesh.vertices.astype(np.float32)
                    faces = mesh.faces.astype(np.uint32)
                    ratio = target_faces / max(current_faces, 1)
                    new_vertices, new_faces = fast_simplification.simplify(points, faces, ratio)
                    mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
                except Exception:
                    mesh = mesh.simplify_quadric_decimation(target_faces)

        mesh.process(validate=True)
        mesh.remove_unreferenced_vertices()
        mesh.export(output_path)
        return len(mesh.vertices)
