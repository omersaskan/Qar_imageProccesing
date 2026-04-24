import trimesh
import numpy as np
from typing import Dict, Tuple, Optional

class AlignmentProcessor:
    def __init__(self):
        pass

    def align_to_ground(self, input_path: str, output_path: str) -> Tuple[float, Dict[str, float]]:
        """
        Performs real ground alignment and pivot centering.
        Calculates the shift required to place the object's base at Z=0 and center it at (0,0,X).
        """
        mesh = trimesh.load(input_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        if len(mesh.vertices) == 0:
            return 0.0, {"x": 0.0, "y": 0.0, "z": 0.0}

        # 1. Calculate Bounding Box and Centroid
        bounds = mesh.bounds
        min_corner = bounds[0]
        max_corner = bounds[1]
        
        # 2. Alignment logic
        # Shift X,Y to be centered around (0,0)
        # Shift Z so the bottom is at 0
        current_center = mesh.centroid
        target_center_xy = [0.0, 0.0]
        
        shift_x = -current_center[0]
        shift_y = -current_center[1]
        shift_z = -min_corner[2] # Move bottom to Z=0
        
        translation = [shift_x, shift_y, shift_z]
        
        # 3. Apply transformation
        matrix = trimesh.transformations.translation_matrix(translation)
        mesh.apply_transform(matrix)
        
        # 4. Save result
        mesh.export(output_path)
        
        pivot_offset = {"x": float(shift_x), "y": float(shift_y), "z": float(shift_z)}
        ground_offset = float(shift_z)

        return ground_offset, pivot_offset
