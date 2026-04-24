import trimesh
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from modules.operations.logging_config import get_component_logger

logger = get_component_logger("texture_reprojector")

class TextureReprojector:
    """
    Experimental lightweight texture reprojection / transfer layer.
    Allows transferring visual data from a high-poly textured mesh to a cleaned low-poly mesh.
    """
    def __init__(self):
        pass

    def transfer_texture(self, source_mesh_path: str, target_mesh_path: str, output_path: str) -> bool:
        """
        Attempts to transfer UVs and materials from source to target mesh.
        Initially implements a 'preserve via safer path' fallback logic.
        """
        try:
            source = trimesh.load(source_mesh_path)
            target = trimesh.load(target_mesh_path)
            
            if isinstance(source, trimesh.Scene):
                source = source.dump(concatenate=True)
            if isinstance(target, trimesh.Scene):
                target = target.dump(concatenate=True)
                
            has_source_uv = hasattr(source.visual, 'uv') and source.visual.uv is not None
            
            if not has_source_uv:
                logger.warning("Source mesh has no UVs to transfer.")
                return False

            # Minimal transfer: If topology is similar (e.g. just decimated), 
            # we might use nearest neighbor for UVs if target lost them.
            # However, for the first stable version, we focus on 'preserve during simplification' 
            # and use this module for 'explicit degraded-texture mode' detection.
            
            if not hasattr(target.visual, 'uv') or target.visual.uv is None:
                logger.info("Target mesh lost UVs. Attempting nearest-neighbor UV transfer...")
                # Simple vertex-based transfer if vertices are close
                _, _, index = target.kdtree.query(source.vertices)
                # This is actually reverse, we want to find source vertices for target vertices
                _, index = source.kdtree.query(target.vertices)
                target.visual = trimesh.visual.TextureVisuals(
                    uv=source.visual.uv[index],
                    material=source.visual.material
                )
            
            target.export(output_path)
            return True
            
        except Exception as e:
            logger.error(f"Texture transfer failed: {e}")
            return False

    def get_degraded_status(self, original_uv_count: int, final_uv_count: int) -> dict:
        """Returns details about texture degradation if any."""
        return {
            "degraded": final_uv_count < original_uv_count * 0.5,
            "residual_ratio": final_uv_count / original_uv_count if original_uv_count > 0 else 0
        }
