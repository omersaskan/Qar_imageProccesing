"""
SAM2 Video Segmentation Backend
=================================

Implements temporal segmentation using SAM2 Video Predictor API.
Supports video propagation from a seed frame (usually frame 0).
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

logger = logging.getLogger("sam2_video_backend")

class SAM2VideoBackend:
    def __init__(self, model_cfg: str, checkpoint: str, device: str = "cuda"):
        self.model_cfg = model_cfg
        self.checkpoint = checkpoint
        self.device = device
        self.predictor = None
        self._load_model()

    def _load_model(self):
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor
            
            logger.info(f"Loading SAM2 Video Predictor: {self.checkpoint}")
            self.predictor = build_sam2_video_predictor(
                self.model_cfg, 
                self.checkpoint, 
                device=self.device
            )
            logger.info("SAM2 Video Predictor loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SAM2 Video Predictor: {e}")
            self.predictor = None

    def is_available(self) -> bool:
        return self.predictor is not None

    def _prepare_video_dir(self, frames_dir: Path) -> str:
        """
        SAM2 expects frames to be named as integers (e.g. 00000.jpg).
        Creates a temporary directory with symlinks/copies if needed.
        """
        tmp_dir = tempfile.mkdtemp(prefix="sam2_video_")
        frame_files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
        
        for i, f_path in enumerate(frame_files):
            # Create a name like 00000.jpg
            new_name = f"{i:05d}.jpg"
            dest = Path(tmp_dir) / new_name
            # Copy file (symlinks might be tricky on Windows depending on permissions)
            shutil.copy(str(f_path), str(dest))
            
        return tmp_dir

    def segment_video(
        self, 
        frames_dir: Path, 
        seed_frame_idx: int = 0,
        seed_box: Optional[List[int]] = None,
        seed_points: Optional[np.ndarray] = None,
        seed_labels: Optional[np.ndarray] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[int, np.ndarray]:
        """
        Propagate masks through a sequence of frames.
        """
        if not self.is_available():
            logger.error("SAM2 Video Predictor not available.")
            return {}

        import torch
        
        # 1. Prepare video directory (SAM2 naming requirement)
        video_dir = self._prepare_video_dir(frames_dir)
        
        try:
            # 2. Initialize state
            inference_state = self.predictor.init_state(video_path=video_dir)
            
            masks_dict = {}
            obj_id = 1
            
            # 3. Add prompt to seed frame
            if seed_box is not None:
                box_np = np.array(seed_box, dtype=np.float32)
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=seed_frame_idx,
                    obj_id=obj_id,
                    box=box_np,
                )
            elif seed_points is not None and seed_labels is not None:
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=seed_frame_idx,
                    obj_id=obj_id,
                    points=seed_points,
                    labels=seed_labels,
                )
            else:
                logger.error("No seed prompt provided for video segmentation.")
                return {}

            # 4. Propagate forward
            logger.info(f"Starting video propagation from frame {seed_frame_idx}...")
            for frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                # out_mask_logits is [num_objs, 1, H, W]
                mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8) * 255
                masks_dict[frame_idx] = mask
                
                if output_dir:
                    mask_filename = f"frame_{frame_idx:04d}.jpg.png"
                    cv2.imwrite(str(output_dir / mask_filename), mask)

            logger.info(f"Video propagation complete. Generated {len(masks_dict)} masks.")
            
        except Exception as e:
            logger.error(f"Error during video propagation: {e}")
        finally:
            # Clean up state to free GPU memory
            try:
                self.predictor.reset_state(inference_state)
            except:
                pass
            # Clean up temp dir
            try:
                shutil.rmtree(video_dir)
            except:
                pass
                
        return masks_dict

    def get_status(self) -> Dict[str, Any]:
        return {
            "backend": "sam2_video",
            "available": self.is_available(),
            "model_cfg": self.model_cfg,
            "checkpoint": self.checkpoint,
            "device": self.device
        }
