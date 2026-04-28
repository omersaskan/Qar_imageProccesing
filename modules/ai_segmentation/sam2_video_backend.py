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
        
        # --- Observability Status ---
        self.sam2_mode = "video_temporal"
        self.temporal_consistency = True
        self.api_type = "video_predictor"
        self.seed_frame_idx = 0
        self.seed_prompt_source = "unknown"
        self.expected_frame_count = 0
        self.masks_generated = 0
        self.mask_propagation_failure_count = 0
        self.video_propagation_failed = False
        self.fallback_used = False
        self.propagation_direction = "forward"
        
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
            self.video_propagation_failed = True

    def is_available(self) -> bool:
        return self.predictor is not None

    def _prepare_video_dir(self, frames_dir: Path) -> Tuple[str, List[str]]:
        """
        SAM2 expects frames to be named as integers (e.g. 00000.jpg).
        Creates a temporary directory with symlinks/copies if needed.
        Returns (tmp_dir, list_of_original_filenames).
        """
        tmp_dir = tempfile.mkdtemp(prefix="sam2_video_")
        frame_files = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))
        original_names = [f.name for f in frame_files]
        
        for i, f_path in enumerate(frame_files):
            new_name = f"{i:05d}.jpg"
            dest = Path(tmp_dir) / new_name
            shutil.copy(str(f_path), str(dest))
            
        return tmp_dir, original_names

    def segment_video(
        self, 
        frames_dir: Path, 
        seed_frame_idx: int = 0,
        seed_box: Optional[List[int]] = None,
        seed_points: Optional[np.ndarray] = None,
        seed_labels: Optional[np.ndarray] = None,
        seed_prompt_source: str = "unknown",
        output_dir: Optional[Path] = None
    ) -> Dict[int, np.ndarray]:
        """
        Propagate masks through a sequence of frames.
        """
        self.seed_frame_idx = seed_frame_idx
        self.seed_prompt_source = seed_prompt_source
        self.masks_generated = 0
        self.mask_propagation_failure_count = 0
        self.video_propagation_failed = False
        self.fallback_used = False

        if not self.is_available():
            logger.error("SAM2 Video Predictor not available.")
            self.video_propagation_failed = True
            return {}

        # Validate seed_box
        if seed_box is not None:
            if len(seed_box) != 4:
                logger.error(f"Invalid seed_box length: {len(seed_box)}")
                self.video_propagation_failed = True
                return {}
            x1, y1, x2, y2 = seed_box
            if x2 <= x1 or y2 <= y1:
                logger.error(f"Invalid seed_box dimensions: {seed_box}")
                self.video_propagation_failed = True
                return {}

        import torch
        
        # 1. Prepare video directory
        video_dir, original_names = self._prepare_video_dir(frames_dir)
        self.expected_frame_count = len(original_names)
        
        masks_dict = {}
        obj_id = 1
        
        try:
            # 2. Initialize state
            inference_state = self.predictor.init_state(video_path=video_dir)
            
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
                self.video_propagation_failed = True
                return {}

            # 4. Propagate forward
            logger.info(f"Starting video propagation from frame {seed_frame_idx} (Source: {seed_prompt_source})...")
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)

            for frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                # out_mask_logits is [num_objs, 1, H, W]
                if out_mask_logits is None or len(out_mask_logits) == 0:
                    self.mask_propagation_failure_count += 1
                    continue

                mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(np.uint8) * 255
                masks_dict[frame_idx] = mask
                self.masks_generated += 1
                
                if output_dir and frame_idx < len(original_names):
                    # Preserve original frame filename
                    orig_name = original_names[frame_idx]
                    mask_filename = f"{orig_name}.png"
                    cv2.imwrite(str(output_dir / mask_filename), mask)

            logger.info(f"Video propagation complete. Generated {self.masks_generated} masks.")
            
        except Exception as e:
            logger.error(f"Error during video propagation: {e}")
            self.video_propagation_failed = True
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
        
        if self.masks_generated < self.expected_frame_count:
            self.mask_propagation_failure_count = self.expected_frame_count - self.masks_generated
            logger.warning(f"Incomplete propagation: {self.masks_generated}/{self.expected_frame_count} masks.")
                
        return masks_dict

    def get_status(self) -> Dict[str, Any]:
        return {
            "backend": "sam2_video",
            "available": self.is_available(),
            "model_cfg": self.model_cfg,
            "checkpoint": self.checkpoint,
            "device": self.device,
            "sam2_mode": self.sam2_mode,
            "temporal_consistency": self.temporal_consistency,
            "api_type": self.api_type,
            "seed_frame_idx": self.seed_frame_idx,
            "seed_prompt_source": self.seed_prompt_source,
            "expected_frame_count": self.expected_frame_count,
            "masks_generated": self.masks_generated,
            "mask_propagation_failure_count": self.mask_propagation_failure_count,
            "video_propagation_failed": self.video_propagation_failed,
            "fallback_used": self.fallback_used,
            "propagation_direction": self.propagation_direction
        }
