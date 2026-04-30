import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np
import cv2
import base64

from modules.operations.settings import settings

logger = logging.getLogger(__name__)

class SegmentationPreviewProvider(ABC):
    @abstractmethod
    def get_mask(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Processes an image and returns a mask preview.
        Returns a dict with mask data and metadata.
        """
        pass

class LegacyProvider(SegmentationPreviewProvider):
    def get_mask(self, image: np.ndarray) -> Dict[str, Any]:
        h, w = image.shape[:2]
        # Create a simple centered box as a "legacy" mask
        mask = np.zeros((h, w), dtype=np.uint8)
        padding_w = int(w * 0.25)
        padding_h = int(h * 0.2)
        mask[padding_h:h-padding_h, padding_w:w-padding_w] = 255
        
        # Convert to polygon for lightweight response
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon = []
        if contours:
            # Take the largest contour
            largest = max(contours, key=cv2.contourArea)
            # Simplify polygon
            epsilon = 0.01 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)
            polygon = approx.reshape(-1, 2).tolist()

        return {
            "provider": "legacy",
            "mask_format": "polygon",
            "mask_width": w,
            "mask_height": h,
            "mask": polygon,
            "confidence": 0.5, # Legacy is low confidence
            "fallback_used": False
        }

class SAM2Provider(SegmentationPreviewProvider):
    def get_mask(self, image: np.ndarray) -> Dict[str, Any]:
        # Scaffold for SAM2
        if not settings.sam2_enabled:
            logger.warning("SAM2Provider called but SAM2_ENABLED=false")
            return self._get_fallback(image, "sam2")
            
        # In Phase A, we just scaffold this. Real integration would call SAM2Wrapper.
        # For now, return a slightly better box or a dummy.
        return {
            "provider": "sam2",
            "mask_format": "polygon",
            "mask_width": image.shape[1],
            "mask_height": image.shape[0],
            "mask": [], # Empty polygon implies failure/scaffold
            "confidence": 0.0,
            "fallback_used": True
        }

    def _get_fallback(self, image: np.ndarray, provider: str) -> Dict[str, Any]:
        return {
            "provider": provider,
            "mask_format": "polygon",
            "mask_width": image.shape[1],
            "mask_height": image.shape[0],
            "mask": [],
            "confidence": 0.0,
            "fallback_used": True
        }

class SAM3Provider(SegmentationPreviewProvider):
    def get_mask(self, image: np.ndarray) -> Dict[str, Any]:
        # Scaffold for SAM3 (Text-prompted)
        if not settings.sam3_enabled:
             return self._get_fallback(image, "sam3")

        return {
            "provider": "sam3",
            "mask_format": "polygon",
            "mask_width": image.shape[1],
            "mask_height": image.shape[0],
            "mask": [],
            "confidence": 0.0,
            "fallback_used": True
        }

    def _get_fallback(self, image: np.ndarray, provider: str) -> Dict[str, Any]:
        return {
            "provider": provider,
            "mask_format": "polygon",
            "mask_width": image.shape[1],
            "mask_height": image.shape[0],
            "mask": [],
            "confidence": 0.0,
            "fallback_used": True
        }

def get_preview_provider() -> SegmentationPreviewProvider:
    provider_name = settings.segmentation_preview_provider.lower()
    if provider_name == "sam2":
        return SAM2Provider()
    if provider_name == "sam3":
        return SAM3Provider()
    return LegacyProvider()
