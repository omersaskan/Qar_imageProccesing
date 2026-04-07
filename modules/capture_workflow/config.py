from pydantic import BaseModel

class QualityThresholds(BaseModel):
    # Blur detection (higher variance = sharper image)
    min_blur_score: float = 5.0
    
    # Exposure (0-255 scale)
    min_exposure_score: float = 40.0
    max_exposure_score: float = 230.0
    
    # Extraction & Similarity
    frame_sample_rate: int = 15 # Sample every 15th frame
    min_similarity_score: float = 0.95 # Threshold for rejecting redundant frames

class ExtractionConfig(BaseModel):
    max_frames: int = 100
    min_frames: int = 5

# Global configuration that can be overriden in tests
default_quality_thresholds = QualityThresholds()
default_extraction_config = ExtractionConfig()
