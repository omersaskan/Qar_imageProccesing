from typing import Optional
from modules.shared_contracts.errors import AssetFactoryError

class ReconstructionError(AssetFactoryError):
    """Base class for all reconstruction-related errors."""
    pass

class InsufficientInputError(ReconstructionError):
    """Raised when provided keyframes are not enough for reconstruction (pre-flight)."""
    pass

class InsufficientReconstructionError(ReconstructionError):
    """Raised when reconstruction succeeds but result is too sparse/small for densification."""
    pass

class PreprocessingError(ReconstructionError):
    """Raised when frame preprocessing (resize, denoise, etc.) fails."""
    pass

class RuntimeReconstructionError(ReconstructionError):
    """Raised when the geometric reconstruction engine fails during execution."""
    def __init__(self, message: str, output_snippet: Optional[str] = None):
        if output_snippet:
            full_msg = f"{message} | Reason: {output_snippet.strip()}"
        else:
            full_msg = message
        super().__init__(full_msg)

class MissingArtifactError(ReconstructionError):
    """Raised when expected outputs (mesh, texture, manifest) are missing after successful run."""
    def __init__(self, artifact_name: str):
        self.artifact_name = artifact_name
        super().__init__(f"Expected artifact '{artifact_name}' is missing from job directory.")

class DenseMaskAlignmentError(ReconstructionError):
    """Raised when dynamically generated dense masks fail alignment or sanity checks."""
    def __init__(self, message: str, image_path: Optional[str] = None):
        if image_path:
            full_msg = f"Mask alignment failure at {image_path}: {message}"
        else:
            full_msg = message
        super().__init__(full_msg)

class TexturingFailed(ReconstructionError):
    """Raised when TextureMesh fails to produce expected outputs."""
    def __init__(self, message: str, log_path: Optional[str] = None):
        if log_path:
            full_msg = f"{message} | Log: {log_path}"
        else:
            full_msg = message
        super().__init__(full_msg)
        self.log_path = log_path
