from shared_contracts.errors import AssetFactoryError

class ReconstructionError(AssetFactoryError):
    """Base class for all reconstruction-related errors."""
    pass

class InsufficientInputError(ReconstructionError):
    """Raised when provided keyframes are not enough for reconstruction."""
    pass

class PreprocessingError(ReconstructionError):
    """Raised when frame preprocessing (resize, denoise, etc.) fails."""
    pass

class RuntimeReconstructionError(ReconstructionError):
    """Raised when the geometric reconstruction engine fails during execution."""
    pass

class MissingArtifactError(ReconstructionError):
    """Raised when expected outputs (mesh, texture, manifest) are missing after successful run."""
    def __init__(self, artifact_name: str):
        self.artifact_name = artifact_name
        super().__init__(f"Expected artifact '{artifact_name}' is missing from job directory.")
