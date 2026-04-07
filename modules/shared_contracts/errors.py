class AssetFactoryError(Exception):
    """Base class for all exceptions in the asset factory."""
    pass

class InvalidTransitionError(AssetFactoryError):
    """Raised when an asset status transition is not allowed."""
    def __init__(self, from_status: str, to_status: str):
        self.from_status = from_status
        self.to_status = to_status
        super().__init__(f"Invalid transition from '{from_status}' to '{to_status}'")

class ValidationError(AssetFactoryError):
    """Raised when asset validation fails."""
    pass

class MetadataCorruptionError(AssetFactoryError):
    """Raised when a metadata file is corrupted or invalid."""
    pass

class PathSafetyError(AssetFactoryError):
    """Raised when a path safety violation (e.g., traversal) is detected."""
    pass

class DuplicateAssetError(AssetFactoryError):
    """Raised when attempting to register an asset ID that already exists."""
    pass
