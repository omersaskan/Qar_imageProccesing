from typing import Dict
from .base import SegmentationBackend

class BackendFactory:
    _instances: Dict[str, SegmentationBackend] = {}

    @classmethod
    def get_backend(cls, name: str) -> SegmentationBackend:
        if name not in cls._instances:
            if name == "heuristic":
                from .heuristic_backend import HeuristicBackend
                cls._instances[name] = HeuristicBackend()
            elif name == "rembg":
                from .rembg_backend import RembgBackend
                cls._instances[name] = RembgBackend()
            else:
                raise ValueError(f"Unknown segmentation backend: {name}")
        return cls._instances[name]
