import json
import logging
import os
from pathlib import Path
from .schema import TrainingDataManifest

logger = logging.getLogger("dataset_registry")

class DatasetRegistry:
    def __init__(self, registry_file: Path):
        self.registry_file = registry_file
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        
    def register(self, manifest: TrainingDataManifest):
        try:
            with open(self.registry_file, "a", encoding="utf-8") as f:
                f.write(manifest.model_dump_json() + "\n")
        except Exception as e:
            logger.warning(f"Failed to register training data manifest for {manifest.session_id}: {e}")

    def get_all(self):
        manifests = []
        if not self.registry_file.exists():
            return manifests
            
        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        manifests.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Failed to read dataset registry: {e}")
            
        return manifests
