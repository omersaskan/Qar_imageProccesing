import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any
from .schema import TrainingDataManifest

logger = logging.getLogger("dataset_registry")

class DatasetRegistry:
    def __init__(self, registry_file: Path):
        self.registry_file = registry_file
        
    def register(self, manifest: TrainingDataManifest):
        try:
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_file, "a", encoding="utf-8") as f:
                f.write(manifest.model_dump_json() + "\n")
        except Exception as e:
            logger.warning(f"Failed to register training data manifest for {manifest.session_id}: {e}")

    def get_all(self) -> List[Dict[str, Any]]:
        if not self.registry_file.exists():
            return []
            
        manifests_by_session = {}
        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "session_id" in data:
                            manifests_by_session[data["session_id"]] = data
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            logger.warning(f"Failed to read dataset registry: {e}")
            
        return list(manifests_by_session.values())
