import json
from pathlib import Path
from modules.training_data.manifest_builder import TrainingManifestBuilder
from modules.training_data.dataset_registry import DatasetRegistry
from modules.training_data.schema import TrainingDataManifest
import pytest

def test_training_manifest_builder_with_missing_reports(tmp_path: Path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    
    session_id = "test_session_1"
    product_id = "prod_x"
    
    # Empty session
    builder = TrainingManifestBuilder(data_root=data_root)
    manifest = builder.build(session_id, product_id, eligible_for_training=False)
    
    assert manifest.session_id == session_id
    assert manifest.eligible_for_training is False
    assert manifest.product_hash != product_id  # hashed
    assert manifest.poly_count == 0

def test_dataset_registry_append(tmp_path: Path):
    registry_file = tmp_path / "dataset_registry.jsonl"
    registry = DatasetRegistry(registry_file)
    
    manifest1 = TrainingDataManifest(session_id="s1", product_hash="h1", eligible_for_training=False)
    manifest2 = TrainingDataManifest(session_id="s2", product_hash="h2", eligible_for_training=True)
    
    registry.register(manifest1)
    registry.register(manifest2)
    
    all_manifests = registry.get_all()
    assert len(all_manifests) == 2
    assert all_manifests[0]["session_id"] == "s1"
    assert all_manifests[1]["eligible_for_training"] is True
