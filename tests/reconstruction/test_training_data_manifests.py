import json
from pathlib import Path
from modules.training_data.manifest_builder import TrainingManifestBuilder
from modules.training_data.dataset_registry import DatasetRegistry
from modules.training_data.schema import TrainingDataManifest, CaptureTrainingMetrics
from modules.training_data.label_taxonomy import AssetLabel, FailureReasonLabel
import pytest
from datetime import datetime, timezone

def test_training_manifest_builder_with_missing_reports(tmp_path: Path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    
    session_id = "test_session_1"
    product_id = "prod_x"
    
    # Empty session
    builder = TrainingManifestBuilder(data_root=data_root)
    manifest = builder.build(session_id, product_id, eligible_for_training=True, consent_status="unknown")
    
    assert manifest.session_id == session_id
    # Unknown consent must force eligible_for_training = False
    assert manifest.eligible_for_training is False
    assert manifest.product_id_hash != product_id  # hashed
    assert manifest.export.poly_count == 0
    assert manifest.schema_version == "1.0"
    
    # Check that it writes to both locations
    assert (data_root / "training_manifests" / f"{session_id}.json").exists()
    assert (data_root / "captures" / session_id / "reports" / "training_manifest.json").exists()

def test_dataset_registry_latest_wins(tmp_path: Path):
    registry_file = tmp_path / "data" / "training_registry" / "index.jsonl"
    registry = DatasetRegistry(registry_file)
    
    # Create two manifests for same session ID
    manifest1 = TrainingDataManifest(
        session_id="s1", 
        product_id_hash="h1", 
        created_at=datetime.now(timezone.utc).isoformat(),
        eligible_for_training=False,
        consent_status="unknown"
    )
    
    manifest2 = TrainingDataManifest(
        session_id="s1", 
        product_id_hash="h1", 
        created_at=datetime.now(timezone.utc).isoformat(),
        eligible_for_training=True,
        consent_status="granted"
    )
    
    registry.register(manifest1)
    registry.register(manifest2)
    
    all_manifests = registry.get_all()
    assert len(all_manifests) == 1
    assert all_manifests[0]["session_id"] == "s1"
    assert all_manifests[0]["eligible_for_training"] is True
    assert all_manifests[0]["consent_status"] == "granted"

def test_label_taxonomy():
    assert AssetLabel.CUSTOMER_READY == "customer_ready"
    assert FailureReasonLabel.CAPTURE_BLURRY == "capture_blurry"
    
def test_manifest_schema_validation():
    # Test valid nested
    m = TrainingDataManifest(
        session_id="s1",
        product_id_hash="h",
        created_at="2026-04-24T00:00:00Z"
    )
    m.capture = CaptureTrainingMetrics(duration_sec=10.0, fps=30.0, frame_count=300)
    assert m.capture.duration_sec == 10.0
