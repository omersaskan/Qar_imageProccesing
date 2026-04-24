import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import logging

from .schema import (
    TrainingDataManifest,
    DeviceMetadata,
    CaptureTrainingMetrics,
    ReconstructionTrainingMetrics,
    ExportTrainingMetrics,
    TrainingLabels,
    TrainingDataPaths
)

logger = logging.getLogger("manifest_builder")

class TrainingManifestBuilder:
    def __init__(self, data_root: Path):
        self.data_root = data_root

    def _safe_load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read or parse {path}: {e}")
            return {}

    def _coerce_bbox(self, bbox_data: Any) -> Optional[List[float]]:
        """
        Coerces various bbox formats into a list of 3 floats [x, y, z].
        Supports:
        - [x, y, z]
        - {"x": 1, "y": 2, "z": 3}
        - {"width": 1, "height": 2, "depth": 3}
        - {"size": [1, 2, 3]}
        """
        if not bbox_data:
            return None
            
        if isinstance(bbox_data, list) and len(bbox_data) == 3:
            return [float(v) for v in bbox_data]
            
        if isinstance(bbox_data, dict):
            # Check for {"x", "y", "z"}
            if all(k in bbox_data for k in ("x", "y", "z")):
                return [float(bbox_data["x"]), float(bbox_data["y"]), float(bbox_data["z"])]
            # Check for {"width", "height", "depth"}
            if all(k in bbox_data for k in ("width", "height", "depth")):
                return [float(bbox_data["width"]), float(bbox_data["height"]), float(bbox_data["depth"])]
            # Check for {"size": [...]}
            if "size" in bbox_data and isinstance(bbox_data["size"], list) and len(bbox_data["size"]) == 3:
                return [float(v) for v in bbox_data["size"]]
                
        return None

    def build(self, session_id: str, product_id: str, eligible_for_training: bool = False, consent_status: str = "unknown") -> TrainingDataManifest:
        # Enforce eligible_for_training=False if consent is unknown
        if consent_status == "unknown":
            eligible_for_training = False
            
        product_hash = hashlib.sha256(product_id.encode()).hexdigest()[:16]
        
        manifest = TrainingDataManifest(
            session_id=session_id,
            product_id_hash=product_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
            consent_status=consent_status,
            eligible_for_training=eligible_for_training
        )
        
        captures_dir = self.data_root / "captures" / session_id
        reports_dir = captures_dir / "reports"
        
        # Load reports
        quality_report = self._safe_load_json(reports_dir / "quality_report.json")
        coverage_report = self._safe_load_json(reports_dir / "coverage_report.json")
        export_metrics = self._safe_load_json(reports_dir / "export_metrics.json")
        validation_report = self._safe_load_json(reports_dir / "validation_report.json")
        
        reconstructions_dir = self.data_root / "reconstructions" / f"job_{session_id}"
        reconstruction_audit = self._safe_load_json(reconstructions_dir / "reconstruction_audit.json")
        
        cleaned_dir = self.data_root / "cleaned" / f"job_{session_id}"
        cleanup_stats = self._safe_load_json(cleaned_dir / "cleanup_stats.json")
        
        session_data = self._safe_load_json(self.data_root / "sessions" / f"{session_id}.json")
        
        # Populate Device
        if "device" in session_data:
            dev = session_data["device"]
            manifest.device = DeviceMetadata(
                platform=dev.get("platform"),
                os=dev.get("os"),
                os_version=dev.get("os_version"),
                app_version=dev.get("app_version")
            )
            
        # Populate Capture
        manifest.capture = CaptureTrainingMetrics(
            duration_sec=float(quality_report.get("duration_sec", 0.0)),
            resolution=quality_report.get("resolution"),
            fps=float(quality_report.get("fps", 0.0)),
            frame_count=int(quality_report.get("frame_count", len(session_data.get("extracted_frames", [])))),
            bounding_box=self._coerce_bbox(export_metrics.get("bbox"))
        )
        
        # Populate Reconstruction
        best_attempt = None
        attempts = reconstruction_audit.get("attempts", [])
        if attempts:
            best_idx = reconstruction_audit.get("selected_best_index", -1)
            if 0 <= best_idx < len(attempts):
                best_attempt = attempts[best_idx]
            else:
                best_attempt = attempts[-1]
                
        if best_attempt:
            metrics = best_attempt.get("metrics", {})
            # Vertex/Face counts should come from explicit fields if available, otherwise 0
            # Density metrics are kept separate
            sparse_points = best_attempt.get("sparse_points", metrics.get("sparse_points", 0))
            dense_points_fused = best_attempt.get("dense_points_fused", metrics.get("dense_points_fused", 0))
            mesher_used = best_attempt.get("mesher_used", metrics.get("mesher_used"))

            manifest.reconstruction = ReconstructionTrainingMetrics(
                vertex_count=int(best_attempt.get("vertex_count", 0)),
                face_count=int(best_attempt.get("face_count", 0)),
                density_metrics={"sparse_points": sparse_points, "dense_points_fused": dense_points_fused},
                mesher_used=mesher_used
            )
            
        # Populate Export
        manifest.export = ExportTrainingMetrics(
            poly_count=int(export_metrics.get("face_count", 0)),
            texture_integrity_status=export_metrics.get("texture_integrity_status", "missing"),
            material_semantic_status=export_metrics.get("material_semantic_status", "geometry_only")
        )
        
        # Populate Paths
        asset_id = session_data.get("asset_id")
        manifest.asset_id = asset_id
        
        # Try to resolve paths safely
        clean_model = session_data.get("cleanup_mesh_path")
        orig_vid = session_data.get("source_video_path")
        frames = None
        if "extracted_frames" in session_data and session_data["extracted_frames"]:
            try:
                frames = str(Path(session_data["extracted_frames"][0]).parent)
            except:
                pass
                
        # Make paths relative
        def _rel(p_str: Optional[str]) -> Optional[str]:
            if not p_str: return None
            try:
                p = Path(p_str)
                return str(p.relative_to(self.data_root))
            except Exception:
                return p_str

        manifest.paths = TrainingDataPaths(
            clean_model=_rel(clean_model),
            original_video=_rel(orig_vid),
            frames_dir=_rel(frames)
        )
        
        # Populate Labels
        # 1. Labels from export_metrics
        status = export_metrics.get("material_semantic_status", "geometry_only")
        if status == "geometry_only":
            manifest.labels.asset_labels.append("geometry_only")
        elif status == "diffuse_textured":
            manifest.labels.asset_labels.append("draft_asset")
        elif status == "pbr_textured":
            manifest.labels.asset_labels.append("customer_ready")
            
        if export_metrics.get("texture_integrity_status") == "missing":
            manifest.labels.asset_labels.append("texture_missing")
            
        if not export_metrics.get("has_uv", True):
            manifest.labels.asset_labels.append("uv_missing")

        # 2. Labels from validation_report
        if validation_report:
            if validation_report.get("final_decision") == "pass":
                if "customer_ready" not in manifest.labels.asset_labels:
                    manifest.labels.asset_labels.append("customer_ready")
            else:
                contam = validation_report.get("contamination_report", "validation_failed")
                if isinstance(contam, dict):
                    manifest.labels.failure_reasons.append("validation_failed")
                elif isinstance(contam, str):
                    try:
                        from .label_taxonomy import FailureReasonLabel
                        if any(contam == item.value for item in FailureReasonLabel):
                            manifest.labels.failure_reasons.append(contam)
                        else:
                            manifest.labels.failure_reasons.append("validation_failed")
                    except Exception:
                        manifest.labels.failure_reasons.append("validation_failed")
                else:
                    manifest.labels.failure_reasons.append("validation_failed")

        # Write to both expected locations
        manifest_json = manifest.model_dump_json(indent=2)
        
        # 1. training_manifests directory
        manifests_dir = self.data_root / "training_manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(manifests_dir / f"{session_id}.json", "w", encoding="utf-8") as f:
                f.write(manifest_json)
        except Exception as e:
            logger.warning(f"Failed to write manifest to training_manifests: {e}")
            
        # 2. reports directory
        reports_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(reports_dir / "training_manifest.json", "w", encoding="utf-8") as f:
                f.write(manifest_json)
        except Exception as e:
            logger.warning(f"Failed to write manifest to reports: {e}")

        return manifest
