import os
import sys
import json
import time
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root and modules to sys.path
sys.path.insert(0, r'c:\modelPlate')
sys.path.insert(0, r'c:\modelPlate\modules')

from modules.export_pipeline.glb_exporter import GLBExporter

# Failure Buckets
BUCKET_NONE = "None (Success)"
BUCKET_RESOLUTION = "1. Path/File Resolution"
BUCKET_INCONSISTENCY = "2. Workspace Inconsistency / Ingestion Failure"
BUCKET_RESOURCE = "3. Memory/Resource"
BUCKET_TOPOLOGY = "4. Mesh Topology/Geometry"
BUCKET_NUMERICAL = "5. Seam-Leveling/Numerical"
BUCKET_INTEGRATION = "6. Manifest/Export Integration"
BUCKET_RENDERING = "7. Dashboard/Rendering"
BUCKET_UNKNOWN = "8. Unknown"
BUCKET_MISSING_ARTIFACT = "9. Missing Artifact"

class ValidationResult:
    def __init__(self, job_id: str, sample_type: str):
        self.job_id = job_id
        self.sample_type = sample_type
        self.success = False
        self.failure_bucket = BUCKET_NONE
        self.failure_reason = ""
        self.metrics = {
            "image_count": 0,
            "vertex_count": 0,
            "face_count": 0,
            "texture_atlas_size": 0,
            "peak_ram_mb": 0,
            "runtime_sec": 0.0
        }
        self.criteria_met = {
            "texturemesh_exit_0": False,
            "mesh_exists": False,
            "atlas_exists": False,
            "manifest_textured": False,
            "has_texture_true": False,
            "glb_export_success": False,
            "dashboard_compatible": False
        }

def validate_sample(job_id: str, sample_type: str, force_poisson: bool = False) -> ValidationResult:
    start_time = time.time()
    result = ValidationResult(job_id, sample_type)
    
    job_dir = Path(r'c:\modelPlate\data\reconstructions') / job_id
    if not job_dir.exists():
        result.failure_bucket = BUCKET_MISSING_ARTIFACT
        result.failure_reason = f"Job directory {job_id} not found."
        return result

    try:
        # Pre-execution: Clean old OpenMVS artifacts
        print(f"\n--- Validating {job_id} ({sample_type.upper()}) ---")
        for f in ['scene_texture.ply', 'scene_texture.obj', 'scene_texture.mlp', 'scene_mesh.ply', 'scene.mvs']:
             p = job_dir / f
             if p.exists(): p.unlink()
        for f in job_dir.glob("scene_texture*.png"): f.unlink()

        openmvs_dir = os.getenv("OPENMVS_BIN_PATH", r"C:\OpenMVS")
        texture_mesh_bin = Path(openmvs_dir) / "TextureMesh.exe"
        reconstruct_mesh_bin = Path(openmvs_dir) / "ReconstructMesh.exe"
        interface_colmap = Path(openmvs_dir) / "InterfaceCOLMAP.exe"

        # 1. InterfaceCOLMAP
        cmd = [str(interface_colmap), "-i", "dense/0", "-o", "scene.mvs"]
        r = subprocess.run(cmd, cwd=str(job_dir), capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            result.failure_bucket = BUCKET_INCONSISTENCY
            result.failure_reason = f"InterfaceCOLMAP failed: {r.stderr[:200]}"
            return result

        # 2. Choose Mesh
        texturing_mesh = None
        if not force_poisson:
            # Try Delaunay
            cmd = [str(reconstruct_mesh_bin), "scene.mvs"]
            r = subprocess.run(cmd, cwd=str(job_dir), capture_output=True, text=True, errors="replace")
            if r.returncode == 0 and (job_dir / "scene_mesh.ply").exists():
                texturing_mesh = "scene_mesh.ply"
            else:
                print(f"  ReconstructMesh failed or forced fallback.")
        
        if texturing_mesh is None:
            # Fallback to Poisson (or the only available)
            # Find any .ply in dense/0
            poisson_cands = list((job_dir / "dense" / "0").glob("*.ply"))
            if poisson_cands:
                texturing_mesh = str(os.path.relpath(poisson_cands[0], job_dir))
            else:
                result.failure_bucket = BUCKET_MISSING_ARTIFACT
                result.failure_reason = "No candidate mesh found in dense/0"
                return result

        # 3. TextureMesh
        cmd = [str(texture_mesh_bin), "scene.mvs", "-m", texturing_mesh, "-o", "scene_texture.ply", "--resolution-level", "2"]
        r = subprocess.run(cmd, cwd=str(job_dir), capture_output=True, text=True, errors="replace")
        
        result.criteria_met["texturemesh_exit_0"] = (r.returncode == 0)
        if r.returncode != 0:
            if "nan residual" in r.stdout or "nan residual" in r.stderr:
                result.failure_bucket = BUCKET_NUMERICAL
            else:
                result.failure_bucket = BUCKET_TOPOLOGY
            result.failure_reason = f"TextureMesh failed with code {r.returncode}"
            return result

        # 4. Artifact Check
        textured_mesh = job_dir / "scene_texture.ply"
        texture_atlas = None
        for cand in sorted(job_dir.glob("scene_texture*.png")):
            if cand.stat().st_size > 0:
                texture_atlas = cand
                break
        
        result.criteria_met["mesh_exists"] = textured_mesh.exists()
        result.criteria_met["atlas_exists"] = texture_atlas is not None
        
        if not textured_mesh.exists():
            result.failure_bucket = BUCKET_INTEGRATION
            result.failure_reason = "TextureMesh exited 0 but scene_texture.ply missing"
            return result

        # 5. Manifest Check
        # (Assuming the adapter would do this, we verify the logic)
        manifest_path = job_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
            # Update it for validation
            manifest_data["mesh_path"] = str(textured_mesh)
            manifest_data["texture_path"] = str(texture_atlas) if texture_atlas else ""
            manifest_data["mesh_metadata"]["has_texture"] = (texture_atlas is not None)
            
            # Metrics
            import trimesh
            tm = trimesh.load(str(textured_mesh))
            result.metrics["vertex_count"] = len(tm.vertices)
            result.metrics["face_count"] = len(tm.faces)
            manifest_data["mesh_metadata"]["vertex_count"] = result.metrics["vertex_count"]
            manifest_data["mesh_metadata"]["face_count"] = result.metrics["face_count"]
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest_data, f, indent=2)
            
            result.criteria_met["manifest_textured"] = ("dummy" not in manifest_data["texture_path"])
            result.criteria_met["has_texture_true"] = manifest_data["mesh_metadata"]["has_texture"]

        # 6. GLB Export
        blobs_dir = Path("data/registry/blobs")
        blobs_dir.mkdir(parents=True, exist_ok=True)
        # Use a validation-specific export path to avoid collision if desired, 
        # but here we follow the standard logic.
        output_glb = blobs_dir / f"val_{job_id}.glb"
        exporter = GLBExporter()
        try:
            exp_res = exporter.export(str(textured_mesh), str(output_glb), "standard")
            result.criteria_met["glb_export_success"] = True
            
            # 7. Dashboard Check (Round-trip)
            loaded = trimesh.load(str(output_glb))
            has_uv = False
            if isinstance(loaded, trimesh.Scene):
                for g in loaded.geometry.values():
                    if hasattr(g, 'visual') and hasattr(g.visual, 'uv'):
                        has_uv = True
            elif hasattr(loaded, 'visual') and hasattr(loaded.visual, 'uv'):
                has_uv = True
            result.criteria_met["dashboard_compatible"] = has_uv
        except Exception as e:
            result.failure_bucket = BUCKET_RENDERING
            result.failure_reason = f"GLB Export/Load failed: {e}"
            return result

        result.success = all(result.criteria_met.values())
        if not result.success:
            result.failure_bucket = BUCKET_INTEGRATION
            result.failure_reason = f"Criteria failed: {[k for k,v in result.criteria_met.items() if not v]}"

    except Exception as e:
        result.failure_bucket = BUCKET_UNKNOWN
        result.failure_reason = str(e)
    
    # Extract image count for reporting
    img_dir = job_dir / "images"
    if img_dir.exists():
        result.metrics["image_count"] = len(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    
    result.metrics["runtime_sec"] = time.time() - start_time
    return result

def print_report(results: List[ValidationResult]):
    print("\n" + "="*80)
    print(f"{'Job ID':<20} | {'Type':<10} | {'Outcome':<10} | {'Failure Bucket':<25}")
    print("-"*80)
    for r in results:
        outcome = "[PASS]" if r.success else "[FAIL]"
        bucket = r.failure_bucket if not r.success else "-"
        print(f"{r.job_id:<20} | {r.sample_type:<10} | {outcome:<10} | {bucket:<25}")
        if not r.success:
            print(f"   Reason: {r.failure_reason}")
    print("="*80)

if __name__ == "__main__":
    matrix = []
    
    # Run the matrix
    # 1. Real - Reference
    matrix.append(validate_sample("job_cap_1775661348", "Real"))
    
    # 2. Real - Discovery (Point Cloud)
    matrix.append(validate_sample("job_cap_1775612356", "Real"))
    
    # 3. Derived - True Sparsity (17/34 Frames, COLMAP Re-run)
    # Selected every 2nd frame (0, 2, 4...) for uniform coverage.
    matrix.append(validate_sample("S-True-Sparse", "Derived"))
    
    # 4. Derived - Topology (Forced Poisson)
    matrix.append(validate_sample("S1-Topology", "Derived", force_poisson=True))
    
    print_report(matrix)
