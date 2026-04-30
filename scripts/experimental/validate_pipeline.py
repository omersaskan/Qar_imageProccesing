"""
Phase 3 Automated Pipeline Validation
======================================
Simulates the exact flow that worker.py would execute:
1. Adapter runs COLMAP → InterfaceCOLMAP → ReconstructMesh → TextureMesh
2. Runner creates manifest
3. GLBExporter exports textured GLB
4. Verifies all 7 artifact-level success criteria

Since COLMAP has already run, we only re-run the OpenMVS texturing stage
by removing previous OpenMVS artifacts and re-running the adapter's
OpenMVS section only (via a targeted test).
"""
import sys, os, json, time
sys.path.insert(0, r'c:\modelPlate')
sys.path.insert(0, r'c:\modelPlate\modules')

from pathlib import Path

job_dir = Path(r'c:\modelPlate\data\reconstructions\job_cap_1775661348')

# ── Step 0: Clean old OpenMVS artifacts for clean test ──
print("=== Step 0: Cleaning old OpenMVS artifacts ===")
for f in ['scene.mvs', 'scene_mesh.ply', 'scene_mesh.mvs',
          'scene_texture.ply', 'scene_texture.obj', 'scene_texture.mlp']:
    p = job_dir / f
    if p.exists():
        p.unlink()
        print(f"  Removed: {f}")

# Remove indexed texture atlases
for f in job_dir.glob("scene_texture*.png"):
    f.unlink()
    print(f"  Removed: {f.name}")

# Remove old depth maps
for f in job_dir.glob("depth*.dmap"):
    f.unlink()
    print(f"  Removed: {f.name}")

# Remove old umasks
for f in job_dir.glob("umask*.png"):
    f.unlink()
    print(f"  Removed: {f.name}")

print()

# ── Step 1: Run the adapter's run_reconstruction ──
# We replicate the adapter logic for just the OpenMVS stage
# since COLMAP already ran and produced meshed-poisson.ply
print("=== Step 1: Running OpenMVS texturing via adapter logic ===")
import subprocess, shutil

openmvs_dir = os.getenv("OPENMVS_BIN_PATH", r"C:\OpenMVS")
interface_colmap = Path(openmvs_dir) / "InterfaceCOLMAP.exe"
texture_mesh_bin = Path(openmvs_dir) / "TextureMesh.exe"
reconstruct_mesh_bin = Path(openmvs_dir) / "ReconstructMesh.exe"
res_level = "2"
dense_dir = Path("dense") / "0"

# Step 1a: InterfaceCOLMAP
print("  Running InterfaceCOLMAP...")
cmd = [str(interface_colmap), "-i", str(dense_dir), "-o", "scene.mvs"]
r = subprocess.run(cmd, cwd=str(job_dir), capture_output=True, text=True, errors="replace")
print(f"  InterfaceCOLMAP exit code: {r.returncode}")
if r.returncode != 0:
    print(f"  FAILED: {r.stderr[:500]}")
    sys.exit(1)

# Step 1b: ReconstructMesh
print("  Running ReconstructMesh...")
cmd = [str(reconstruct_mesh_bin), "scene.mvs"]
r = subprocess.run(cmd, cwd=str(job_dir), capture_output=True, text=True, errors="replace")
print(f"  ReconstructMesh exit code: {r.returncode}")
scene_mesh = job_dir / "scene_mesh.ply"
if r.returncode == 0 and scene_mesh.exists():
    print(f"  scene_mesh.ply: {scene_mesh.stat().st_size} bytes")
    texturing_mesh = "scene_mesh.ply"
else:
    print("  FAILED: falling back to Poisson mesh")
    texturing_mesh = str(os.path.relpath(job_dir / "dense" / "0" / "meshed-poisson.ply", job_dir))

# Step 1c: TextureMesh
print(f"  Running TextureMesh with mesh: {texturing_mesh}...")
cmd = [str(texture_mesh_bin), "scene.mvs", "-m", texturing_mesh, "-o", "scene_texture.ply", "--resolution-level", res_level]
r = subprocess.run(cmd, cwd=str(job_dir), capture_output=True, text=True, errors="replace")
print(f"  TextureMesh exit code: {r.returncode}")

# Check for nan residual (informational)
if "nan residual" in r.stdout:
    print("  Note: nan residual present but process continued")

print()

# ── Step 2: Verify artifact files ──
print("=== Step 2: Artifact verification ===")
criteria = {}

# Criterion 1: TextureMesh exit code 0
criteria["1_texturemesh_exit_0"] = r.returncode == 0
print(f"  [{'PASS' if criteria['1_texturemesh_exit_0'] else 'FAIL'}] TextureMesh exit code 0: {r.returncode}")

# Criterion 2: scene_texture mesh exists
textured_mesh_path = None
for candidate in ["scene_texture.ply", "scene_texture.obj"]:
    p = job_dir / candidate
    if p.exists() and p.stat().st_size > 0:
        textured_mesh_path = p
        break

criteria["2_textured_mesh_exists"] = textured_mesh_path is not None
if textured_mesh_path:
    print(f"  [PASS] Textured mesh: {textured_mesh_path.name} ({textured_mesh_path.stat().st_size} bytes)")
else:
    print("  [FAIL] No textured mesh found")

# Criterion 3: Texture atlas exists
texture_atlas = None
for candidate in sorted(job_dir.glob("scene_texture*.png")):
    if candidate.stat().st_size > 0:
        texture_atlas = candidate
        break

criteria["3_texture_atlas_exists"] = texture_atlas is not None
if texture_atlas:
    print(f"  [PASS] Texture atlas: {texture_atlas.name} ({texture_atlas.stat().st_size} bytes)")
else:
    print("  [FAIL] No texture atlas found")

# ── Step 3: Update manifest ──
print("\n=== Step 3: Updating manifest ===")
manifest_path = job_dir / "manifest.json"
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

if textured_mesh_path:
    manifest["mesh_path"] = str(textured_mesh_path)
if texture_atlas:
    manifest["texture_path"] = str(texture_atlas)

manifest["mesh_metadata"]["has_texture"] = bool(textured_mesh_path and texture_atlas)

# Get real counts
if textured_mesh_path:
    import trimesh
    tmesh = trimesh.load(str(textured_mesh_path))
    manifest["mesh_metadata"]["vertex_count"] = len(tmesh.vertices)
    manifest["mesh_metadata"]["face_count"] = len(tmesh.faces)

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

# Criterion 4: Manifest points to real texture
criteria["4_manifest_real_texture"] = "dummy_texture" not in manifest.get("texture_path", "dummy_texture")
print(f"  [{'PASS' if criteria['4_manifest_real_texture'] else 'FAIL'}] Manifest texture: {manifest.get('texture_path', 'N/A')}")

# Criterion 5: has_texture == true
criteria["5_has_texture_true"] = manifest["mesh_metadata"]["has_texture"] == True
print(f"  [{'PASS' if criteria['5_has_texture_true'] else 'FAIL'}] has_texture: {manifest['mesh_metadata']['has_texture']}")
print(f"  Vertex count: {manifest['mesh_metadata']['vertex_count']}")
print(f"  Face count: {manifest['mesh_metadata']['face_count']}")

# ── Step 4: GLB Export ──
print("\n=== Step 4: GLB Export ===")
from modules.export_pipeline.glb_exporter import GLBExporter

blobs_dir = Path(r'c:\modelPlate\data\registry\blobs')
blob_path = blobs_dir / "22_1775663894.glb"

if textured_mesh_path:
    exporter = GLBExporter()
    try:
        result = exporter.export(
            mesh_path=str(textured_mesh_path),
            output_path=str(blob_path),
            profile_name="standard"
        )
        criteria["6_glb_export_success"] = True
        print(f"  [PASS] GLB exported: {result['filesize']} bytes")
        print(f"    Vertices: {result['vertex_count']}, Faces: {result['face_count']}")
    except Exception as e:
        criteria["6_glb_export_success"] = False
        print(f"  [FAIL] GLB export failed: {e}")
else:
    criteria["6_glb_export_success"] = False
    print("  [FAIL] No textured mesh to export")

# Criterion 7: GLB has texture (round-trip check)
print("\n=== Step 5: GLB Round-trip Verification ===")
import trimesh
try:
    loaded = trimesh.load(str(blob_path))
    has_texture = False
    if isinstance(loaded, trimesh.Scene):
        for name, geom in loaded.geometry.items():
            if isinstance(geom.visual, trimesh.visual.TextureVisuals):
                has_texture = True
    elif isinstance(loaded, trimesh.Trimesh):
        if isinstance(loaded.visual, trimesh.visual.TextureVisuals):
            has_texture = True
    criteria["7_dashboard_textured"] = has_texture
    print(f"  [{'PASS' if has_texture else 'FAIL'}] GLB has TextureVisuals: {has_texture}")
except Exception as e:
    criteria["7_dashboard_textured"] = False
    print(f"  [FAIL] GLB round-trip failed: {e}")

# ── Final Report ──
print("\n" + "=" * 60)
print("PHASE 3 AUTOMATED VALIDATION REPORT")
print("=" * 60)
all_pass = all(criteria.values())
for key, val in criteria.items():
    status = "✅ PASS" if val else "❌ FAIL"
    label = key.split("_", 1)[1].replace("_", " ").title()
    print(f"  {status}  {label}")

print()
if all_pass:
    print("🎉 ALL 7 CRITERIA MET — AUTOMATED PIPELINE SUCCESS")
else:
    failed = [k for k, v in criteria.items() if not v]
    print(f"⚠️  {len(failed)} criteria failed: {', '.join(failed)}")
    print("Infrastructure improved, but sample-level artifact generation is still incomplete.")
