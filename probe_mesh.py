import trimesh
import time
from pathlib import Path

mesh_path = r"data\reconstructions\job_real_20260426_065607\recon\attempt_0_default\dense\meshed-poisson.ply"
if not Path(mesh_path).exists():
    print(f"File not found: {mesh_path}")
    exit(1)

print(f"Starting load of {mesh_path}...")
start = time.time()
try:
    m = trimesh.load(mesh_path)
    print(f"Loaded in {time.time() - start:.2f}s")
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    print(f"Vertices: {len(m.vertices)}")
    print(f"Faces: {len(m.faces)}")
except Exception as e:
    print(f"Error: {e}")
