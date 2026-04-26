import fast_simplification
import numpy as np
import trimesh

mesh = trimesh.creation.uv_sphere(radius=1.0, count=[400, 400])
pre_faces = len(mesh.faces)
points = mesh.vertices.astype(np.float32)
faces = mesh.faces.astype(np.uint32)

target_faces = 150000
# Hypothesis: ratio is REDUCTION ratio
target_ratio = 1.0 - (target_faces / pre_faces)

print(f"Pre-faces: {pre_faces}")
print(f"Target faces: {target_faces}")
print(f"Target Ratio (Reduction): {target_ratio}")

new_vertices, new_faces = fast_simplification.simplify(points, faces, target_ratio)
print(f"Post-faces: {len(new_faces)}")
