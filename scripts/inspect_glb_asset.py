import os
import sys
import json
import argparse
from pathlib import Path
import trimesh
from modules.export_pipeline.glb_exporter import GLBExporter
from modules.qa_validation.texture_quality import TextureQualityAnalyzer
from modules.operations.settings import settings

def inspect_glb(glb_path: str):
    if not os.path.exists(glb_path):
        print(f"Error: File not found: {glb_path}")
        return

    exporter = GLBExporter()
    inspection = exporter.inspect_exported_asset(glb_path)
    
    # Load for texture analysis
    loaded = trimesh.load(glb_path, force="scene")
    
    # Extract textures
    textures = []
    for geom in loaded.geometry.values():
        if isinstance(geom, trimesh.Trimesh) and hasattr(geom.visual, 'material'):
            mat = geom.visual.material
            if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                textures.append(mat.baseColorTexture)
            elif hasattr(mat, 'image') and mat.image is not None:
                textures.append(mat.image)

    # Analyze textures
    analyzer = TextureQualityAnalyzer()
    texture_reports = []
    
    # trimesh might extract multiple copies of the same image, so we deduplicate by hash if possible
    # or just analyze each.
    
    import numpy as np
    from PIL import Image
    
    highest_black_ratio = 0.0
    all_success = True
    failure_reasons = []

    for i, tex in enumerate(textures):
        # Convert PIL to numpy BGR for CV2
        img_np = np.array(tex.convert("RGB"))
        img_bgr = img_np[:, :, ::-1].copy() # RGB to BGR
        
        report = analyzer.analyze_image(img_bgr)
        report["index"] = i
        texture_reports.append(report)
        
        highest_black_ratio = max(highest_black_ratio, report.get("black_pixel_ratio", 0.0))
        if report.get("texture_quality_status") == "fail":
            all_success = False
            failure_reasons.extend(report.get("texture_quality_reasons", []))

    # Strict Delivery Gate Check
    delivery_ready = True
    gate_failures = []

    if not inspection["all_primitives_have_position"]:
        delivery_ready = False
        gate_failures.append("MISSING_POSITION")
    if not inspection["all_primitives_have_normal"]:
        delivery_ready = False
        gate_failures.append("MISSING_NORMAL")
    if not inspection["all_textured_primitives_have_texcoord_0"]:
        delivery_ready = False
        gate_failures.append("MISSING_TEXCOORD_0")
    if inspection["material_count"] == 0:
        delivery_ready = False
        gate_failures.append("ZERO_MATERIALS")
    if inspection["texture_count"] == 0:
        delivery_ready = False
        gate_failures.append("ZERO_TEXTURES")
    if not all_success:
        delivery_ready = False
        gate_failures.append("TEXTURE_QUALITY_FAIL")
    
    # Final vertex/face counts from trimesh inspection
    final_vertex_count = inspection["final_vertex_count"]
    final_face_count = inspection["final_face_count"]

    result = {
        "glb_path": glb_path,
        "final_vertex_count": final_vertex_count,
        "final_face_count": final_face_count,
        "material_count": inspection["material_count"],
        "texture_count": inspection["texture_count"],
        "has_texcoord_0": inspection["all_textured_primitives_have_texcoord_0"],
        "texture_applied": inspection["texture_count"] > 0,
        "texture_quality_reports": texture_reports,
        "highest_black_pixel_ratio": highest_black_ratio,
        "delivery_ready": delivery_ready,
        "failure_reasons": gate_failures + failure_reasons,
    }

    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect GLB asset for AR delivery readiness.")
    parser.add_argument("glb_path", help="Path to the GLB file")
    args = parser.parse_args()
    inspect_glb(args.glb_path)
