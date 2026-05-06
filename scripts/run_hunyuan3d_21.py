#!/usr/bin/env python3
"""
Standalone runner for Tencent Hunyuan3D-2.1 shape-only inference.
Executed as a subprocess by Hunyuan3D21Provider.

Contract:
  - Always writes a JSON manifest to --manifest-out before exiting.
  - Exits 0; manifest status field ("ok" / "failed") carries the result.
  - Texture mode (shape_and_texture) is NOT implemented here.
  - Never imports main-process modules; runs in Hunyuan's own Python env.
  - PYTHONPATH is set by the provider to point at the Hunyuan repo root.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Hunyuan3D-2.1 Shape-Only Runner")
    parser.add_argument("--input-image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["shape_only", "shape_and_texture"],
                        default="shape_only")
    parser.add_argument("--model-path", default="tencent/Hunyuan3D-2.1")
    parser.add_argument("--subfolder", default="hunyuan3d-dit-v2-1")
    parser.add_argument("--texgen-model-path", default="tencent/Hunyuan3D-2.1")
    parser.add_argument("--texture-resolution", type=int, default=512)
    parser.add_argument("--max-num-view", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--low-vram-mode", action="store_true")
    parser.add_argument("--mock-runner", action="store_true",
                        help="Test-only: generate dummy output without loading any model")
    parser.add_argument("--manifest-out", required=True)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_manifest(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _sanitize_error(exc: Exception) -> str:
    """Return a safe, non-leaking error string for the manifest."""
    msg = str(exc)
    t = type(exc).__name__
    if "out of memory" in msg.lower() or "CUDA out of memory" in msg:
        return "cuda_oom"
    if "ModuleNotFoundError" in t or "ImportError" in t:
        return f"import_error:{t}"
    first_line = msg.splitlines()[0] if msg else ""
    return f"{t}: {first_line[:120]}"


def _alpha_is_all_opaque(image) -> bool:
    """Return True if every pixel in the alpha channel is 255 (no mask present)."""
    try:
        from PIL import ImageStat
        stat = ImageStat.Stat(image.split()[3])
        return stat.extrema[0][0] == 255
    except Exception:
        return True  # conservative: assume no mask → remove background


def _remove_background(image):
    """
    Apply BackgroundRemover from hy3dshape.rembg.
    Returns RGBA image.  Falls back to original on any error.
    """
    try:
        from hy3dshape.rembg import BackgroundRemover
        remover = BackgroundRemover()
        return remover(image.convert("RGB")).convert("RGBA")
    except Exception:
        return image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # The provider sets PYTHONPATH to the Hunyuan repo root.  Python processes
    # PYTHONPATH automatically at startup, but on some Windows configurations
    # it may not be reflected in sys.path until we do so explicitly.
    _pythonpath = os.environ.get("PYTHONPATH", "")
    if _pythonpath and _pythonpath not in sys.path:
        sys.path.insert(0, _pythonpath)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "status": "failed",
        "output_glb_path": None,
        "warnings": [],
        "error": None,
        "peak_mem_mb": None,
    }

    # ── Mock mode (test-only) ─────────────────────────────────────────────────
    if args.mock_runner:
        output_glb = output_dir / "output.glb"
        # Minimal syntactically plausible GLB header (12-byte magic+version+length)
        output_glb.write_bytes(b"glTF\x02\x00\x00\x00\x14\x00\x00\x00\x00\x00\x00\x00")
        manifest.update({
            "status": "ok",
            "output_glb_path": str(output_glb.resolve()),
            "warnings": ["mock_output_used"],
            "peak_mem_mb": 128,
            "mode": "shape_only",
            "model_path": args.model_path,
            "device": args.device,
        })
        write_manifest(args.manifest_out, manifest)
        sys.exit(0)

    # ── Texture mode guard ────────────────────────────────────────────────────
    if args.mode != "shape_only":
        manifest["error"] = f"mode_not_implemented:{args.mode}"
        write_manifest(args.manifest_out, manifest)
        sys.exit(0)

    # ── Import Hunyuan + PIL ──────────────────────────────────────────────────
    try:
        from PIL import Image
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    except ImportError as exc:
        manifest["error"] = _sanitize_error(exc)
        write_manifest(args.manifest_out, manifest)
        sys.exit(0)

    # ── Real shape-only inference ─────────────────────────────────────────────
    try:
        # Load pipeline from pretrained weights
        pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            args.model_path,
            subfolder=args.subfolder,
        )

        # Low-VRAM mode: prefer fp16 if the pipeline supports it
        if args.low_vram_mode:
            try:
                import torch
                pipeline_shapegen = pipeline_shapegen.to(torch.float16)
            except Exception:
                pass  # fall through to default precision

        pipeline_shapegen = pipeline_shapegen.to(args.device)

        # Open input image
        image = Image.open(args.input_image).convert("RGBA")

        # Apply background removal when the image carries no alpha mask
        if _alpha_is_all_opaque(image):
            image = _remove_background(image)

        # Reset VRAM peak counter before inference
        if args.device == "cuda":
            try:
                import torch
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        # Shape generation
        mesh = pipeline_shapegen(image=image)[0]

        # Capture peak VRAM
        peak_mem_mb = None
        if args.device == "cuda":
            try:
                import torch
                peak_mem_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
            except Exception:
                pass

        # Export GLB
        output_glb_path = output_dir / "output.glb"
        mesh.export(str(output_glb_path))

        if not output_glb_path.exists():
            manifest["error"] = "glb_not_written_after_export"
            write_manifest(args.manifest_out, manifest)
            sys.exit(0)

        manifest.update({
            "status": "ok",
            "output_glb_path": str(output_glb_path.resolve()),
            "warnings": ["ai_generated_not_true_scan"],
            "error": None,
            "peak_mem_mb": peak_mem_mb,
            "mode": "shape_only",
            "model_path": args.model_path,
            "device": args.device,
        })
        write_manifest(args.manifest_out, manifest)
        sys.exit(0)

    except Exception as exc:
        manifest["error"] = _sanitize_error(exc)
        write_manifest(args.manifest_out, manifest)
        sys.exit(0)


if __name__ == "__main__":
    main()
