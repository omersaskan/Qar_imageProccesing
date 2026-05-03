"""
SF3D Isolated Worker Script.

Run by SF3DProvider via subprocess using the SF3D venv Python.
Prints exactly ONE JSON object to stdout. All logs go to stderr.

Usage:
  python scripts/sf3d_worker.py --image <path> --output-dir <dir> [options]
  python scripts/sf3d_worker.py --image <path> --output-dir <dir> --dry-run

Args:
  --image              Input image path (prepared ai3d_input.png recommended)
  --output-dir         Directory to write output GLB and preview
  --device             cpu | cuda | auto  (default: auto)
  --input-size         Input resolution (default: 512)
  --texture-resolution Texture resolution (default: 1024)
  --remesh             none | quad | triangle  (default: none)
  --output-format      glb | obj  (default: glb)
  --pretrained-model   HuggingFace model id or local path
                       (default: stabilityai/stable-fast-3d)
  --foreground-ratio   Foreground ratio for resize (default: 0.85)
  --no-remove-bg       Skip rembg background removal (use prepared alpha)
  --dry-run            Validate args only, no inference
"""

import sys
import json
import argparse
import contextlib
import logging
from pathlib import Path

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="[sf3d_worker] %(levelname)s %(message)s")
log = logging.getLogger("sf3d_worker")


def _out(data: dict):
    print(json.dumps(data), flush=True)


def main():
    parser = argparse.ArgumentParser(description="SF3D Isolated Worker")
    parser.add_argument("--image",              required=True)
    parser.add_argument("--output-dir",         required=True)
    parser.add_argument("--device",             default="auto")
    parser.add_argument("--input-size",         type=int, default=512)
    parser.add_argument("--texture-resolution", type=int, default=1024)
    parser.add_argument("--remesh",             default="none",
                        choices=["none", "quad", "triangle"])
    parser.add_argument("--output-format",      default="glb",
                        choices=["glb", "obj"])
    parser.add_argument("--pretrained-model",   default="stabilityai/stable-fast-3d")
    parser.add_argument("--foreground-ratio",   type=float, default=0.85)
    parser.add_argument("--no-remove-bg",       action="store_true")
    parser.add_argument("--dry-run",            action="store_true")
    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)

    # ── dry-run: validate paths and return early ──────────────────────────────
    if args.dry_run:
        issues = []
        if not image_path.exists():
            issues.append(f"image not found: {image_path}")
        if issues:
            _out({"status": "failed", "dry_run": True, "issues": issues})
            sys.exit(1)
        _out({"status": "ok", "dry_run": True})
        return

    # ── check image path ──────────────────────────────────────────────────────
    if not image_path.exists():
        _out({
            "status": "failed",
            "error_code": "input_image_missing",
            "message": f"Input image not found: {image_path}",
            "warnings": [],
        })
        sys.exit(1)

    # ── attempt SF3D import ───────────────────────────────────────────────────
    try:
        import sf3d  # noqa: F401 — lazy import, only in worker
    except ImportError as exc:
        log.warning("sf3d package not importable: %s", exc)
        _out({
            "status": "unavailable",
            "error_code": "sf3d_package_missing",
            "message": (
                "sf3d package is not installed in this Python environment. "
                "Install it under external/stable-fast-3d/.venv_sf3d. "
                f"Details: {exc}"
            ),
            "warnings": ["sf3d_package_missing"],
        })
        sys.exit(0)   # exit 0 — unavailable is not a crash

    # ── real inference (only reached when sf3d is installed) ──────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── resolve device ────────────────────────────────────────────────────────
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    log.info("Using device: %s", device)

    try:
        import torch
        from PIL import Image
        from sf3d.system import SF3D
        from sf3d.utils import remove_background, resize_foreground

        # ── load model ───────────────────────────────────────────────────────
        log.info("Loading SF3D model from: %s", args.pretrained_model)
        try:
            model = SF3D.from_pretrained(
                args.pretrained_model,
                config_name="config.yaml",
                weight_name="model.safetensors",
            )
        except Exception as auth_exc:
            _auth_msg = str(auth_exc)
            if "401" in _auth_msg or "gated" in _auth_msg.lower() or "GatedRepoError" in type(auth_exc).__name__:
                _out({
                    "status": "unavailable",
                    "error_code": "sf3d_model_auth_required",
                    "message": (
                        "stabilityai/stable-fast-3d is a gated model. "
                        "Accept the license at https://huggingface.co/stabilityai/stable-fast-3d "
                        "then set HF_TOKEN env var or run: huggingface-cli login"
                    ),
                    "warnings": ["sf3d_model_auth_required"],
                })
                sys.exit(0)   # unavailable — not a crash
            raise  # re-raise unexpected errors
        model.to(device)
        model.eval()
        log.info("Model loaded on %s", device)

        # ── load and preprocess image ────────────────────────────────────────
        img = Image.open(str(image_path)).convert("RGBA")

        if not args.no_remove_bg:
            import rembg
            log.info("Removing background with rembg...")
            rembg_session = rembg.new_session()
            img = remove_background(img, rembg_session)

        img = resize_foreground(img, args.foreground_ratio)
        log.info("Image preprocessed: size=%s mode=%s", img.size, img.mode)

        # ── inference ────────────────────────────────────────────────────────
        log.info("Running SF3D inference...")
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device == "cuda"
            else contextlib.nullcontext()
        )

        peak_mem_mb = None
        # Redirect Python stdout → stderr during model.run_image so that
        # native remesh progress lines ("After Remesh N M") go to stderr
        # and cannot contaminate the single JSON object we write to stdout.
        with torch.no_grad(), autocast_ctx, contextlib.redirect_stdout(sys.stderr):
            mesh, glob_dict = model.run_image(
                [img],
                bake_resolution=args.texture_resolution,
                remesh=args.remesh,
            )

        if device == "cuda" and torch.cuda.is_available():
            peak_mem_mb = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
            log.info("Peak GPU memory: %.1f MB", peak_mem_mb)

        # ── export GLB ───────────────────────────────────────────────────────
        output_glb = output_dir / f"output.{args.output_format}"
        # Redirect stdout during export too — trimesh/gpytoolbox may print
        with contextlib.redirect_stdout(sys.stderr):
            mesh.export(str(output_glb), include_normals=True)
        log.info("GLB exported: %s  size=%d bytes", output_glb, output_glb.stat().st_size)

        if not output_glb.exists() or output_glb.stat().st_size == 0:
            _out({
                "status": "failed",
                "error_code": "sf3d_output_empty",
                "message": f"Output GLB missing or empty: {output_glb}",
                "warnings": [],
            })
            sys.exit(1)

        _out({
            "status": "ok",
            "output_path": str(output_glb),
            "output_format": args.output_format,
            "model_name": "stable-fast-3d",
            "preview_image_path": None,
            "warnings": ["ai_generated_not_true_scan"],
            "metadata": {
                "device": device,
                "input_size": args.input_size,
                "texture_resolution": args.texture_resolution,
                "remesh": args.remesh,
                "pretrained_model": args.pretrained_model,
                "foreground_ratio": args.foreground_ratio,
                "peak_mem_mb": peak_mem_mb,
                "output_size_bytes": output_glb.stat().st_size,
            },
        })

    except Exception as exc:
        log.error("SF3D inference failed: %s", exc, exc_info=True)
        _out({
            "status": "failed",
            "error_code": "sf3d_inference_error",
            "message": str(exc),
            "warnings": [],
        })
        sys.exit(1)


if __name__ == "__main__":
    main()
