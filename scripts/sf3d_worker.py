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
  --dry-run            Validate args only, no inference
"""

import sys
import json
import argparse
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
    parser.add_argument("--dry-run",            action="store_true")
    args = parser.parse_args()

    image_path  = Path(args.image)
    output_dir  = Path(args.output_dir)

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

    # ── real inference (only reached when sf3d is installed) ─────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    log.info("Using device: %s", device)

    try:
        import sf3d.run as sf3d_run   # adjust to actual SF3D API

        output_glb = output_dir / f"output.{args.output_format}"
        sf3d_run.run(
            image=str(image_path),
            output_dir=str(output_dir),
            device=device,
            texture_resolution=args.texture_resolution,
            remesh_option=args.remesh,
        )

        if not output_glb.exists():
            # SF3D may use a different output filename; find it
            candidates = list(output_dir.glob(f"*.{args.output_format}"))
            output_glb = candidates[0] if candidates else output_glb

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
