import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2


def read_ply_vertex_count(ply_path: Path) -> int:
    if not ply_path.exists():
        return 0

    try:
        with open(ply_path, "rb") as f:
            for _ in range(80):
                line = f.readline().decode("ascii", errors="ignore").strip()
                if line.startswith("element vertex"):
                    return int(line.split()[-1])
                if line == "end_header":
                    break
    except Exception:
        return 0

    return 0


def read_image_shape(path: Path):
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None
    return list(arr.shape[:2])


def inspect_mask_alignment(images_dir: Path, masks_dir: Path) -> dict:
    images = sorted(
        list(images_dir.glob("*.jpg"))
        + list(images_dir.glob("*.jpeg"))
        + list(images_dir.glob("*.png"))
    )

    report = {
        "mask_path": str(masks_dir),
        "mask_path_exists": masks_dir.exists(),
        "dense_images_path": str(images_dir),
        "dense_images_path_exists": images_dir.exists(),
        "image_count": len(images),
        "mask_count": len(list(masks_dir.glob("*.png"))) if masks_dir.exists() else 0,
        "exact_filename_match_count": 0,
        "stem_filename_match_count": 0,
        "dimension_match_count": 0,
        "missing_exact_samples": [],
        "dimension_mismatch_samples": [],
    }

    if not masks_dir.exists() or not images:
        return report

    for img in images:
        exact_mask = masks_dir / f"{img.name}.png"
        stem_mask = masks_dir / f"{img.stem}.png"

        if exact_mask.exists():
            report["exact_filename_match_count"] += 1

            img_shape = read_image_shape(img)
            mask_shape = read_image_shape(exact_mask)

            if img_shape == mask_shape and img_shape is not None:
                report["dimension_match_count"] += 1
            else:
                if len(report["dimension_mismatch_samples"]) < 10:
                    report["dimension_mismatch_samples"].append(
                        {
                            "image": img.name,
                            "mask": exact_mask.name,
                            "image_shape": img_shape,
                            "mask_shape": mask_shape,
                        }
                    )

        elif stem_mask.exists():
            report["stem_filename_match_count"] += 1
            if len(report["missing_exact_samples"]) < 10:
                report["missing_exact_samples"].append(
                    {
                        "image": img.name,
                        "expected_exact": exact_mask.name,
                        "found_stem": stem_mask.name,
                    }
                )
        else:
            if len(report["missing_exact_samples"]) < 10:
                report["missing_exact_samples"].append(
                    {
                        "image": img.name,
                        "expected_exact": exact_mask.name,
                        "found_stem": None,
                    }
                )

    report["filename_match_ok"] = report["exact_filename_match_count"] == report["image_count"]
    report["dimension_match_ok"] = report["dimension_match_count"] == report["image_count"]

    return report


def build_stereo_fusion_cmd(
    colmap_bin: str,
    dense_workspace: Path,
    output_ply: Path,
    mask_path: Path | None,
    min_num_pixels: int,
    max_reproj_error: float,
    max_depth_error: float,
    max_normal_error: float,
) -> list[str]:
    cmd = [
        colmap_bin,
        "stereo_fusion",
        "--workspace_path",
        str(dense_workspace),
        "--output_path",
        str(output_ply),
        "--StereoFusion.min_num_pixels",
        str(min_num_pixels),
        "--StereoFusion.max_reproj_error",
        str(max_reproj_error),
        "--StereoFusion.max_depth_error",
        str(max_depth_error),
        "--StereoFusion.max_normal_error",
        str(max_normal_error),
    ]

    if mask_path is not None:
        cmd += ["--StereoFusion.mask_path", str(mask_path)]

    return cmd


def run_variant(colmap_bin: str, workspace: Path, variant: dict) -> dict:
    out_dir = workspace / "fusion_ablation" / variant["name"]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_ply = out_dir / "fused.ply"
    log_path = out_dir / "stereo_fusion.log"

    dense_workspace = workspace / "dense"
    dense_images_dir = workspace / "dense" / "images"
    mask_path = variant["mask_path"]

    alignment_report = None
    if mask_path is not None:
        alignment_report = inspect_mask_alignment(dense_images_dir, mask_path)

    cmd = build_stereo_fusion_cmd(
        colmap_bin=colmap_bin,
        dense_workspace=dense_workspace,
        output_ply=output_ply,
        mask_path=mask_path,
        min_num_pixels=variant["min_num_pixels"],
        max_reproj_error=variant["max_reproj_error"],
        max_depth_error=variant["max_depth_error"],
        max_normal_error=variant["max_normal_error"],
    )

    proc = subprocess.run(
        cmd,
        cwd=str(workspace),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    log_path.write_text(proc.stdout or "", encoding="utf-8")

    ply_vertex_count = read_ply_vertex_count(output_ply)

    result = {
        "variant": variant["name"],
        "command": " ".join(cmd),
        "return_code": proc.returncode,
        "output_ply": str(output_ply),
        "ply_exists": output_ply.exists(),
        "fused_point_count": ply_vertex_count,
        "ply_vertex_count": ply_vertex_count,
        "mask_alignment": alignment_report,
        "log_path": str(log_path),
    }

    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        required=True,
        help="Prepared COLMAP workspace root, e.g. workspace_quality_test",
    )
    parser.add_argument(
        "--colmap",
        required=True,
        help="COLMAP executable or COLMAP.bat path",
    )
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()

    dense_workspace = workspace / "dense"
    depth_maps = workspace / "dense" / "stereo" / "depth_maps"

    if not dense_workspace.exists():
        raise SystemExit(f"Missing dense workspace: {dense_workspace}")

    if not depth_maps.exists():
        raise SystemExit(f"Missing depth maps: {depth_maps}. Run patch_match_stereo first.")

    raw_masks = workspace / "masks"
    dense_masks = workspace / "dense" / "stereo" / "masks"

    variants = [
        {
            "name": "A1_no_mask_default",
            "mask_path": None,
            "min_num_pixels": 2,
            "max_reproj_error": 2.0,
            "max_depth_error": 0.01,
            "max_normal_error": 10.0,
        },
        {
            "name": "A2_raw_masks_default",
            "mask_path": raw_masks,
            "min_num_pixels": 2,
            "max_reproj_error": 2.0,
            "max_depth_error": 0.01,
            "max_normal_error": 10.0,
        },
        {
            "name": "A3_dense_masks_default",
            "mask_path": dense_masks,
            "min_num_pixels": 2,
            "max_reproj_error": 2.0,
            "max_depth_error": 0.01,
            "max_normal_error": 10.0,
        },
        {
            "name": "C1_dense_masks_relaxed",
            "mask_path": dense_masks,
            "min_num_pixels": 1,
            "max_reproj_error": 4.0,
            "max_depth_error": 0.03,
            "max_normal_error": 25.0,
        },
    ]

    results = []
    for variant in variants:
        print(f"\n=== Running {variant['name']} ===")
        result = run_variant(args.colmap, workspace, variant)
        results.append(result)
        print(
            f"{variant['name']}: "
            f"return={result['return_code']} "
            f"points={result['fused_point_count']} "
            f"ply={result['ply_exists']}"
        )

    summary_path = workspace / "fusion_ablation" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
