import os
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

import trimesh
from .failures import TexturingFailed


class OpenMVSTexturer:
    """
    Coordinates OpenMVS to process a selected COLMAP mesh and output a textured artifact.
    """

    def __init__(self, bin_dir: str = None):
        if not bin_dir:
            from modules.operations.settings import settings
            self.bin_dir = Path(settings.openmvs_path)
        else:
            self.bin_dir = Path(bin_dir)

        self._interface_colmap = self.bin_dir / "InterfaceCOLMAP.exe"
        self._texture_mesh = self.bin_dir / "TextureMesh.exe"

        if os.name != "nt":
            self._interface_colmap = self.bin_dir / "InterfaceCOLMAP"
            self._texture_mesh = self.bin_dir / "TextureMesh"

    def is_available(self) -> bool:
        return self._interface_colmap.exists() and self._texture_mesh.exists()

    def _run_command(self, cmd: List[str], cwd: Path, log_file) -> None:
        log_file.write(f"\n--- Running: {' '.join(cmd)} ---\n")
        log_file.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(cwd),
        )

        if process.stdout:
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()

        process.wait()
        if process.returncode != 0:
            raise RuntimeError(
                f"OpenMVS command failed with exit code {process.returncode}: {' '.join(cmd)}"
            )

    def run_texturing(
        self,
        colmap_workspace: Path,
        dense_workspace: Path,
        selected_mesh: str,
        output_dir: Path,
        expected_color: str = "unknown",
        image_folder_override: Optional[Path] = None,
        top_n: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Runs InterfaceCOLMAP and then TextureMesh.
        """
        log_path = output_dir / "texturing.log"
        scene_mvs = output_dir / "scene.mvs"
        textured_obj = output_dir / "textured_model.obj"

        used_output_stem = "textured_model"
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"Starting OpenMVS Texturing using mesh: {selected_mesh}\n")

            if not self.is_available():
                msg = f"OpenMVS binaries missing at {self.bin_dir}. Texturing skipped."
                log_file.write(msg + "\n")
                raise RuntimeError(msg)

            image_folder = dense_workspace / "images"
            if not image_folder.exists():
                image_folder = dense_workspace.parent / "images"

            log_file.write(f"OpenMVS raw image-folder: {image_folder}\n")
            
            if image_folder_override:
                selected_image_folder = image_folder_override
                log_file.write(f"Using image folder override: {selected_image_folder}\n")
            else:
                # SPRINT 5C: Filter images before texturing
                from .texture_frame_filter import TextureFrameFilter
                filter = TextureFrameFilter()
                filter_results = filter.filter_session_images(image_folder, output_dir, expected_color=expected_color)
                
                selected_image_folder = Path(filter_results["selected_images_dir"])
                
                if top_n and filter_results["selected_count"] > top_n:
                    # Create a new sub-folder for top N
                    top_n_dir = output_dir / f"selected_images_top_{top_n}"
                    if top_n_dir.exists(): shutil.rmtree(top_n_dir)
                    top_n_dir.mkdir(parents=True, exist_ok=True)
                    
                    selected_frames = filter_results.get("selected_frames", [])
                    # Frames are already ranked in filter_results["selected_frames"]
                    for s in selected_frames[:top_n]:
                        shutil.copy2(Path(s["path"]), top_n_dir / s["name"])
                    
                    selected_image_folder = top_n_dir
                    log_file.write(f"Limited to top {top_n} frames: {selected_image_folder}\n")

                log_file.write(f"Filtered image-folder: {selected_image_folder}\n")
                original_images = list(image_folder.glob('*.jpg')) + list(image_folder.glob('*.png'))
                log_file.write(f"Original image count: {len(original_images)}\n")
                log_file.write(f"Selected image count: {len(list(selected_image_folder.glob('*.jpg')))}\n")
                log_file.write(f"Fallback used: {filter_results['fallback_used']}\n")
                
                rejected_names = [s["name"] for s in filter_results.get("rejected_frames", [])]
                if rejected_names:
                    log_file.write(f"Rejected image names: {', '.join(rejected_names)}\n")

            log_file.write(f"COLMAP workspace: {colmap_workspace}\n")
            log_file.write(f"Dense workspace: {dense_workspace}\n")

            cmd_interface = [
                str(self._interface_colmap),
                "-i",
                str(dense_workspace),
                "-o",
                str(scene_mvs),
                "--working-folder",
                str(dense_workspace),
                "--image-folder",
                str(selected_image_folder),
            ]

            try:
                self._run_command(cmd_interface, output_dir, log_file)
            except Exception as e:
                raise RuntimeError(f"InterfaceCOLMAP failed: {e}")

            if not scene_mvs.exists():
                raise RuntimeError("Failed to generate scene.mvs")

            # Fix 4: Try PLY mesh input for TextureMesh
            final_mesh_for_texture = Path(selected_mesh)
            if final_mesh_for_texture.suffix.lower() == ".obj":
                ply_path = output_dir / "pre_aligned_mesh_for_texture.ply"
                log_file.write(f"Converting OBJ to PLY for TextureMesh: {selected_mesh} -> {ply_path}\n")
                try:
                    mesh = trimesh.load(str(selected_mesh))
                    if isinstance(mesh, trimesh.Scene):
                        mesh = mesh.dump(concatenate=True)
                    mesh.export(str(ply_path))
                    final_mesh_for_texture = ply_path
                except Exception as e:
                    log_file.write(f"WARNING: OBJ to PLY conversion failed: {e}. Falling back to original OBJ.\n")

            # Fix 5: Improve OpenMVSTexturer diagnostics
            log_file.write(f"selected_mesh path: {selected_mesh}\n")
            log_file.write(f"texture_input_mesh path: {final_mesh_for_texture}\n")
            log_file.write(f"scene.mvs exists: {scene_mvs.exists()}\n")
            if scene_mvs.exists():
                log_file.write(f"scene.mvs size: {scene_mvs.stat().st_size} bytes\n")

            # Fix 3: Try explicit output extension -o textured_model.obj
            cmd_texture = [
                str(self._texture_mesh),
                "-i",
                str(scene_mvs),
                "--mesh-file",
                str(final_mesh_for_texture),
                "--export-type",
                "obj",
                "-o",
                str(output_dir / "textured_model.obj"),
                "--working-folder",
                str(output_dir),
            ]

            log_file.write(f"TextureMesh Command: {' '.join(cmd_texture)}\n")
            try:
                self._run_command(cmd_texture, output_dir, log_file)
                log_file.write(f"TextureMesh returned exit code 0\n")
            except Exception as e:
                log_file.write(f"WARNING: TextureMesh failed with default settings: {e}\n")
                log_file.write("Retrying with safe profile (resolution-level 2)...\n")
                
                # SPRINT 4: Use a separate output basename for safe profile to avoid conflicts
                # with potentially corrupted partial files from the first attempt.
                used_output_stem = "textured_model_safe"
                safe_output_base = output_dir / f"{used_output_stem}.obj"
                cmd_texture_safe = [
                    str(self._texture_mesh),
                    "-i", str(scene_mvs),
                    "--mesh-file", str(final_mesh_for_texture),
                    "--export-type", "obj",
                    "-o", str(safe_output_base),
                    "--working-folder", str(output_dir),
                    "--resolution-level", "2",
                ]
                
                log_file.write(f"TextureMesh (Safe) Command: {' '.join(cmd_texture_safe)}\n")
                try:
                    self._run_command(cmd_texture_safe, output_dir, log_file)
                    log_file.write("TextureMesh successful with safe profile.\n")
                    # Update textured_obj to point to the safe version
                    textured_obj = output_dir / f"{used_output_stem}.obj"
                except Exception as e2:
                    log_file.write(f"ERROR: TextureMesh failed even with safe profile: {e2}\n")
                    raise TexturingFailed(f"TextureMesh failed even with safe profile: {e2}", log_path=str(log_path))

            # Fix 1: TextureMesh output must be verified immediately after command
            log_file.write(f"Verifying outputs in {output_dir}...\n")
            all_files = os.listdir(str(output_dir))
            log_file.write(f"Files in output_dir: {all_files}\n")

            obj_exists = any(f.endswith(".obj") for f in all_files)
            mtl_exists = any(f.endswith(".mtl") for f in all_files)
            texture_exists = any("map_Kd" in f for f in all_files) or any(f.endswith((".png", ".jpg", ".jpeg")) for f in all_files if "textured_model" in f)

            if not (obj_exists and mtl_exists and texture_exists):
                log_file.write(f"CRITICAL ERROR: TextureMesh finished but outputs are missing. obj={obj_exists}, mtl={mtl_exists}, tex={texture_exists}\n")
                raise TexturingFailed(f"TextureMesh finished but outputs are missing. obj={obj_exists}, mtl={mtl_exists}, tex={texture_exists}", log_path=str(log_path))

            # Collect textures that match the successful output stem
            generated_textures = list(output_dir.glob(f"{used_output_stem}*_map_Kd.*"))
            
            # SPRINT 4: Robust MTL parsing to discover textures regardless of naming convention
            mtl_file = output_dir / f"{used_output_stem}.mtl"
            if mtl_file.exists():
                try:
                    with open(mtl_file, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            if line.strip().startswith("map_Kd"):
                                # Use maxsplit=1 to handle filenames with spaces
                                parts = line.strip().split(None, 1)
                                if len(parts) > 1:
                                    tex_name = parts[1].strip()
                                    tex_path = output_dir / tex_name
                                    if tex_path.exists() and tex_path not in generated_textures:
                                        generated_textures.append(tex_path)
                                        log_file.write(f"Texture discovered via MTL: {tex_name}\n")
                except Exception as e:
                    log_file.write(f"WARNING: Failed to parse MTL for textures: {e}\n")

            if not generated_textures:
                # Fallback to general images but still filter by used_output_stem to avoid partials
                generated_textures = [
                    p for p in (list(output_dir.glob(f"{used_output_stem}*.png")) + list(output_dir.glob(f"{used_output_stem}*.jpg")))
                    if p.name != f"{used_output_stem}.png" and p.name != f"{used_output_stem}.jpg"
                ]

            log_file.write(f"Texturing completed successfully. Used stem: {used_output_stem}, Atlas count: {len(generated_textures)}\n")

        return {
            "textured_mesh_path": str(textured_obj),
            "texture_atlas_paths": [str(p) for p in generated_textures],
            "texturing_engine": "openmvs",
            "log_path": str(log_path)
        }
