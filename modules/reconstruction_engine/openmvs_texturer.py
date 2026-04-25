import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List


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
    ) -> Dict[str, Any]:
        """
        Convert COLMAP output into MVS scene, and run TextureMesh.

        Important fix:
        InterfaceCOLMAP should prefer dense/images when available. Some previous
        runs used dense_workspace.parent / images, which can point to the raw
        input folder instead of COLMAP undistorted images.
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

            log_file.write(f"OpenMVS image-folder selected: {image_folder}\n")
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
                str(image_folder),
            ]

            try:
                self._run_command(cmd_interface, output_dir, log_file)
            except Exception as e:
                raise RuntimeError(f"InterfaceCOLMAP failed: {e}")

            if not scene_mvs.exists():
                raise RuntimeError("Failed to generate scene.mvs")

            cmd_texture = [
                str(self._texture_mesh),
                "-i",
                str(scene_mvs),
                "--mesh-file",
                str(selected_mesh),
                "--export-type",
                "obj",
                "-o",
                str(output_dir / "textured_model"),
                "--working-folder",
                str(output_dir),
            ]

            try:
                self._run_command(cmd_texture, output_dir, log_file)
            except Exception as e:
                log_file.write(f"WARNING: TextureMesh failed with default settings: {e}\n")
                log_file.write("Retrying with safe profile (resolution-level 2)...\n")
                
                # SPRINT 4: Use a separate output basename for safe profile to avoid conflicts
                # with potentially corrupted partial files from the first attempt.
                used_output_stem = "textured_model_safe"
                safe_output_base = output_dir / used_output_stem
                cmd_texture_safe = [
                    str(self._texture_mesh),
                    "-i", str(scene_mvs),
                    "--mesh-file", str(selected_mesh),
                    "--export-type", "obj",
                    "-o", str(safe_output_base),
                    "--working-folder", str(output_dir),
                    "--resolution-level", "2",
                ]
                
                try:
                    self._run_command(cmd_texture_safe, output_dir, log_file)
                    log_file.write("TextureMesh successful with safe profile.\n")
                    # Update textured_obj to point to the safe version
                    textured_obj = output_dir / f"{used_output_stem}.obj"
                except Exception as e2:
                    log_file.write(f"ERROR: TextureMesh failed even with safe profile: {e2}\n")
                    raise RuntimeError(f"TextureMesh failed even with safe profile: {e2}")

            if not textured_obj.exists():
                raise RuntimeError(f"TextureMesh finished but obj file missing: {textured_obj}")

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
            "log_path": str(log_path),
        }
