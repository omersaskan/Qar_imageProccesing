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

        # On non-Windows, default to lack of extensions
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
            raise RuntimeError(f"OpenMVS command failed with exit code {process.returncode}: {' '.join(cmd)}")

    def run_texturing(
        self,
        colmap_workspace: Path,
        dense_workspace: Path,
        selected_mesh: str,
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Convert COLMAP output into MVS scene, and run TextureMesh.
        """
        log_path = output_dir / "texturing.log"
        scene_mvs = output_dir / "scene.mvs"
        textured_obj = output_dir / "textured_model.obj"
        texture_png = output_dir / "textured_model_material_0_map_Kd.png"

        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"Starting OpenMVS Texturing using mesh: {selected_mesh}\n")

            if not self.is_available():
                msg = f"OpenMVS binaries missing at {self.bin_dir}. Texturing skipped."
                log_file.write(msg + "\n")
                raise RuntimeError(msg)

            # Step 1: InterfaceCOLMAP
            cmd_interface = [
                str(self._interface_colmap),
                "-i", str(dense_workspace),
                "-o", str(scene_mvs),
                "--working-folder", str(dense_workspace),
                "--image-folder", str(dense_workspace / "images")
            ]
            
            try:
                self._run_command(cmd_interface, output_dir, log_file)
            except Exception as e:
                raise RuntimeError(f"InterfaceCOLMAP failed: {e}")

            if not scene_mvs.exists():
                raise RuntimeError("Failed to generate scene.mvs")

            # Step 2: TextureMesh
            # We explicitly pass the selected mesh so we texture the cleaned or best candidate.
            cmd_texture = [
                str(self._texture_mesh),
                "-i", str(scene_mvs),
                "--mesh-file", str(selected_mesh),
                "--export-type", "obj",
                "-o", str(output_dir / "textured_model"),
                "--working-folder", str(output_dir)
            ]
            
            try:
                self._run_command(cmd_texture, output_dir, log_file)
            except Exception as e:
                raise RuntimeError(f"TextureMesh failed: {e}")
                
            # Texture files can be generated as multiple atlas maps depending on settings.
            # Usually textured_model.obj, textured_model.mtl, textured_model_material_0_map_Kd.png
            # Let's discover generated textures honestly.
            
            if not textured_obj.exists():
                raise RuntimeError("TextureMesh finished but obj file missing.")

            generated_textures = list(output_dir.glob("*_map_Kd.*"))
            if not generated_textures:
                generated_textures = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))
            
            log_file.write("Texturing completed successfully.\n")

        return {
            "textured_mesh_path": str(textured_obj),
            "texture_atlas_paths": [str(p) for p in generated_textures if "textured_model" in p.name],
            "texturing_engine": "openmvs",
            "log_path": str(log_path),
        }
