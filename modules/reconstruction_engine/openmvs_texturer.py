import os
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

import trimesh
import cv2
import numpy as np
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

    def _simplify_mesh(self, input_mesh: Path, output_mesh: Path, target_faces: int, log_file) -> Path:
        """
        Simplifies mesh to target face count using trimesh/fast_simplification if available.
        """
        log_file.write(f"Simplifying mesh {input_mesh.name} to {target_faces} faces...\n")
        try:
            mesh = trimesh.load(str(input_mesh))
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            
            current_faces = len(mesh.faces)
            if current_faces <= target_faces:
                log_file.write(f"Mesh already below target ({current_faces} <= {target_faces}), skipping simplification.\n")
                # Still export to the new path to keep it consistent
                mesh.export(str(output_mesh))
                return output_mesh

            # Use fast_simplification if available, else trimesh built-in
            try:
                import fast_simplification
                # fast_simplification uses a ratio
                ratio = target_faces / current_faces
                points, faces = fast_simplification.simplify(mesh.vertices, mesh.faces, ratio)
                new_mesh = trimesh.Trimesh(vertices=points, faces=faces)
            except ImportError:
                log_file.write("fast_simplification not found, using trimesh.simplify_quadratic...\n")
                new_mesh = mesh.simplify_quadratic(target_faces)

            new_mesh.export(str(output_mesh))
            log_file.write(f"Simplification complete: {current_faces} -> {len(new_mesh.faces)} faces.\n")
            return output_mesh
        except Exception as e:
            log_file.write(f"WARNING: Mesh simplification failed: {e}. Using original mesh.\n")
            return input_mesh

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
            # Fix 8: Improve failure diagnostics
            log_file.write(f"\nERROR: Command failed with exit code {process.returncode}\n")
            log_file.write(f"Check output above for errors.\n")
            raise RuntimeError(
                f"OpenMVS command failed with exit code {process.returncode}: {' '.join(cmd)}"
            )

    def _create_compatible_image_folder(
        self, 
        original_images_dir: Path, 
        target_dir: Path, 
        selected_names: List[str], 
        masked_images_dir: Optional[Path] = None,
        use_masks: bool = False,
        log_file = None
    ) -> None:
        """
        Creates a folder containing all original filenames.
        - Selected frames: copied from original or masked_images_dir.
        - Rejected frames: neutralized (solid cream).
        """
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        selected_names_set = set(selected_names)
        original_images = list(original_images_dir.glob("*.jpg")) + list(original_images_dir.glob("*.png"))
        
        if log_file:
            log_file.write(f"Creating compatible image folder at {target_dir}\n")
            log_file.write(f"Original images count: {len(original_images)}\n")
            log_file.write(f"Selected images count: {len(selected_names_set)}\n")
            log_file.write(f"Using masks: {use_masks}\n")

        # Cream color: (220, 245, 245) in BGR
        CREAM_COLOR = (220, 245, 245)

        for img_path in original_images:
            dest_path = target_dir / img_path.name
            if img_path.name in selected_names_set:
                # Use high-quality version
                source = img_path
                if use_masks and masked_images_dir:
                    m_source = masked_images_dir / img_path.name
                    if m_source.exists():
                        source = m_source
                
                shutil.copy2(source, dest_path)
            else:
                # Create neutralized version
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        cream = np.full((h, w, 3), CREAM_COLOR, dtype=np.uint8)
                        cv2.imwrite(str(dest_path), cream)
                    else:
                        shutil.copy2(img_path, dest_path) # Fallback
                except Exception as e:
                    if log_file:
                        log_file.write(f"Warning: Failed to neutralize {img_path.name}: {e}\n")
                    shutil.copy2(img_path, dest_path)

    def _check_image_folder_completeness(self, dense_workspace: Path, image_folder: Path, log_file) -> None:
        """
        Verifies that image_folder contains all images referenced in dense_workspace/images.
        """
        original_images_dir = dense_workspace / "images"
        if not original_images_dir.exists():
             original_images_dir = dense_workspace.parent / "images"
        
        if not original_images_dir.exists():
            log_file.write("Preflight check: Warning, original images dir not found, skipping completeness check.\n")
            return
            
        referenced_images = list(original_images_dir.glob("*.jpg")) + list(original_images_dir.glob("*.png"))
        provided_images = {p.name for p in (list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png")))}
        
        missing = []
        for ref in referenced_images:
            if ref.name not in provided_images:
                missing.append(ref.name)
        
        log_file.write(f"Preflight check: referenced={len(referenced_images)}, provided={len(provided_images)}\n")
        if missing:
            log_file.write(f"ERROR: TEXTURE_IMAGE_FOLDER_INCOMPLETE. Missing filenames: {', '.join(missing[:10])}{'...' if len(missing)>10 else ''}\n")
            raise RuntimeError(f"TEXTURE_IMAGE_FOLDER_INCOMPLETE: {len(missing)} images missing from {image_folder}")

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
        Runs InterfaceCOLMAP and then TextureMesh with a retry ladder.
        """
        from modules.operations.settings import settings
        
        log_path = output_dir / "texturing.log"
        scene_mvs = output_dir / "scene.mvs"
        
        # SPRINT 5C: Load targets from settings
        target_60k = settings.texture_texturing_target_faces
        target_40k = settings.texture_safe_texturing_target_faces
        target_crash_retry = settings.texture_native_crash_retry_faces
        max_selected_frames = settings.texture_max_selected_frames
        
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
            
            compatible_image_folder = output_dir / "compatible_images"
            
            # --- IMAGE SELECTION & COMPATIBILITY ---
            has_masks_available = False
            masked_images_dir = None
            
            if image_folder_override:
                log_file.write(f"Using image folder override: {image_folder_override}\n")
                selected_names = [p.name for p in (list(image_folder_override.glob("*.jpg")) + list(image_folder_override.glob("*.png")))]
                is_masked = "masked" in str(image_folder_override).lower()
                has_masks_available = is_masked
                masked_images_dir = image_folder_override if is_masked else None
                
                self._create_compatible_image_folder(
                    original_images_dir=image_folder,
                    target_dir=compatible_image_folder,
                    selected_names=selected_names,
                    masked_images_dir=masked_images_dir,
                    use_masks=is_masked,
                    log_file=log_file
                )
            else:
                from .texture_frame_filter import TextureFrameFilter
                filter = TextureFrameFilter()
                filter_results = filter.filter_session_images(image_folder, output_dir, expected_color=expected_color, target_count=top_n or max_selected_frames)
                
                selected_frames = filter_results.get("selected_frames", [])
                has_masks_available = filter_results.get("has_masks_available", False)
                masked_images_dir = Path(filter_results["masked_images_dir"]) if filter_results.get("masked_images_dir") else None
                
                selected_names = [s["name"] for s in selected_frames]
                
                self._create_compatible_image_folder(
                    original_images_dir=image_folder,
                    target_dir=compatible_image_folder,
                    selected_names=selected_names,
                    masked_images_dir=masked_images_dir,
                    use_masks=False, # Default to compatible neutralization
                    log_file=log_file
                )

            # --- OPERATOR GUIDANCE ---
            operator_guidance = None
            if not has_masks_available:
                operator_guidance = "Mask unavailable; cluttered scene cannot be safely isolated. Use single object on plain matte background."
                log_file.write(f"GUIDANCE: {operator_guidance}\n")

            # --- INTERFACE COLMAP ---
            log_file.write(f"Final compatible image-folder: {compatible_image_folder}\n")
            try:
                self._check_image_folder_completeness(dense_workspace, compatible_image_folder, log_file)
            except Exception as pre_err:
                log_file.write(f"CRITICAL PREFLIGHT FAILURE: {pre_err}\n")
                raise

            cmd_interface = [
                str(self._interface_colmap),
                "-i", str(dense_workspace),
                "-o", str(scene_mvs),
                "--working-folder", str(dense_workspace),
                "--image-folder", str(compatible_image_folder),
            ]
            self._run_command(cmd_interface, output_dir, log_file)

            if not scene_mvs.exists():
                raise RuntimeError("Failed to generate scene.mvs")

            # --- RETRY LADDER ---
            # Attempt A: compatible_images + 60k mesh + default
            # Attempt B: compatible_images + 60k mesh + resolution-level 2
            # Attempt C: compatible_images + 40k mesh + resolution-level 2
            # Attempt D: raw_all_images + 40k mesh + resolution-level 2
            
            attempts = [
                {"name": "Attempt A", "mesh_faces": target_60k, "res_level": 0, "use_raw_all": False},
                {"name": "Attempt B", "mesh_faces": target_60k, "res_level": 2, "use_raw_all": False},
                {"name": "Attempt C", "mesh_faces": target_40k, "res_level": 2, "use_raw_all": False},
                {"name": "Attempt D", "mesh_faces": target_40k, "res_level": 2, "use_raw_all": True},
            ]
            
            success = False
            final_textured_obj = None
            
            for i, att in enumerate(attempts):
                log_file.write(f"\n>>> Starting {att['name']} <<<\n")
                
                # 1. Prepare Mesh
                mesh_path = output_dir / f"texturing_mesh_{att['mesh_faces']//1000}k.ply"
                if not mesh_path.exists():
                    self._simplify_mesh(Path(selected_mesh), mesh_path, att['mesh_faces'], log_file)
                
                # 2. Prepare Scene (if raw_all requested)
                current_scene = scene_mvs
                if att['use_raw_all']:
                    if not settings.texture_retry_raw_all:
                        log_file.write("TEXTURE_RETRY_RAW_ALL is disabled, skipping Attempt D.\n")
                        continue
                        
                    raw_scene = output_dir / "scene_raw_all.mvs"
                    if not raw_scene.exists():
                        log_file.write("Creating raw_all scene for retry...\n")
                        cmd_raw = [
                            str(self._interface_colmap),
                            "-i", str(dense_workspace),
                            "-o", str(raw_scene),
                            "--working-folder", str(dense_workspace),
                            "--image-folder", str(image_folder),
                        ]
                        try:
                            self._run_command(cmd_raw, output_dir, log_file)
                            current_scene = raw_scene
                        except Exception as raw_err:
                            log_file.write(f"Failed to create raw scene: {raw_err}. Reverting to compatible.\n")
                    else:
                        current_scene = raw_scene

                # 3. Run TextureMesh
                out_stem = f"textured_model_{att['name'].replace(' ', '_').lower()}"
                out_obj = output_dir / f"{out_stem}.obj"
                
                cmd_texture = [
                    str(self._texture_mesh),
                    "-i", str(current_scene),
                    "--mesh-file", str(mesh_path),
                    "-o", str(out_obj),
                    "--export-type", "obj",
                    "--working-folder", str(output_dir),
                    "--resolution-level", str(att['res_level']),
                ]
                
                try:
                    self._run_command(cmd_texture, output_dir, log_file)
                    # Verify output
                    if out_obj.exists():
                        # Quick check for texture atlas
                        if any(output_dir.glob(f"{out_stem}*map_Kd*")):
                            log_file.write(f"{att['name']} SUCCESSFUL.\n")
                            success = True
                            final_textured_obj = out_obj
                            used_output_stem = out_stem
                            break
                    log_file.write(f"{att['name']} finished but output missing.\n")
                except Exception as e:
                    exit_code = None
                    if "exit code" in str(e):
                        try:
                            exit_code = int(str(e).split("exit code ")[1].split(":")[0])
                        except: pass
                    
                    log_file.write(f"{att['name']} FAILED: {e} (exit_code={exit_code})\n")
                    
                    if exit_code == 3221226505:
                        log_file.write("NATIVE CRASH DETECTED. Mesh simplification might help in next attempt.\n")
                        # Special case: if Attempt C or D crashed, we might want one more even smaller mesh
                        if att['name'] in ["Attempt C", "Attempt D"] and target_crash_retry < att['mesh_faces']:
                            log_file.write(f"Extra Retry with even lower face count: {target_crash_retry}\n")
                            # We could dynamically add an attempt or just let it fail
                
            if not success:
                # Diagnostics: List files even after failure
                log_file.write(f"Diagnostics: Files in output_dir after all attempts failed: {os.listdir(str(output_dir))}\n")
                raise TexturingFailed(
                    "All TextureMesh attempts in retry ladder failed.", 
                    log_path=str(log_path)
                )

            # --- POST-PROCESSING & ATLAS DISCOVERY ---
            log_file.write(f"Verifying final outputs for {used_output_stem}...\n")
            generated_textures = list(output_dir.glob(f"{used_output_stem}*_map_Kd.*"))
            
            # Robust MTL parsing
            mtl_file = output_dir / f"{used_output_stem}.mtl"
            if mtl_file.exists():
                try:
                    with open(mtl_file, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            if line.strip().startswith("map_Kd"):
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
                generated_textures = [
                    p for p in (list(output_dir.glob(f"{used_output_stem}*.png")) + list(output_dir.glob(f"{used_output_stem}*.jpg")))
                    if p.name != f"{used_output_stem}.png" and p.name != f"{used_output_stem}.jpg"
                ]

            log_file.write(f"Texturing completed successfully. Used stem: {used_output_stem}, Atlas count: {len(generated_textures)}\n")

        return {
            "textured_mesh_path": str(final_textured_obj),
            "texture_atlas_paths": [str(p) for p in generated_textures],
            "texturing_engine": "openmvs",
            "log_path": str(log_path),
            "has_masks_available": has_masks_available,
            "operator_guidance": operator_guidance,
            "attempt_used": used_output_stem
        }
