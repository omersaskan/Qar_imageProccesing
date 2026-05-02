import os
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

import trimesh
import cv2
import numpy as np
import json
from .failures import TexturingFailed


class OpenMVSTexturer:
    """
    Coordinates OpenMVS to process a selected COLMAP mesh and output a textured artifact.
    """

    def __init__(self, bin_dir: str = None, settings_override=None, command_config=None):
        from modules.operations.settings import settings as _global_settings
        self._settings = settings_override or _global_settings
        # Sprint 4.5: optional preset-aware command config
        self.command_config = command_config

        if not bin_dir:
            self.bin_dir = Path(self._settings.openmvs_path)
        else:
            self.bin_dir = Path(bin_dir)

        self._interface_colmap = self.bin_dir / "InterfaceCOLMAP.exe"
        self._texture_mesh = self.bin_dir / "TextureMesh.exe"

        if os.name != "nt":
            self._interface_colmap = self.bin_dir / "InterfaceCOLMAP"
            self._texture_mesh = self.bin_dir / "TextureMesh"

    def is_available(self) -> bool:
        return self._interface_colmap.exists() and self._texture_mesh.exists()

    def _sanitize_mesh(self, mesh: trimesh.Trimesh, log_file) -> trimesh.Trimesh:
        """
        Aggressively cleans the mesh to prevent OpenMVS native crashes.
        """
        initial_faces = len(mesh.faces)
        try:
            # SPRINT 5C: Use trimesh.process() for standard cleanup
            # It handles duplicate faces, unreferenced vertices, etc.
            mesh.process(validate=True)
            
            # Explicitly remove degenerate faces if process didn't catch them all
            mesh.update_faces(mesh.nondegenerate_faces())
            
            # Remove infinite values if any
            mesh.remove_infinite_values()
        except Exception as e:
            log_file.write(f"WARNING: Mesh sanitization (process) failed: {e}\n")
        
        # Remove zero-area triangles if any left
        try:
            if len(mesh.faces) > 0:
                face_areas = mesh.area_faces
                non_zero = face_areas > 1e-12
                if not np.all(non_zero):
                    mesh.update_faces(non_zero)
        except Exception as e:
            log_file.write(f"WARNING: Mesh area-based sanitization failed: {e}\n")

        final_faces = len(mesh.faces)
        if final_faces < initial_faces:
            log_file.write(f"Mesh sanitized: removed {initial_faces - final_faces} problematic faces.\n")
        return mesh

    def _simplify_mesh(self, input_mesh: Path, output_mesh: Path, target_faces: int, log_file) -> Path:
        """
        Simplifies mesh to target face count using trimesh/fast_simplification if available.
        """
        log_file.write(f"Simplifying mesh {input_mesh.name} to {target_faces} faces...\n")
        try:
            mesh = trimesh.load(str(input_mesh))
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            
            # SPRINT: Sanitize mesh before simplification to prevent native crashes
            mesh = self._sanitize_mesh(mesh, log_file)
            
            current_faces = len(mesh.faces)
            if current_faces <= target_faces:
                log_file.write(f"Mesh already below target ({current_faces} <= {target_faces}), skipping simplification.\n")
                # Still export to the new path to keep it consistent
                mesh.export(str(output_mesh))
                return output_mesh

            # Use fast_simplification if available, else trimesh built-in
            try:
                import fast_simplification
                # fast_simplification uses a reduction ratio (fraction of faces to REMOVE)
                reduction_ratio = 1.0 - (target_faces / current_faces)
                points, faces = fast_simplification.simplify(mesh.vertices, mesh.faces, reduction_ratio)
                new_mesh = trimesh.Trimesh(vertices=points, faces=faces)
            except ImportError:
                log_file.write("fast_simplification not found, using trimesh.simplify_quadratic...\n")
                new_mesh = mesh.simplify_quadratic(target_faces)

            new_mesh.export(str(output_mesh))
            log_file.write(f"Simplification complete: {current_faces} -> {len(new_mesh.faces)} faces.\n")
            return output_mesh
        except Exception as e:
            log_file.write(f"WARNING: Mesh simplification failed: {e}. Using original mesh.\n")
            import shutil
            shutil.copy2(str(input_mesh), str(output_mesh))
            return output_mesh

    def _run_command(self, cmd: List[str], cwd: Path, log_file, timeout: Optional[int] = None, max_threads: Optional[int] = None) -> None:
        import time
        
        # Inject max-threads if requested and not already in cmd
        if max_threads is not None and "--max-threads" not in cmd:
            # Check if cmd already has it
            has_threads = False
            for i, part in enumerate(cmd):
                if part == "--max-threads":
                    cmd[i+1] = str(max_threads)
                    has_threads = True
                    break
            if not has_threads:
                cmd.extend(["--max-threads", str(max_threads)])

        log_file.write(f"\n--- Running: {' '.join(cmd)} (timeout={timeout}s) ---\n")
        log_file.flush()

        # Create a temporary file for stdout/stderr to ensure we capture EVERYTHING
        # even if the process crashes natively. 
        temp_log_path = cwd / f"process_{int(time.time()*1000)}.tmp.log"
        
        try:
            with open(temp_log_path, "w", encoding="utf-8") as tmp_f:
                start_time = time.time()
                process = subprocess.Popen(
                    cmd,
                    stdout=tmp_f,
                    stderr=subprocess.STDOUT,
                    cwd=str(cwd),
                )

                # Monitor loop
                while process.poll() is None:
                    if timeout and (time.time() - start_time) > timeout:
                        process.kill()
                        log_file.write("\n\n!!! ERROR: TIMEOUT EXCEEDED !!!\n")
                        raise RuntimeError(f"OpenMVS command timed out after {timeout}s: {' '.join(cmd)}")
                    time.sleep(0.5)

            # Append the temp log content to the main log file
            if temp_log_path.exists():
                with open(temp_log_path, "r", encoding="utf-8", errors="ignore") as tmp_f:
                    content = tmp_f.read()
                    if content:
                        log_file.write(content)
                temp_log_path.unlink()
            
            if process.returncode != 0:
                # SPRINT: Check for specific native crash code 3221226505
                raise RuntimeError(f"Command failed with exit code {process.returncode}")
                
        except Exception as e:
            # Cleanup if needed
            if temp_log_path.exists():
                try:
                    with open(temp_log_path, "r", encoding="utf-8", errors="ignore") as tmp_f:
                        content = tmp_f.read()
                        if content:
                            log_file.write(content)
                    temp_log_path.unlink()
                except: pass
            raise e
            if "timed out" in str(e):
                raise
            # If any other error occurs, make sure to kill the process
            if process.poll() is None:
                process.kill()
            raise

    def _create_compatible_image_folder(
        self, 
        original_images_dir: Path, 
        target_dir: Path, 
        selected_names: List[str], 
        masked_images_dir: Optional[Path] = None,
        use_masks: bool = False,
        neutralization_type: str = "cream",
        log_file = None
    ) -> Dict[str, Any]:
        """
        Creates a folder containing all original filenames.
        - Selected frames: copied from original or masked_images_dir.
        - Rejected frames: neutralized (solid cream or black mask).
        Returns counts of images processed from each source.
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
            log_file.write(f"Using masks for selected: {use_masks}\n")
            log_file.write(f"Neutralization type: {neutralization_type}\n")

        # Cream color: (220, 245, 245) in BGR
        CREAM_COLOR = (220, 245, 245)
        
        counts = {
            "selected_from_masked": 0,
            "selected_from_raw": 0,
            "rejected_neutralized": 0
        }

        for img_path in original_images:
            dest_path = target_dir / img_path.name
            if img_path.name in selected_names_set:
                # Use high-quality version
                source = img_path
                from_masked = False
                if use_masks and masked_images_dir:
                    m_source = masked_images_dir / img_path.name
                    if m_source.exists():
                        source = m_source
                        from_masked = True
                
                shutil.copy2(source, dest_path)
                if from_masked:
                    counts["selected_from_masked"] += 1
                else:
                    counts["selected_from_raw"] += 1
            else:
                # Create neutralized version
                try:
                    if neutralization_type == "selected_only":
                        # For experimental 'C', we just skip these if possible. 
                        # But OpenMVS needs them. So we'll use black as safest neutral.
                        neutralization_type = "black_mask" 
                    
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        if neutralization_type == "black_mask":
                            neutral = np.zeros((h, w, 3), dtype=np.uint8)
                        else: # default cream
                            neutral = np.full((h, w, 3), CREAM_COLOR, dtype=np.uint8)
                        
                        cv2.imwrite(str(dest_path), neutral)
                        counts["rejected_neutralized"] += 1
                    else:
                        shutil.copy2(img_path, dest_path) # Fallback
                except Exception as e:
                    if log_file:
                        log_file.write(f"Warning: Failed to neutralize {img_path.name}: {e}\n")
                    shutil.copy2(img_path, dest_path)

        return counts

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
        neutralization_type: str = "cream",
        product_profile: str = "generic",
    ) -> Dict[str, Any]:
        """
        Runs InterfaceCOLMAP and then TextureMesh with a retry ladder.
        """
        # Use profile-overridden settings (falls back to global if no override)
        active = self._settings

        log_path = output_dir / "texturing.log"
        scene_mvs = output_dir / "scene.mvs"

        # SPRINT 5C: Load targets from active (profile-aware) settings
        target_60k = active.texture_texturing_target_faces
        target_40k = active.texture_safe_texturing_target_faces
        target_crash_retry = active.texture_native_crash_retry_faces
        max_selected_frames = active.texture_max_selected_frames

        # Sprint 4.5: command_config can clamp the texture atlas resolution
        # (used by glb_exporter / texture pipeline downstream — surfaced in log).
        if self.command_config is not None:
            try:
                cmd_tex_res = int(self.command_config.openmvs.texture_resolution)
                cmd_threads = int(self.command_config.openmvs.max_threads)
                # Persist intent to log so QA can audit
                with open(log_path, "a", encoding="utf-8") as _lf:
                    _lf.write(
                        f"[command_config] preset={self.command_config.source_preset_name} "
                        f"texture_resolution={cmd_tex_res} max_threads={cmd_threads}\n"
                    )
            except Exception:
                pass
        
        used_output_stem = "textured_model"
        
        # Metrics to persist
        texture_metrics = {
            "selected_texture_frame_names": [],
            "masked_images_dir": None,
            "use_masks_for_selected_frames": False,
            "selected_frames_from_masked": 0,
            "selected_frames_from_raw": 0,
            "rejected_frames_neutralized": 0,
            "neutralization_type": neutralization_type,
            "cameras_loaded_for_texture_selection": False,
            "azimuths_computed": 0,
            "selected_frame_azimuth_list": [],
            "max_texture_coverage_gap_degrees": 0.0
        }

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
                
                source_counts = self._create_compatible_image_folder(
                    original_images_dir=image_folder,
                    target_dir=compatible_image_folder,
                    selected_names=selected_names,
                    masked_images_dir=masked_images_dir,
                    use_masks=is_masked,
                    neutralization_type=neutralization_type,
                    log_file=log_file
                )
            else:
                from .texture_frame_filter import TextureFrameFilter
                filter = TextureFrameFilter()
                filter_results = filter.filter_session_images(
                    image_folder, 
                    output_dir, 
                    dense_workspace=dense_workspace,
                    expected_color=expected_color, 
                    target_count=top_n or max_selected_frames,
                    product_profile=product_profile
                )
                
                selected_frames = filter_results.get("selected_frames", [])
                has_masks_available = filter_results.get("has_masks_available", False)
                masked_images_dir = Path(filter_results["masked_images_dir"]) if filter_results.get("masked_images_dir") else None
                
                selected_names = [s["name"] for s in selected_frames]
                
                # BUG FIX: If has_masks_available and masked_images_dir exists, use masks
                use_masks_for_selected = has_masks_available and masked_images_dir is not None and masked_images_dir.exists()
                
                source_counts = self._create_compatible_image_folder(
                    original_images_dir=image_folder,
                    target_dir=compatible_image_folder,
                    selected_names=selected_names,
                    masked_images_dir=masked_images_dir,
                    use_masks=use_masks_for_selected,
                    neutralization_type=neutralization_type,
                    log_file=log_file
                )
                
                # Update metrics
                texture_metrics["use_masks_for_selected_frames"] = use_masks_for_selected
                texture_metrics["cameras_loaded_for_texture_selection"] = filter_results.get("cameras_loaded_for_texture_selection", False)
                texture_metrics["azimuths_computed"] = filter_results.get("azimuths_computed", 0)
                texture_metrics["selected_frame_azimuth_list"] = filter_results.get("selected_azimuths", [])
                texture_metrics["max_texture_coverage_gap_degrees"] = filter_results.get("max_gap_degrees", 0.0)

            # Populate metrics
            texture_metrics["selected_texture_frame_names"] = selected_names
            texture_metrics["masked_images_dir"] = str(masked_images_dir) if masked_images_dir else None
            texture_metrics["selected_frames_from_masked"] = source_counts["selected_from_masked"]
            texture_metrics["selected_frames_from_raw"] = source_counts["selected_from_raw"]
            texture_metrics["rejected_frames_neutralized"] = source_counts["rejected_neutralized"]

            # Persist metrics to disk for later reporting
            with open(output_dir / "texturing_metrics.json", "w") as f:
                json.dump(texture_metrics, f, indent=2)

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
                "--working-folder", str(output_dir),
                "--image-folder", str(compatible_image_folder),
            ]
            self._run_command(cmd_interface, output_dir, log_file, timeout=settings.texture_timeout_sec)

            if not scene_mvs.exists():
                raise RuntimeError("Failed to generate scene.mvs")

            # --- RETRY LADDER ---
            # Attempt A: compatible_images + 60k mesh + default
            # Attempt B: compatible_images + 60k mesh + resolution-level 2
            # Attempt C: compatible_images + 40k mesh + resolution-level 2
            # Attempt D: raw_all_images + 40k mesh + resolution-level 2
            
            attempts = [
                {"name": "Attempt A", "mesh_faces": target_60k, "res_level": 1, "use_raw_all": False, "threads": None},
                {"name": "Attempt B", "mesh_faces": target_60k, "res_level": 2, "use_raw_all": False, "threads": None},
                {"name": "Attempt C", "mesh_faces": target_40k, "res_level": 2, "use_raw_all": False, "threads": None},
                {"name": "Attempt D", "mesh_faces": target_40k, "res_level": 2, "use_raw_all": True, "threads": None},
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
                            "--working-folder", str(output_dir),
                            "--image-folder", str(image_folder),
                        ]
                        try:
                            self._run_command(cmd_raw, output_dir, log_file, timeout=settings.texture_timeout_sec)
                            current_scene = raw_scene
                        except Exception as raw_err:
                            log_file.write(f"Failed to create raw scene: {raw_err}. Reverting to compatible.\n")
                    else:
                        current_scene = raw_scene

                # 3. Run TextureMesh
                out_stem = f"textured_model_{att['name'].replace(' ', '_').lower()}"
                out_glb = output_dir / f"{out_stem}.glb"
                
                cmd_texture = [
                    str(self._texture_mesh),
                    "-i", str(current_scene),
                    "--mesh-file", str(mesh_path),
                    "-o", str(out_glb),
                    "--working-folder", str(output_dir),
                    "--resolution-level", str(att['res_level']),
                    "--export-type", "glb"
                ]
                
                # Add threads if specified in attempt
                max_threads = att.get("threads")
                
                try:
                    self._run_command(cmd_texture, output_dir, log_file, timeout=settings.texture_timeout_sec, max_threads=max_threads)
                    
                    # Verify output (handle case where OpenMVS still outputs .ply or PNG with a different name)
                    out_ply = output_dir / f"{out_stem}.ply"
                    
                    # If it outputted PLY instead of OBJ, we convert it or use it. We'll convert it to OBJ since the pipeline expects it.
                    if out_ply.exists() and not out_obj.exists():
                        log_file.write(f"WARNING: OpenMVS outputted .ply instead of .obj. Converting {out_ply.name} to .obj...\n")
                        import trimesh
                        try:
                            m = trimesh.load(str(out_ply))
                            m.export(str(out_obj))
                        except Exception as conv_err:
                            log_file.write(f"Failed to convert PLY to OBJ: {conv_err}\n")

                    if out_glb.exists():
                        log_file.write(f"{att['name']} SUCCESSFUL.\n")
                        success = True
                        final_textured_obj = out_glb
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
                            if not any(a["name"] == "Attempt E" for a in attempts):
                                log_file.write(f"Extra Retry with even lower face count: {target_crash_retry}\n")
                                attempts.append({
                                    "name": "Attempt E",
                                    "mesh_faces": target_crash_retry,
                                    "res_level": 2,
                                    "use_raw_all": att['use_raw_all'],
                                    "threads": None
                                })
                                # SPRINT: Add even deeper fallbacks for native crashes
                                attempts.append({
                                    "name": "Attempt F",
                                    "mesh_faces": target_crash_retry,
                                    "res_level": 2,
                                    "use_raw_all": att['use_raw_all'],
                                    "threads": 8 # Limit threads to 8
                                })
                                attempts.append({
                                    "name": "Attempt G",
                                    "mesh_faces": target_crash_retry,
                                    "res_level": 3, # Lower resolution (1/8th)
                                    "use_raw_all": att['use_raw_all'],
                                    "threads": 4 # Limit threads to 4
                                })
                
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
