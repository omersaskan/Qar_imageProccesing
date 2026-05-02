import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import trimesh

from modules.operations.settings import settings, ReconstructionPipeline, AppEnvironment
from modules.shared_contracts.models import (
    ReconstructionJob,
    ReconstructionAttemptResult,
    ReconstructionAttemptType,
    ReconstructionAudit,
)
from modules.utils.file_persistence import atomic_write_json, calculate_checksum
from .adapter import COLMAPAdapter, ReconstructionAdapter, SimulatedAdapter
from .failures import (
    InsufficientInputError,
    MissingArtifactError,
    RuntimeReconstructionError,
    InsufficientReconstructionError,
)
from .output_manifest import MeshMetadata, OutputManifest
from modules.utils.mesh_inspection import get_mesh_stats_cheaply


class ReconstructionRunner:
    def __init__(self, adapter: Optional[ReconstructionAdapter] = None):
        self._explicit_adapter = adapter
        self._effective_settings = None  # set per-job in run() when a profile manifest exists

    def _resolve_effective_settings(self, job: "ReconstructionJob"):
        """
        Walk up from job_dir / capture session to find session_capture_profile.json
        or extraction_manifest.json, build a Settings clone with profile overrides.
        Returns the global settings unchanged when no profile is found.
        """
        try:
            from modules.operations.capture_profile import (
                CaptureProfile, apply_profile_to_settings,
            )
            search_roots = []
            try:
                search_roots.append(Path(job.job_dir))
                search_roots.extend(list(Path(job.job_dir).parents)[:4])
            except Exception:
                pass
            try:
                search_roots.append(Path(settings.data_root) / "captures" / job.capture_session_id)
            except Exception:
                pass

            cp = None
            for root in search_roots:
                if not root or not root.exists():
                    continue
                # Direct hits
                for fname in ("session_capture_profile.json", "extraction_manifest.json"):
                    cand = root / fname
                    if cand.exists():
                        try:
                            with open(cand, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            payload = data if fname == "session_capture_profile.json" else data.get("capture_profile")
                            if isinstance(payload, dict) and payload:
                                cp = CaptureProfile.from_dict(payload)
                                break
                        except Exception:
                            continue
                if cp:
                    break

            if cp:
                eff = apply_profile_to_settings(cp, settings)
                logging.info(
                    f"Reconstruction effective settings from profile {cp.preset_key}: "
                    f"max_image_size={eff.recon_max_image_size}, "
                    f"poisson_depth={eff.recon_poisson_depth}, "
                    f"mesh_budget={eff.recon_mesh_budget_faces}, "
                    f"texture_target={eff.texture_texturing_target_faces}"
                )
                return eff
        except Exception as e:
            logging.warning(f"Profile resolve failed for reconstruction; using env settings: {e}")
        return settings

    def _run_preset_hardening(self, job: "ReconstructionJob") -> Dict[str, Any]:
        """
        Sprint 4 — derive profile + preflight + preset, plus an intrinsics_cache
        lookup.  Returned block is written to manifest under
        `reconstruction_hardening`.
        """
        from .reconstruction_profile import derive_profile
        from .reconstruction_preset_resolver import resolve_preset, get_preset_by_name, PRESET_NAME_BASELINE
        from .reconstruction_preflight import evaluate_preflight, PreflightDecision
        from .intrinsics_cache import IntrinsicsCache, disabled_lookup

        # 1. Load extraction manifest from session captures dir
        em: Dict[str, Any] = {}
        try:
            cap_dir = Path(settings.data_root) / "captures" / job.capture_session_id / "frames"
            mf = cap_dir / "extraction_manifest.json"
            if mf.exists():
                em = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            em = {}

        profile = derive_profile(extraction_manifest=em, selected_keyframe_count=len(job.input_frames or []))

        # 2. Preflight on the keyframe set
        preflight = evaluate_preflight(
            selected_keyframes=list(job.input_frames or []),
            capture_gate=em.get("capture_gate"),
        )

        # 3. Preset selection — review degrades to baseline; reject keeps profile_safe
        if preflight.decision == PreflightDecision.REVIEW:
            preset = get_preset_by_name(PRESET_NAME_BASELINE, profile)
            preset["rationale"] = "preflight=review → baseline (safe path)"
        else:
            preset = resolve_preset(profile)

        # 4. Intrinsics cache (probe only — does not yet feed COLMAP)
        intrinsics_status: Dict[str, Any] = {"status": "disabled", "cache_key": "disabled", "source": "default"}
        if getattr(settings, "intrinsics_cache_enabled", False):
            try:
                first_frame = (job.input_frames or [None])[0]
                w, h = 0, 0
                if first_frame:
                    img = self._read_image(Path(first_frame))
                    if img is not None:
                        h, w = img.shape[:2]
                if w > 0 and h > 0:
                    cache = IntrinsicsCache(Path(settings.data_root) / "intrinsics_cache.json")
                    res = cache.lookup(width=w, height=h)
                    intrinsics_status = res.to_dict()
            except Exception as e:
                intrinsics_status = {"status": "disabled", "cache_key": "error", "source": "default",
                                      "error": str(e)[:200]}

        # Sprint 4.5: build typed command config from preset (always non-None)
        from .reconstruction_command_config import from_preset
        command_config = from_preset(preset)

        runtime_fallback_enabled = bool(
            getattr(settings, "reconstruction_runtime_fallback_enabled", False)
        )
        # Sprint 4.6: hardening_mode encodes whether the runtime loop is wired.
        # disabled         — preset hardening flag is False (block not built).
        # manifest_only    — flag True but runtime loop disabled (Sprint 4.5).
        # runtime_enforced — flag True + runtime fallback wired (Sprint 4.6).
        hardening_mode = "runtime_enforced" if runtime_fallback_enabled else "manifest_only"

        return {
            "version": "v1.6",
            "enabled": True,
            "hardening_mode": hardening_mode,
            "runtime_fallback_enabled": runtime_fallback_enabled,
            "profile": profile.to_dict(),
            "preflight": preflight.to_dict(),
            "preset": preset,
            "active_preset": preset.get("name"),
            "command_config": command_config.to_dict(),
            "intrinsics_cache": intrinsics_status,
            "attempts": [],
            "fallback_attempts": [],  # legacy alias, kept for backward compat
            "final_attempt": 0,
            "final_status": "pending",
            # Cached object the runner uses to build adapters (not serialized).
            "_command_config_obj": command_config,
        }

    def _write_hardening_manifest(self, job_dir: Path, block: Dict[str, Any]):
        """Write reconstruction_hardening.json next to the audit; don't touch manifest.json."""
        try:
            # Strip non-serializable cached object before writing
            serializable = {k: v for k, v in block.items() if not k.startswith("_")}
            atomic_write_json(job_dir / "reconstruction_hardening.json", serializable)
        except Exception as e:
            logging.warning(f"Failed to write reconstruction_hardening.json: {e}")

    def _record_fallback_attempt(
        self,
        attempt_num: int,
        preset_name: str,
        status: str,
        failure_class: Optional[str] = None,
        exit_code: Optional[int] = None,
        next_action: Optional[str] = None,
        error_excerpt: Optional[str] = None,
        command_config: Optional[Dict[str, Any]] = None,
        started_at: Optional[str] = None,
        finished_at: Optional[str] = None,
        next_preset: Optional[str] = None,
        error_summary: Optional[str] = None,
    ):
        """
        Sprint 4.5/4.6: append an attempt record to hardening block + advance
        the command_config if a next_action is supplied.  Caller orchestrates
        the actual retry loop; this only records bookkeeping.

        Sprint 4.6 added fields (all optional, additive):
          - attempt_index, preset_name, command_config snapshot,
            started_at, finished_at, next_preset, error_summary
        """
        block = getattr(self, "_hardening_block", None)
        if not block:
            return
        excerpt = (error_excerpt or error_summary or "")
        excerpt = excerpt[:240] if excerpt else None
        summary = (error_summary or error_excerpt or "")
        summary = summary[:240] if summary else None
        next_preset_val = next_preset or next_action

        record = {
            # legacy keys (Sprint 4.5)
            "attempt": int(attempt_num),
            "preset": preset_name,
            "status": status,
            "failure_class": failure_class,
            "exit_code": exit_code,
            "next_action": next_action or next_preset,
            "error_excerpt": excerpt,
            # Sprint 4.6 keys
            "attempt_index": int(attempt_num),
            "preset_name": preset_name,
            "command_config": command_config,
            "started_at": started_at,
            "finished_at": finished_at,
            "error_summary": summary,
            "next_preset": next_preset_val,
        }
        block.setdefault("fallback_attempts", []).append(record)
        block.setdefault("attempts", []).append(record)
        block["final_attempt"] = int(attempt_num)
        if status == "passed":
            block["final_status"] = "reconstructed"
            block["active_preset"] = preset_name
        elif next_preset_val is None and status == "failed":
            block["final_status"] = "failed"
        else:
            block["final_status"] = "retrying"

    def _swap_to_next_preset(self, error_excerpt: Optional[str]) -> Optional[str]:
        """
        Pick a fallback preset, swap _hardening_block.{_command_config_obj, command_config},
        invalidate cached adapters so next access rebuilds with the new config.

        Returns the new preset name, or None if the ladder is exhausted /
        max_attempts reached.
        """
        block = getattr(self, "_hardening_block", None)
        if not block:
            return None

        from .fallback_ladder import pick_next_preset, FallbackAttempt
        from .reconstruction_command_config import from_preset
        from .reconstruction_profile import ReconstructionProfile

        attempts = block.get("fallback_attempts", []) or []
        max_attempts = int(getattr(settings, "fallback_ladder_max_attempts", 3) or 0)
        if len(attempts) >= max_attempts:
            block["final_status"] = "failed"
            return None

        # Build prior FallbackAttempt list for ladder de-dup
        prior_used: List[FallbackAttempt] = [
            FallbackAttempt(
                step_index=i,
                preset_name=str(a.get("preset", "")),
                triggered_by="recorded",
            )
            for i, a in enumerate(attempts)
        ]

        # Reconstruct profile from manifest dict
        try:
            profile_dict = block.get("profile") or {}
            profile = ReconstructionProfile()
            for k, v in profile_dict.items():
                if hasattr(profile, k):
                    try:
                        setattr(profile, k, type(getattr(profile, k))(v))
                    except Exception:
                        pass
        except Exception:
            profile = None

        nxt = pick_next_preset(profile=profile, error_excerpt=error_excerpt,
                               attempts_so_far=prior_used)
        if not nxt:
            block["final_status"] = "failed"
            return None

        # Swap command_config in block + invalidate cached adapters
        new_cfg = from_preset(nxt.preset_snapshot)
        block["_command_config_obj"] = new_cfg
        block["command_config"] = new_cfg.to_dict()
        block["preset"] = nxt.preset_snapshot
        block["active_preset"] = nxt.preset_name
        for attr in ("_colmap_cached", "_openmvs_cached"):
            if hasattr(self, attr):
                delattr(self, attr)
        return nxt.preset_name

    @staticmethod
    def _classify_attempt_failure(exc: BaseException) -> Tuple[str, Optional[int], str]:
        """
        Sprint 4.6: classify a reconstruction attempt failure into a stable
        crash class for the fallback ladder.

        Returns (failure_class, exit_code_or_None, error_summary).

        Classes:
          - native_crash : exit 3221226505 / 0xC0000005 / TextureMesh native crash
          - oom          : "out of memory" / "memory allocation"
          - missing_file : "no such file" / "file not found" / MissingArtifactError
          - timeout      : "timeout" / "timed out"
          - unknown      : everything else
        """
        from .failures import MissingArtifactError, TexturingFailed

        msg = str(exc) if exc is not None else ""
        if not msg:
            msg = repr(exc)
        lo = msg.lower()
        upper = msg.upper()

        exit_code: Optional[int] = None
        ec_attr = getattr(exc, "exit_code", None)
        if isinstance(ec_attr, int):
            exit_code = ec_attr

        # Missing artifact / file
        if isinstance(exc, MissingArtifactError):
            return "missing_file", exit_code, msg[:240]
        if "no such file" in lo or "file not found" in lo or "missing artifact" in lo:
            return "missing_file", exit_code, msg[:240]

        # Native crash — TextureMesh on Windows
        if (
            "3221226505" in msg
            or "0XC0000005" in upper
            or "TEXTUREMESH_NATIVE_CRASH" in upper
            or "native crash" in lo
            or isinstance(exc, TexturingFailed)
        ):
            if exit_code is None:
                exit_code = 3221226505
            return "native_crash", exit_code, msg[:240]

        # OOM
        if (
            "out of memory" in lo
            or "memory allocation" in lo
            or "cuda error: out of memory" in lo
            or "oom" in lo
        ):
            return "oom", exit_code, msg[:240]

        # Timeout
        if "timeout" in lo or "timed out" in lo:
            return "timeout", exit_code, msg[:240]

        return "unknown", exit_code, msg[:240]

    def _runtime_fallback_active(self) -> bool:
        """Sprint 4.6: gate for the new preset-driven retry loop."""
        block = getattr(self, "_hardening_block", None)
        if not block:
            return False
        if not bool(getattr(settings, "reconstruction_runtime_fallback_enabled", False)):
            return False
        return True

    def _run_runtime_fallback_loop(
        self,
        job: ReconstructionJob,
        validated_frames: List[str],
        audit: ReconstructionAudit,
        job_dir: Path,
    ) -> Tuple[Optional[Dict[str, Any]], int, str]:
        """
        Sprint 4.6: preset-aware retry loop replacing the legacy fallback_steps
        loop when hardening + runtime fallback are both enabled.

        Returns (best_results_or_None, best_index, engine_used). Records each
        attempt to both the audit and the hardening block.  Caps at
        settings.fallback_ladder_max_attempts.

        On exhaustion without success, returns (None, -1, "").  Caller decides
        whether to raise InsufficientReconstructionError.
        """
        block = self._hardening_block
        max_attempts = max(1, int(getattr(settings, "fallback_ladder_max_attempts", 3) or 1))

        best_results: Optional[Dict[str, Any]] = None
        best_index = -1
        best_score = float("-inf")
        best_engine = ""

        attempt_num = 0
        while attempt_num < max_attempts:
            attempt_num += 1
            current_cfg = block.get("_command_config_obj")
            preset_name = (block.get("preset") or {}).get("name") or block.get("active_preset") or "baseline"
            cfg_snapshot = current_cfg.to_dict() if current_cfg is not None else None

            attempt_dir = job_dir / f"attempt_{attempt_num}_{preset_name}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            block["active_preset"] = preset_name

            started_at = datetime.now(timezone.utc).isoformat()
            logging.info(
                f"[runtime-fallback] attempt {attempt_num}/{max_attempts} preset={preset_name}"
            )

            try:
                current_adapter = self.adapter
                execution_dir, final_dir = self._prepare_execution_workspace(job)
                if final_dir:
                    final_dir = attempt_dir
                else:
                    execution_dir = attempt_dir

                results = current_adapter.run_reconstruction(
                    validated_frames,
                    execution_dir,
                    density=1.0,
                    enforce_masks=True,
                )

                if final_dir:
                    self._sync_workspace_back(execution_dir, final_dir)
                    results = self._remap_artifact_paths(results, execution_dir, final_dir)
                    shutil.rmtree(execution_dir, ignore_errors=True)

                engine_used = current_adapter.engine_type
                if isinstance(results, dict):
                    results["engine_used"] = engine_used

                score = self._score_attempt(results)

                attempt_res = ReconstructionAttemptResult(
                    attempt_type=ReconstructionAttemptType.DEFAULT,
                    status="success",
                    frames_used=len(validated_frames),
                    registered_images=results.get("registered_images", 0),
                    sparse_points=results.get("sparse_points", 0),
                    dense_points_fused=results.get("dense_points_fused", 0),
                    mesher_used=results.get("mesher_used", "none"),
                    mesh_path=results.get("mesh_path"),
                    log_path=results.get("log_path"),
                    metrics_rank_score=score,
                    metadata={
                        "engine": engine_used,
                        "attempt_dir": str(attempt_dir),
                        "preset": preset_name,
                        "runtime_fallback": True,
                    },
                )
                audit.attempts.append(attempt_res)

                finished_at = datetime.now(timezone.utc).isoformat()
                self._record_fallback_attempt(
                    attempt_num=attempt_num,
                    preset_name=preset_name,
                    status="passed",
                    command_config=cfg_snapshot,
                    started_at=started_at,
                    finished_at=finished_at,
                )

                if score > best_score:
                    best_score = score
                    best_results = results
                    best_index = len(audit.attempts) - 1
                    best_engine = engine_used
                # Success — stop the ladder.
                return best_results, best_index, best_engine

            except Exception as exc:
                finished_at = datetime.now(timezone.utc).isoformat()
                failure_class, exit_code, error_summary = self._classify_attempt_failure(exc)
                logging.warning(
                    f"[runtime-fallback] attempt {attempt_num} preset={preset_name} "
                    f"failed: class={failure_class} exit={exit_code}"
                )

                attempt_res = ReconstructionAttemptResult(
                    attempt_type=ReconstructionAttemptType.DEFAULT,
                    status="failed",
                    frames_used=0,
                    error_message=error_summary,
                    metrics_rank_score=-1000.0,
                    metadata={
                        "attempt_dir": str(attempt_dir),
                        "preset": preset_name,
                        "failure_class": failure_class,
                        "exit_code": exit_code,
                        "runtime_fallback": True,
                    },
                )
                audit.attempts.append(attempt_res)

                if failure_class == "missing_file":
                    self._record_fallback_attempt(
                        attempt_num=attempt_num,
                        preset_name=preset_name,
                        status="failed",
                        failure_class=failure_class,
                        exit_code=exit_code,
                        next_action=None,
                        next_preset=None,
                        error_excerpt=error_summary,
                        error_summary=error_summary,
                        command_config=cfg_snapshot,
                        started_at=started_at,
                        finished_at=finished_at,
                    )
                    return None, -1, ""

                # Decide next preset BEFORE recording, so we have next_action.
                attempts_before_swap = len(block.get("fallback_attempts", []))
                if attempts_before_swap + 1 >= max_attempts:
                    next_preset = None
                else:
                    next_preset = self._peek_next_preset(error_summary)

                self._record_fallback_attempt(
                    attempt_num=attempt_num,
                    preset_name=preset_name,
                    status="failed",
                    failure_class=failure_class,
                    exit_code=exit_code,
                    next_action=next_preset,
                    next_preset=next_preset,
                    error_excerpt=error_summary,
                    error_summary=error_summary,
                    command_config=cfg_snapshot,
                    started_at=started_at,
                    finished_at=finished_at,
                )

                if next_preset is None:
                    block["final_status"] = "failed"
                    return None, -1, ""

                # Swap to next preset (rebuilds adapter on next access).
                applied = self._swap_to_next_preset(error_summary)
                if applied is None:
                    block["final_status"] = "failed"
                    return None, -1, ""
                # Continue loop with new preset.

        # Exhausted attempts without success
        block["final_status"] = "failed"
        return None, -1, ""

    def _peek_next_preset(self, error_excerpt: Optional[str]) -> Optional[str]:
        """Return the name of the preset the ladder would pick, without applying it."""
        block = getattr(self, "_hardening_block", None)
        if not block:
            return None
        from .fallback_ladder import pick_next_preset, FallbackAttempt
        from .reconstruction_profile import ReconstructionProfile

        attempts = block.get("fallback_attempts", []) or []
        prior_used: List[FallbackAttempt] = [
            FallbackAttempt(
                step_index=i,
                preset_name=str(a.get("preset", "")),
                triggered_by="recorded",
            )
            for i, a in enumerate(attempts)
        ]
        try:
            profile_dict = block.get("profile") or {}
            profile = ReconstructionProfile()
            for k, v in profile_dict.items():
                if hasattr(profile, k):
                    try:
                        setattr(profile, k, type(getattr(profile, k))(v))
                    except Exception:
                        pass
        except Exception:
            profile = None

        nxt = pick_next_preset(profile=profile, error_excerpt=error_excerpt, attempts_so_far=prior_used)
        return nxt.preset_name if nxt else None

    def _current_command_config(self):
        """Sprint 4.5: pull the typed command config from hardening block, if active."""
        block = getattr(self, "_hardening_block", None)
        if block:
            return block.get("_command_config_obj")
        return None

    @property
    def colmap_adapter(self) -> "COLMAPAdapter":
        if not hasattr(self, "_colmap_cached"):
            try:
                settings.validate_setup()
            except (ValueError, FileNotFoundError) as e:
                if settings.env in [AppEnvironment.PRODUCTION, AppEnvironment.PILOT]:
                    raise RuntimeError(f"Production environment must be configured: {e}")
                logging.warning(f"Configuration warning: {e}")

            self._colmap_cached = COLMAPAdapter(
                settings_override=self._effective_settings,
                command_config=self._current_command_config(),
            )
        return self._colmap_cached

    @property
    def openmvs_adapter(self) -> "OpenMVSAdapter":
        if not hasattr(self, "_openmvs_cached"):
            try:
                settings.validate_setup()
            except (ValueError, FileNotFoundError) as e:
                if settings.env in [AppEnvironment.PRODUCTION, AppEnvironment.PILOT]:
                    raise RuntimeError(f"Production environment must be configured: {e}")
                logging.warning(f"Configuration warning: {e}")

            from .adapter import OpenMVSAdapter
            self._openmvs_cached = OpenMVSAdapter(
                settings_override=self._effective_settings,
                command_config=self._current_command_config(),
            )
        return self._openmvs_cached

    @property
    def adapter(self) -> "ReconstructionAdapter":
        if self._explicit_adapter:
            return self._explicit_adapter

        raw_choice = settings.recon_pipeline.lower()
        is_production = settings.env in [AppEnvironment.PILOT, AppEnvironment.PRODUCTION]

        if raw_choice in ["openmvs", "colmap_openmvs"]:
            choice = ReconstructionPipeline.COLMAP_OPENMVS
        elif raw_choice in ["colmap", "colmap_dense"]:
            choice = ReconstructionPipeline.COLMAP_DENSE
        elif raw_choice == "simulated":
            choice = ReconstructionPipeline.SIMULATED
        else:
            raise ValueError(
                f"Unsupported reconstruction pipeline: '{raw_choice}'. "
                f"Valid options: {[p.value for p in ReconstructionPipeline]}"
            )

        logging.info(f"Runner selected engine adapter: {choice.value} (from input: {raw_choice})")

        if choice == ReconstructionPipeline.COLMAP_OPENMVS:
            return self.openmvs_adapter
        if choice == ReconstructionPipeline.COLMAP_DENSE:
            return self.colmap_adapter
        if choice == ReconstructionPipeline.SIMULATED:
            if is_production:
                raise RuntimeError("Stub engine strictly prohibited in production.")
            allow_simulated = os.getenv("ALLOW_SIMULATED_RECONSTRUCTION", "false").lower() == "true"
            if not allow_simulated:
                raise RuntimeError("Simulated reconstruction is disabled by default locally.")
            if not hasattr(self, "_simulated_cached"):
                self._simulated_cached = SimulatedAdapter()
            return self._simulated_cached

        raise ValueError(f"Unhandled pipeline choice: {choice}")

    def _read_image(self, image_path: Path):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is not None:
            return image

        try:
            image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
            if image_bytes.size == 0:
                return None
            return cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _validate_input_frames(self, input_frames: List[str]) -> List[str]:
        valid_frames: List[str] = []
        invalid_reasons: List[str] = []

        for frame_path_str in input_frames:
            frame_path = Path(frame_path_str)
            if not frame_path.exists():
                invalid_reasons.append(f"{frame_path}: missing")
                continue
            if frame_path.stat().st_size <= 0:
                invalid_reasons.append(f"{frame_path}: zero-byte")
                continue

            frame = self._read_image(frame_path)
            if frame is None or frame.size == 0:
                invalid_reasons.append(f"{frame_path}: unreadable")
                continue

            valid_frames.append(str(frame_path))

        if invalid_reasons:
            raise InsufficientInputError(
                "Invalid reconstruction input frames detected: " + "; ".join(invalid_reasons)
            )

        if len(valid_frames) < 3:
            raise InsufficientInputError(
                f"At least 3 readable frames are required for reconstruction; got {len(valid_frames)}."
            )

        return valid_frames

    def _is_ascii_safe_path(self, path: Path) -> bool:
        try:
            str(path).encode("ascii")
            return True
        except UnicodeEncodeError:
            return False

    def _prepare_execution_workspace(self, job: ReconstructionJob) -> Tuple[Path, Optional[Path]]:
        job_dir = Path(job.job_dir).resolve()
        if os.name != "nt" or self._is_ascii_safe_path(job_dir):
            return job_dir, None

        ascii_root = Path(os.getenv("MESHYSIZ_ASCII_WORKROOT", r"C:\meshysiz_ascii"))
        ascii_workspace = ascii_root / job.job_id
        if ascii_workspace.exists():
            shutil.rmtree(ascii_workspace, ignore_errors=True)
        ascii_workspace.mkdir(parents=True, exist_ok=True)
        return ascii_workspace, job_dir

    def _sync_workspace_back(self, source_dir: Path, target_dir: Path) -> None:
        if source_dir.resolve() == target_dir.resolve():
            return

        for root, _, files in os.walk(source_dir):
            root_path = Path(root)
            relative_root = root_path.relative_to(source_dir)
            destination_root = target_dir / relative_root
            destination_root.mkdir(parents=True, exist_ok=True)
            for name in files:
                shutil.copy2(root_path / name, destination_root / name)

    def _remap_artifact_paths(self, results: dict, source_dir: Path, target_dir: Path) -> dict:
        remapped = dict(results)
        for key in ("mesh_path", "texture_path", "log_path"):
            value = remapped.get(key)
            if not value:
                continue

            source_path = Path(value)
            try:
                relative = source_path.resolve().relative_to(source_dir.resolve())
            except Exception:
                continue
            remapped[key] = str((target_dir / relative).resolve())

        return remapped

    def _validate_mesh_artifact(self, mesh_path: Path) -> Tuple[int, int]:
        if not mesh_path.exists():
            raise MissingArtifactError(mesh_path.name)

        stats = get_mesh_stats_cheaply(str(mesh_path))
        vertex_count = stats.get("vertex_count", 0)
        face_count = stats.get("face_count", 0)

        if vertex_count == 0 or face_count == 0:
            raise MissingArtifactError(f"{mesh_path.name}: empty or face-less mesh (detected by header scan)")

        return int(vertex_count), int(face_count)

    def _score_attempt(self, results: dict) -> float:
        """
        Ranks reconstruction attempts based on quality metrics.
        Higher is better.
        """
        if not results:
            return -1.0

        score = results.get("registered_images", 0) * 100.0
        score += results.get("sparse_points", 0) * 0.5
        score += results.get("dense_points_fused", 0) * 0.1

        if results.get("mesher_used") == "failed":
            score -= 5000.0
        elif results.get("mesher_used") in ["poisson", "delaunay", "simulated"]:
            score += 2000.0

        # Penalize unsafe mask conditions while still allowing the best empirical output to win.
        if results.get("force_unmasked_fusion"):
            score -= 250.0
        if results.get("dense_mask_valid") is False and results.get("mask_mode") == "masked":
            score -= 500.0

        texture_path = results.get("texture_path")
        has_texture = bool(texture_path and Path(texture_path).exists() and "_no_texture.png" not in str(texture_path))

        texture_penalty = 0.0
        if has_texture:
            score += 3000.0
        elif settings.require_textured_output and results.get("engine_used") != "colmap_dense":
            texture_penalty = -4000.0
            score += texture_penalty

        mesh_load_probe_ok = False
        mesh_probe_vertex_count = 0
        mesh_probe_face_count = 0
        mesh_probe_has_uv = False

        mesh_path = results.get("mesh_path")
        if mesh_path and Path(mesh_path).exists():
            try:
                from modules.utils.mesh_inspection import get_mesh_stats_cheaply
                stats = get_mesh_stats_cheaply(mesh_path)
                mesh_probe_face_count = stats["face_count"]
                mesh_probe_vertex_count = stats["vertex_count"]
                mesh_load_probe_ok = True # Stats extracted successfully
                
                # Only attempt full load for UV probing if mesh is small enough and likely textured
                safe_limit = settings.max_faces_python_decimation
                is_obj = str(mesh_path).lower().endswith(".obj")
                
                if mesh_probe_face_count < safe_limit and (is_obj or has_texture):
                    try:
                        mesh = trimesh.load(mesh_path, process=False)
                        if isinstance(mesh, trimesh.Scene):
                            mesh = mesh.dump(concatenate=True)
                        if isinstance(mesh, trimesh.Trimesh):
                            if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
                                mesh_probe_has_uv = True
                    except Exception:
                        pass
            except Exception:
                pass

        results["has_texture_file"] = has_texture
        results["require_textured_output"] = settings.require_textured_output
        results["texture_required_penalty"] = texture_penalty
        results["mesh_load_probe_ok"] = mesh_load_probe_ok
        results["mesh_probe_vertex_count"] = mesh_probe_vertex_count
        results["mesh_probe_face_count"] = mesh_probe_face_count
        results["mesh_probe_has_uv"] = mesh_probe_has_uv

        return score

    def run(self, job: ReconstructionJob) -> OutputManifest:
        if not job.input_frames or len(job.input_frames) < 3:
            raise InsufficientInputError("At least 3 high-quality frames are required for reconstruction.")

        # Resolve per-job effective settings from capture profile (if any).
        # Drop any cached adapter so the next .adapter access rebuilds with the override.
        self._effective_settings = self._resolve_effective_settings(job)
        for attr in ("_colmap_cached", "_openmvs_cached"):
            if hasattr(self, attr):
                delattr(self, attr)

        # ── Sprint 4: Reconstruction Preset Hardening (opt-in) ───────────
        # Builds reconstruction_hardening.{profile, preflight, preset, ...}
        # for the manifest.  Doesn't yet substitute COLMAP/OpenMVS args
        # (Sprint 4 v2 wires that); manifest visibility is the v1 deliverable.
        self._hardening_block = None
        if getattr(settings, "reconstruction_preset_hardening_enabled", False):
            try:
                self._hardening_block = self._run_preset_hardening(job)
                if self._hardening_block.get("preflight", {}).get("decision") == "reject":
                    job_dir = Path(job.job_dir).resolve()
                    job_dir.mkdir(parents=True, exist_ok=True)
                    # Sprint 4.6: mirror the audit verdict into the hardening
                    # block so manifest readers see the same final_status.
                    self._hardening_block["final_status"] = "capture_quality_rejected"
                    self._save_audit(
                        ReconstructionAudit(
                            capture_session_id=job.capture_session_id,
                            final_status="capture_quality_rejected",
                        ),
                        job_dir,
                    )
                    self._write_hardening_manifest(job_dir, self._hardening_block)
                    raise InsufficientInputError(
                        f"Preflight rejected reconstruction: "
                        f"{'; '.join(self._hardening_block['preflight'].get('reasons', []))}"
                    )
            except InsufficientInputError:
                raise
            except Exception as e:
                logging.warning(f"Preset hardening failed (non-fatal, continuing legacy path): {e}")
                self._hardening_block = None

        validated_frames = self._validate_input_frames(job.input_frames)

        fallback_steps = list(settings.recon_fallback_steps or ["default"])
        # SPRINT 4: Diagnostic override to ensure unmasked is tried
        if settings.recon_diagnostic_enable_unmasked and "unmasked" not in fallback_steps:
            fallback_steps.append("unmasked")

        audit = ReconstructionAudit(capture_session_id=job.capture_session_id)
        job_dir = Path(job.job_dir).resolve()
        job_dir.mkdir(parents=True, exist_ok=True)

        best_results = None
        best_score = float("-inf")
        best_index = -1
        run_start = time.monotonic()

        # ── Sprint 4.6: runtime fallback dispatch ───────────────────────
        # When hardening + runtime fallback are both enabled, drive attempts
        # through the preset-aware ladder instead of the legacy loop.  Legacy
        # behaviour is preserved verbatim when either flag is off.
        if self._runtime_fallback_active():
            best_results, best_index, best_engine = self._run_runtime_fallback_loop(
                job, validated_frames, audit, job_dir
            )
            if best_results is None:
                audit.final_status = "failed"
                self._save_audit(audit, job_dir)
                self._write_hardening_manifest(job_dir, self._hardening_block)
                last_err = (
                    audit.attempts[-1].error_message
                    if audit.attempts else "All preset fallback attempts failed."
                )
                raise InsufficientReconstructionError(
                    f"All preset fallback attempts failed. Last error: {last_err}"
                )
            audit.selected_best_index = best_index
            audit.final_status = "success"
            self._save_audit(audit, job_dir)
            elapsed_seconds = time.monotonic() - run_start
            return self._finalize_best_attempt(
                best_results, job, job_dir, best_engine, elapsed_seconds
            )

        for i, step_name in enumerate(fallback_steps):
            # Check for external cancellation before each attempt
            try:
                from modules.capture_workflow.session_manager import SessionManager
                sm = SessionManager(data_root=str(settings.data_root))
                persisted = sm.get_session(job.capture_session_id)
                if persisted and persisted.status == "failed":
                    logging.info(f"Reconstruction loop ABORTED for {job.capture_session_id} (Session Cancelled)")
                    raise RuntimeReconstructionError("Reconstruction cancelled by user.")
            except Exception as e:
                if "cancelled" in str(e): raise
                logging.debug(f"Cancellation check failed: {e}")

            attempt_type = ReconstructionAttemptType(step_name)

            if attempt_type == ReconstructionAttemptType.UNMASKED:
                # Allow if either enabled globally or via diagnostic toggle
                if not (settings.recon_unmasked_fallback_enabled or settings.recon_diagnostic_enable_unmasked):
                    logging.warning("Skipping UNMASKED fallback as it is disabled in settings.")
                    continue

            attempt_dir = job_dir / f"attempt_{i}_{step_name}"
            attempt_dir.mkdir(parents=True, exist_ok=True)

            logging.info(f"Starting reconstruction attempt {i}: type={step_name}")

            density = 1.0
            enforce_masks = True
            current_frames = validated_frames
            sampling_rate_used = None
            reextracted_frames_dir = None

            if attempt_type == ReconstructionAttemptType.DEFAULT:
                if ReconstructionAttemptType.DENSER_FRAMES.value in fallback_steps:
                    density = 0.5
            elif attempt_type == ReconstructionAttemptType.DENSER_FRAMES:
                density = 1.0
                if job.source_video_path and Path(job.source_video_path).exists():
                    sampling_rate_used = settings.recon_fallback_sample_rate
                    logging.info(
                        f"Attempt {i}: Re-extracting frames with denser sampling rate={sampling_rate_used}"
                    )
                    try:
                        from modules.capture_workflow.frame_extractor import FrameExtractor

                        extractor = FrameExtractor()
                        extractor.thresholds.frame_sample_rate = sampling_rate_used

                        extract_dir = attempt_dir / "extracted_frames"
                        extract_dir.mkdir(parents=True, exist_ok=True)
                        reextracted_frames_dir = str(extract_dir)

                        extracted = extractor.extract_keyframes(job.source_video_path, reextracted_frames_dir)

                        if isinstance(extracted, tuple):
                            new_frames, extraction_report = extracted
                        else:
                            new_frames = extracted
                            extraction_report = None

                        if extraction_report is not None:
                            atomic_write_json(
                                attempt_dir / "denser_extraction_report.json",
                                extraction_report,
                            )

                        if new_frames:
                            current_frames = [str(p) for p in new_frames]
                            logging.info(f"Attempt {i}: Denser extraction successful. count={len(current_frames)}")
                        else:
                            logging.warning("Attempt %s: Denser extraction produced no frames.", i)
                    except Exception as ex_err:
                        logging.error(f"Attempt {i}: Denser extraction failed: {ex_err}")
                else:
                    logging.warning("Attempt %s: source_video_path missing or invalid.", i)

            elif attempt_type == ReconstructionAttemptType.UNMASKED:
                enforce_masks = False
                density = 1.0

            current_frames = self._validate_input_frames([str(p) for p in current_frames])

            try:
                current_adapter = self.adapter
                primary_results = None
                primary_error = None

                try:
                    execution_dir, final_dir = self._prepare_execution_workspace(job)
                    if final_dir:
                        final_dir = attempt_dir
                    else:
                        execution_dir = attempt_dir

                    primary_results = current_adapter.run_reconstruction(
                        current_frames,
                        execution_dir,
                        density=density,
                        enforce_masks=enforce_masks,
                    )

                    if final_dir:
                        self._sync_workspace_back(execution_dir, final_dir)
                        primary_results = self._remap_artifact_paths(primary_results, execution_dir, final_dir)
                        shutil.rmtree(execution_dir, ignore_errors=True)

                except (InsufficientReconstructionError, RuntimeReconstructionError, InsufficientInputError) as e:
                    primary_error = e
                    logging.warning(f"Primary adapter ({current_adapter.engine_type}) failed for {step_name}: {e}")

                results = primary_results
                engine_used = current_adapter.engine_type
                if results and isinstance(results, dict):
                    results["engine_used"] = engine_used

                if primary_error and current_adapter.engine_type == "colmap_openmvs":
                    if settings.require_textured_output:
                        logging.error("OpenMVS failed and require_textured_output is True. Skipping COLMAP fallback.")
                    else:
                        logging.info(f"Attempting COLMAP fallback for {step_name}...")
                        try:
                            fallback_dir = attempt_dir / "colmap_fallback"
                            fallback_dir.mkdir(parents=True, exist_ok=True)

                            results = self.colmap_adapter.run_reconstruction(
                                current_frames,
                                fallback_dir,
                                density=density,
                                enforce_masks=enforce_masks,
                            )
                            engine_used = "colmap (fallback)"
                            primary_error = None
                        except Exception as fe:
                            logging.error(f"COLMAP fallback also failed for {step_name}: {fe}")
                            primary_error = fe

                if primary_error:
                    raise primary_error

                score = self._score_attempt(results)

                attempt_res = ReconstructionAttemptResult(
                    attempt_type=attempt_type,
                    status="success",
                    frames_used=len(current_frames),
                    registered_images=results.get("registered_images", 0),
                    sparse_points=results.get("sparse_points", 0),
                    dense_points_fused=results.get("dense_points_fused", 0),
                    mesher_used=results.get("mesher_used", "none"),
                    mesh_path=results.get("mesh_path"),
                    log_path=results.get("log_path"),
                    sampling_rate_used=sampling_rate_used,
                    source_video_path=job.source_video_path if attempt_type == ReconstructionAttemptType.DENSER_FRAMES else None,
                    reextracted_frames_dir=reextracted_frames_dir,
                    metrics_rank_score=score,
                    metadata={
                        "engine": engine_used,
                        "density": density,
                        "enforce_masks": enforce_masks,
                        "attempt_dir": str(attempt_dir),
                        "reextracted_frames_dir": reextracted_frames_dir,
                        "sampling_rate_used": sampling_rate_used,
                        "mesh_path": results.get("mesh_path"),
                        "texture_path": results.get("texture_path"),
                        "has_texture_file": results.get("has_texture_file"),
                        "mesh_probe_has_uv": results.get("mesh_probe_has_uv"),
                        "mesh_probe_vertex_count": results.get("mesh_probe_vertex_count"),
                        "mesh_probe_face_count": results.get("mesh_probe_face_count"),
                        "mask_mode": results.get("mask_mode"),
                        "feature_mask_path": results.get("feature_mask_path"),
                        "stereo_fusion_mask_path": results.get("stereo_fusion_mask_path"),
                        "dense_mask_valid": results.get("dense_mask_valid"),
                        "force_unmasked_fusion": results.get("force_unmasked_fusion"),
                        "dense_mask_count": results.get("dense_mask_count"),
                        "dense_mask_dimension_matches": results.get("dense_mask_dimension_matches"),
                        "dense_mask_exact_filename_matches": results.get("dense_mask_exact_filename_matches"),
                        "dense_mask_fallback_white_ratio": results.get("dense_mask_fallback_white_ratio"),
                    },
                )

                if score > best_score:
                    best_score = score
                    best_results = results
                    best_index = len(audit.attempts)

            except (InsufficientReconstructionError, RuntimeReconstructionError, InsufficientInputError) as e:
                attempt_res = ReconstructionAttemptResult(
                    attempt_type=attempt_type,
                    status="failed" if isinstance(e, RuntimeReconstructionError) else "weak",
                    frames_used=0,
                    error_message=str(e),
                    sampling_rate_used=sampling_rate_used,
                    source_video_path=job.source_video_path if attempt_type == ReconstructionAttemptType.DENSER_FRAMES else None,
                    reextracted_frames_dir=reextracted_frames_dir,
                    metrics_rank_score=-100.0,
                    metadata={
                        "attempt_dir": str(attempt_dir),
                        "density": density,
                        "enforce_masks": enforce_masks,
                    },
                )
                logging.warning(f"Attempt {i} ({step_name}) was weak or failed: {e}")
            except Exception as e:
                attempt_res = ReconstructionAttemptResult(
                    attempt_type=attempt_type,
                    status="failed",
                    frames_used=0,
                    error_message=f"Unexpected error: {str(e)}",
                    metrics_rank_score=-1000.0,
                    metadata={
                        "attempt_dir": str(attempt_dir),
                        "density": density,
                        "enforce_masks": enforce_masks,
                    },
                )
                logging.error(f"Attempt {i} ({step_name}) crashed: {e}")

            audit.attempts.append(attempt_res)

        if best_results is None:
            audit.final_status = "recapture_required"
            self._save_audit(audit, job_dir)
            last_err = audit.attempts[-1].error_message if audit.attempts else "All reconstruction attempts failed."
            raise InsufficientReconstructionError(f"All fallback attempts failed. Last error: {last_err}")
        
        audit.selected_best_index = best_index
        audit.final_status = "success"
        self._save_audit(audit, job_dir)

        logging.info(
            f"Reconstruction complete. Selected best attempt index: "
            f"{best_index} ({audit.attempts[best_index].attempt_type})"
        )

        best_engine = audit.attempts[best_index].metadata.get("engine", self.adapter.engine_type)
        elapsed_seconds = time.monotonic() - run_start
        return self._finalize_best_attempt(best_results, job, job_dir, best_engine, elapsed_seconds)

    def _save_audit(self, audit: ReconstructionAudit, job_dir: Path):
        audit_path = job_dir / "reconstruction_audit.json"
        atomic_write_json(audit_path, audit.model_dump(mode="json"))

    def _check_obj_uvs(self, obj_path: Path) -> bool:
        if not obj_path.exists() or obj_path.suffix.lower() != ".obj":
            return False
        try:
            with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("vt "):
                        return True
        except Exception:
            pass
        return False

    def _check_glb_uvs(self, glb_path: Path) -> bool:
        if not glb_path.exists() or glb_path.suffix.lower() != ".glb":
            return False
        try:
            from modules.export_pipeline.glb_exporter import inspect_glb_primitive_attributes
            rep = inspect_glb_primitive_attributes(str(glb_path))
            return bool(
                rep.get("all_textured_primitives_have_texcoord_0", False)
                and rep.get("texture_count", 0) > 0
            )
        except Exception:
            return False

    def _finalize_best_attempt(
        self,
        results: dict,
        job: ReconstructionJob,
        job_dir: Path,
        engine_used: str,
        elapsed_seconds: float = 0.0,
    ) -> OutputManifest:
        mesh_path = Path(results["mesh_path"])
        texture_path = Path(results.get("texture_path") or (job_dir / "_no_texture.png"))
        log_path = Path(results["log_path"])

        vertex_count, face_count = self._validate_mesh_artifact(mesh_path)
        checksum = calculate_checksum(mesh_path)

        uv_present = bool(results.get("mesh_probe_has_uv", False))
        if not uv_present:
            suffix = mesh_path.suffix.lower()
            if suffix == ".obj":
                uv_present = self._check_obj_uvs(mesh_path)
            elif suffix == ".glb":
                uv_present = self._check_glb_uvs(mesh_path)

        has_texture = texture_path.exists() and "_no_texture.png" not in str(texture_path)

        manifest = OutputManifest(
            job_id=job.job_id,
            mesh_path=str(mesh_path),
            textured_mesh_path=str(mesh_path) if has_texture else None,
            texture_path=str(texture_path) if has_texture else None,
            texture_atlas_paths=[str(texture_path)] if has_texture else [],
            log_path=str(log_path),
            processing_time_seconds=round(elapsed_seconds, 2),
            engine_type=engine_used,
            texturing_engine="openmvs_texturemesh" if has_texture and "openmvs" in engine_used.lower() else None,
            texturing_status="real" if has_texture else "absent",
            is_stub=self.adapter.is_stub,
            mesh_metadata=MeshMetadata(
                vertex_count=vertex_count,
                face_count=face_count,
                has_texture=has_texture,
                uv_present=uv_present,
            ),
            checksum=checksum,
        )

        manifest_path = job_dir / "manifest.json"
        manifest_data = manifest.model_dump(mode="json")
        # Sprint 4: backward-compat optional block
        if getattr(self, "_hardening_block", None):
            manifest_data["reconstruction_hardening"] = self._hardening_block
            self._write_hardening_manifest(job_dir, self._hardening_block)

        # Sprint 6: Blender headless cleanup (opt-in, default off)
        if getattr(settings, "blender_cleanup_enabled", False):
            try:
                from modules.asset_cleanup.blender_headless_worker import run_blender_cleanup
                from modules.asset_cleanup.mesh_normalization import NormalizationConfig
                from modules.export_pipeline.glb_export_manifest import (
                    build_blender_cleanup_block, write_blender_cleanup_sidecar
                )
                cleanup_cfg = NormalizationConfig(
                    decimate_enabled=bool(getattr(settings, "blender_cleanup_decimate_enabled", False)),
                    decimate_ratio=float(getattr(settings, "blender_cleanup_decimate_ratio", 0.5)),
                )
                output_glb = job_dir / "blender_cleanup" / f"{job.job_id}_clean.glb"
                worker_res = run_blender_cleanup(
                    input_path=str(mesh_path),
                    output_glb=str(output_glb),
                    config=cleanup_cfg,
                )
                cleanup_block = build_blender_cleanup_block(worker_res, original_mesh_path=str(mesh_path))
                manifest_data["blender_cleanup"] = cleanup_block
                write_blender_cleanup_sidecar(job_dir, cleanup_block)
                if worker_res.status == "ok" and worker_res.output_glb:
                    manifest_data["cleaned_glb_path"] = worker_res.output_glb
            except Exception as _ble:
                logging.warning(f"blender_cleanup failed (non-fatal): {_ble}")

        # Sprint 7: glTF-Transform optimization + Khronos validation gate (opt-in)
        if getattr(settings, "gltf_optimization_enabled", False) or getattr(settings, "gltf_validation_enabled", False):
            try:
                from modules.export_pipeline.gltf_transform_optimizer import optimize_glb, GltfTransformConfig
                from modules.qa_validation.gltf_validator import validate_glb
                from modules.qa_validation.ar_asset_gate import evaluate_ar_gate

                target_glb = manifest_data.get("cleaned_glb_path") or str(mesh_path)
                opt_result = None
                val_result = None

                if getattr(settings, "gltf_optimization_enabled", False):
                    opt_out = job_dir / "gltf_optimized" / f"{job.job_id}_opt.glb"
                    opt_result = optimize_glb(target_glb, str(opt_out))
                    manifest_data["gltf_optimization"] = opt_result.to_dict()
                    if opt_result.status == "ok" and opt_result.output_glb:
                        target_glb = opt_result.output_glb

                if getattr(settings, "gltf_validation_enabled", False):
                    val_result = validate_glb(target_glb)
                    manifest_data["gltf_validation"] = val_result.to_dict()

                gate = evaluate_ar_gate(opt_result, val_result)
                manifest_data["ar_asset_gate"] = gate.to_dict()
            except Exception as _s7e:
                logging.warning(f"Sprint 7 gltf pipeline failed (non-fatal): {_s7e}")

        # Sprint 5: pose-backed coverage (opt-in, default off)
        if getattr(settings, "pose_backed_coverage_enabled", False):
            try:
                from .pose_feedback import generate_pose_feedback
                attempt_dir = results.get("metadata", {}).get("attempt_dir") or str(job_dir)
                frame_count = len(getattr(job, "input_frames", None) or [])
                manifest_data["pose_backed_coverage"] = generate_pose_feedback(
                    attempt_dir, input_frame_count=frame_count
                )
            except Exception as _pfe:
                logging.warning(f"pose_backed_coverage generation failed (non-fatal): {_pfe}")

        atomic_write_json(manifest_path, manifest_data)

        # Sprint 8: license manifest + provenance (opt-in, default off)
        if getattr(settings, "license_manifest_enabled", False) or getattr(settings, "provenance_enabled", False):
            try:
                if getattr(settings, "license_manifest_enabled", False):
                    from modules.asset_registry.license_manifest import build_license_manifest
                    active_tools = ["colmap", "openmvs"]
                    if getattr(settings, "blender_cleanup_enabled", False):
                        active_tools.append("blender")
                    if getattr(settings, "gltf_optimization_enabled", False):
                        active_tools.append("gltf_transform")
                    if getattr(settings, "gltf_validation_enabled", False):
                        active_tools.append("gltf_validator")
                    lm = build_license_manifest(
                        asset_id=job.product_id,
                        source_video_path=getattr(job, "source_video_path", None),
                        active_tools=active_tools,
                    )
                    lm.write(job_dir / "license_manifest.json")
                    manifest_data["license_manifest_path"] = str(job_dir / "license_manifest.json")

                if getattr(settings, "provenance_enabled", False):
                    from modules.asset_registry.asset_provenance import provenance_from_manifest
                    prov = provenance_from_manifest(
                        asset_id=job.product_id,
                        job_id=job.job_id,
                        capture_session_id=job.capture_session_id,
                        manifest_data=manifest_data,
                    )
                    prov.write(job_dir / "asset_provenance.json")
                    manifest_data["provenance_path"] = str(job_dir / "asset_provenance.json")

                # Re-write manifest with updated paths
                atomic_write_json(manifest_path, manifest_data)
            except Exception as _s8e:
                logging.warning(f"Sprint 8 license/provenance failed (non-fatal): {_s8e}")

        # Sprint 1: write quality_report.json (scorecard) next to manifest.
        # All inputs optional — empty inputs produce a graded F but never raise.
        try:
            from modules.qa_validation.scorecard import build_scorecard, write_scorecard
            mesh_obj = None
            try:
                if mesh_path.exists():
                    mesh_obj = trimesh.load(str(mesh_path), force="mesh", process=False)
                    if isinstance(mesh_obj, trimesh.Scene):
                        mesh_obj = mesh_obj.dump(concatenate=True)
            except Exception:
                mesh_obj = None
            sc = build_scorecard(
                job_id=job.job_id,
                job_dir=job_dir,
                mesh=mesh_obj,
                texture_path=str(texture_path) if has_texture else None,
                expected_product_color=getattr(settings, "expected_product_color", "unknown"),
                extra_reconstruction_fields={
                    "engine_used": engine_used,
                    "elapsed_seconds": round(elapsed_seconds, 2),
                    "score": float(results.get("metrics_rank_score", 0.0)) if isinstance(results, dict) else 0.0,
                },
            )
            scorecard_path = write_scorecard(job_dir, sc)
            logging.info(f"Scorecard written: {scorecard_path} (grade={sc['overall']['grade']})")
        except Exception as e:
            logging.warning(f"Scorecard generation failed (non-fatal): {e}")

        return manifest

    def remesh_retry(self, job: ReconstructionJob, depth: int, trim: int) -> OutputManifest:
        """
        Retries meshing with different parameters from an existing fused.ply.
        """
        job_dir = Path(job.job_dir).resolve()
        # Find the last attempt directory that has a dense/fused.ply
        # In practice, for budget exceeded it's usually attempt_0_default or attempt_1_denser_frames
        attempts = sorted(job_dir.glob("attempt_*"), key=os.path.getmtime, reverse=True)
        
        target_attempt = None
        for attempt in attempts:
            if (attempt / "dense" / "fused.ply").exists():
                target_attempt = attempt
                break
        
        if not target_attempt:
            raise MissingArtifactError("No attempt with fused.ply found for remesh retry.")

        log_path = target_attempt / "reconstruction.log"
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n--- PROCESSING BUDGET EXCEEDED: Retrying with lower density ---")
            log_file.write(f"\nRecommended Settings: depth={depth}, trim={trim}\n")
            
            # We use COLMAPAdapter specifically for Poisson
            adapter = COLMAPAdapter()
            mesher_used = adapter.poisson_remesh_only(target_attempt, log_file, depth, trim)
            
            # We need to re-validate and update manifest
            mesh_path = target_attempt / "dense" / "meshed-poisson.ply"
            
            # Mock results dict for _finalize_best_attempt
            results = {
                "mesh_path": str(mesh_path),
                "log_path": str(log_path),
                "mesher_used": mesher_used,
                "registered_images": 0, # Not strictly needed for manifest update
                "sparse_points": 0,
            }
            
            return self._finalize_best_attempt(
                results,
                job,
                job_dir,
                "colmap_dense",
                elapsed_seconds=0.0 # Not timing retries yet
            )
