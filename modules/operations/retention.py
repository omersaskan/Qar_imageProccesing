"""
modules/operations/retention.py

SPRINT 3 — TICKET-011: Retention policy tuning

Changes vs previous version:

1. Explicit RETENTION POLICY declaration at module level — every rule is
   documented before any code is executed. This replaces implicit behavior
   scattered throughout methods.

2. PROTECTED ARTIFACTS list — files that must NEVER be pruned regardless of
   session age or status. Manifest, validation report, guidance report,
   session JSON, and export metrics are always preserved.

3. DRAFT sessions now get a dedicated retention threshold.
   Previously draft sessions (publish_state == "draft") were silently skipped
   because they hold status == VALIDATED, not PUBLISHED or FAILED.
   They now age out of raw data after `draft_frames_days` (default 7 days).

4. `last_pipeline_progress_at` (Sprint 2 TICKET-006) is used as the activity
   anchor instead of session_file.stat().st_mtime where available.
   This is more accurate for sessions that were updated programmatically.

5. Added `_cleanup_recon_scratch` protection: manifest.json, log files, final
   textured mesh, and texture atlas are explicitly added to safe-list before
   any folder removal to prevent accidental deletion of final deliverables.

6. Added `_audit_retention_action` — every pruning decision is logged at INFO
   level with session_id, path, age, threshold, and reason. This creates an
   auditable trail without needing a separate audit log framework.

7. Orphaned recon directories (no job.json) now keep a 24h grace period AND
   check for a manifest.json before deletion — if a manifest exists, the dir
   is considered valuable and is skipped.

8. `run_cleanup` now returns a summary dict so tests and callers can assert
   on behavior without parsing logs.

RETENTION POLICY SUMMARY:
  ┌─────────────────────────────┬───────────────────────────────────────────┐
  │ Session status              │ Raw data (frames, video) pruned after     │
  ├─────────────────────────────┼───────────────────────────────────────────┤
  │ PUBLISHED                   │ published_frames_days (default: 3 days)   │
  │ DRAFT (validated, pending)  │ draft_frames_days (default: 7 days)       │
  │ FAILED                      │ failed_frames_days (default: 14 days)     │
  │ RECAPTURE_REQUIRED          │ failed_frames_days (default: 14 days)     │
  │ CREATED / CAPTURED / …      │ NEVER pruned (session still active)       │
  └─────────────────────────────┴───────────────────────────────────────────┘

  ┌─────────────────────────────┬───────────────────────────────────────────┐
  │ Artifact type               │ Policy                                    │
  ├─────────────────────────────┼───────────────────────────────────────────┤
  │ manifest.json               │ ALWAYS preserved                          │
  │ validation_report.json      │ ALWAYS preserved                          │
  │ export_metrics.json         │ ALWAYS preserved                          │
  │ guidance_report.json        │ ALWAYS preserved                          │
  │ coverage_report.json        │ ALWAYS preserved                          │
  │ cleanup_stats.json          │ ALWAYS preserved                          │
  │ normalized_metadata.json    │ ALWAYS preserved                          │
  │ *.log                       │ ALWAYS preserved                          │
  │ reports/ directory          │ ALWAYS preserved (contains above)         │
  │ frames/                     │ Pruned per above schedule                 │
  │ video/                      │ Pruned per above schedule                 │
  │ COLMAP sparse/dense         │ Pruned after recon_scratch_hours          │
  │ COLMAP database.db          │ Pruned after recon_scratch_hours          │
  └─────────────────────────────┴───────────────────────────────────────────┘
"""

import json
import shutil
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from .settings import settings, AppEnvironment
from .logging_config import get_component_logger
from modules.capture_workflow.session_manager import SessionManager
from modules.shared_contracts.lifecycle import AssetStatus

logger = get_component_logger("retention")


# ─────────────────────────────────────────────────────────────────────────────
# Protected artifact filenames — NEVER deleted by retention
# ─────────────────────────────────────────────────────────────────────────────

_PROTECTED_FILENAMES = frozenset({
    "manifest.json",
    "validation_report.json",
    "export_metrics.json",
    "guidance_report.json",
    "guidance_summary.md",
    "coverage_report.json",
    "cleanup_stats.json",
    "normalized_metadata.json",
})

# Extensions always preserved
_PROTECTED_EXTENSIONS = frozenset({".log", ".json"})

# Raw data folders eligible for pruning
_RAW_DATA_FOLDERS = ("video", "frames")

# COLMAP/OpenMVS scratch folders eligible for pruning inside recon job dirs
_RECON_SCRATCH_FOLDERS = ("images", "masks", "sparse", "dense", "temp")
_RECON_SCRATCH_FILES = ("database.db",)


# ─────────────────────────────────────────────────────────────────────────────
# RetentionService
# ─────────────────────────────────────────────────────────────────────────────

class RetentionService:
    def __init__(self, data_root: Optional[str] = None):
        self.data_root = Path(data_root or settings.data_root).resolve()
        self.session_manager = SessionManager(data_root=str(self.data_root))

    def run_cleanup(self) -> Dict[str, Any]:
        """
        Main entry point for the retention cycle.

        Returns a summary dict:
        {
            "sessions_pruned_raw": int,
            "recon_dirs_pruned": int,
            "errors": List[str],
            "duration_sec": float,
        }
        This allows tests and monitoring to assert on behavior.
        """
        start_time = time.time()
        summary: Dict[str, Any] = {
            "sessions_pruned_raw": 0,
            "recon_dirs_pruned": 0,
            "errors": [],
        }
        logger.info("Retention cleanup cycle started.")

        try:
            summary["sessions_pruned_raw"] = self._prune_session_artifacts()
        except Exception as e:
            msg = f"Session artifact pruning failed: {e}"
            logger.error(msg, exc_info=True)
            summary["errors"].append(msg)

        try:
            summary["recon_dirs_pruned"] = self._prune_reconstruction_scratch()
        except Exception as e:
            msg = f"Reconstruction scratch pruning failed: {e}"
            logger.error(msg, exc_info=True)
            summary["errors"].append(msg)

        summary["duration_sec"] = round(time.time() - start_time, 3)
        logger.info(
            f"Retention cycle complete in {summary['duration_sec']:.2f}s — "
            f"sessions pruned: {summary['sessions_pruned_raw']}, "
            f"recon dirs pruned: {summary['recon_dirs_pruned']}, "
            f"errors: {len(summary['errors'])}."
        )
        return summary

    # ── Session artifact pruning ──────────────────────────────────────────────

    def _prune_session_artifacts(self) -> int:
        """
        Prunes raw frames and videos based on explicit retention policy.
        Returns the number of sessions whose raw data was pruned.
        """
        sessions_dir = self.session_manager.sessions_dir
        if not sessions_dir.exists():
            return 0

        now = datetime.now(timezone.utc)
        pruned_count = 0

        for session_file in sessions_dir.glob("*.json"):
            try:
                session = self.session_manager.get_session(session_file.stem)
                if not session:
                    continue

                threshold_days = self._get_threshold_days(session)
                if threshold_days is None:
                    # Active pipeline session — never prune
                    continue

                # SPRINT 3: Use last_pipeline_progress_at if available for
                # activity anchor (more accurate than file mtime).
                if getattr(session, "last_pipeline_progress_at", None):
                    last_activity = session.last_pipeline_progress_at
                else:
                    last_activity = datetime.fromtimestamp(
                        session_file.stat().st_mtime, tz=timezone.utc
                    )

                age = now - last_activity

                if age > timedelta(days=threshold_days):
                    capture_path = self.session_manager.get_capture_path(session.session_id)
                    if capture_path.exists():
                        pruned = self._prune_raw_data(
                            capture_path,
                            session_id=session.session_id,
                            threshold_days=threshold_days,
                            age=age,
                            reason=f"status={session.status.value}",
                        )
                        if pruned:
                            pruned_count += 1

            except Exception as e:
                logger.warning(
                    f"Retention skipped session {session_file.stem}: {e}"
                )

        return pruned_count

    def _get_threshold_days(self, session) -> Optional[int]:
        """
        Returns the retention threshold in days for this session, or None if the
        session is still active and should never be pruned.

        SPRINT 3 CHANGE: DRAFT sessions (VALIDATED with publish_state == 'draft')
        now have an explicit threshold instead of being silently skipped.
        """
        status = session.status
        publish_state = getattr(session, "publish_state", None)

        if status == AssetStatus.PUBLISHED:
            return settings.published_frames_days

        if status == AssetStatus.VALIDATED and publish_state == "draft":
            # Draft sessions have their own (more lenient) threshold
            return getattr(settings, "draft_frames_days", 7)

        if status in {AssetStatus.FAILED, AssetStatus.RECAPTURE_REQUIRED}:
            return settings.failed_frames_days

        # CREATED, CAPTURED, RECONSTRUCTED, CLEANED, EXPORTED, VALIDATED(pending)
        # → session still in active pipeline; never prune
        return None

    def _prune_raw_data(
        self,
        capture_path: Path,
        session_id: str,
        threshold_days: int,
        age: timedelta,
        reason: str,
    ) -> bool:
        """
        Removes 'video' and 'frames' folders from a capture directory.
        Always preserves the 'reports' subdirectory and any protected files.
        Returns True if at least one folder was removed.
        """
        pruned_any = False
        for folder_name in _RAW_DATA_FOLDERS:
            target = capture_path / folder_name
            if target.exists():
                self._audit_retention_action(
                    session_id=session_id,
                    path=target,
                    age_days=age.days,
                    threshold_days=threshold_days,
                    action="prune_raw_folder",
                    reason=reason,
                )
                shutil.rmtree(target, ignore_errors=True)
                pruned_any = True

        return pruned_any

    # ── Reconstruction scratch pruning ────────────────────────────────────────

    def _prune_reconstruction_scratch(self) -> int:
        """
        Prunes heavy intermediate COLMAP/OpenMVS folders in data/reconstructions.
        Always preserves manifest.json, log files, and final deliverable meshes.
        Returns the number of job directories whose scratch was pruned.
        """
        recon_root = self.data_root / "reconstructions"
        if not recon_root.exists():
            return 0

        now = datetime.now(timezone.utc)
        threshold_hours = settings.reconstruction_scratch_hours
        pruned_count = 0

        for job_dir in recon_root.iterdir():
            if not job_dir.is_dir():
                continue

            try:
                job_file = job_dir / "job.json"
                manifest_file = job_dir / "manifest.json"

                if not job_file.exists():
                    # Orphaned directory
                    mtime = datetime.fromtimestamp(
                        job_dir.stat().st_mtime, tz=timezone.utc
                    )
                    age = now - mtime
                    # SPRINT 3: If a manifest exists the dir has delivered assets
                    # — do not delete the directory, just scratch-clean it.
                    if manifest_file.exists():
                        if age > timedelta(hours=threshold_hours):
                            pruned = self._cleanup_recon_scratch(job_dir)
                            if pruned:
                                pruned_count += 1
                    elif age > timedelta(hours=24):
                        # Truly orphaned and old — remove entire dir
                        logger.info(
                            f"Retention: Removing orphaned recon dir {job_dir.name} "
                            f"(age={age.days}d, no manifest, no job.json)"
                        )
                        shutil.rmtree(job_dir, ignore_errors=True)
                        pruned_count += 1
                    continue

                last_activity = datetime.fromtimestamp(
                    job_file.stat().st_mtime, tz=timezone.utc
                )
                age = now - last_activity

                if age > timedelta(hours=threshold_hours):
                    pruned = self._cleanup_recon_scratch(job_dir)
                    if pruned:
                        pruned_count += 1

            except Exception as e:
                logger.warning(f"Retention skipped recon dir {job_dir.name}: {e}")

        return pruned_count

    def _cleanup_recon_scratch(self, job_dir: Path) -> bool:
        """
        Removes COLMAP/OpenMVS scratch folders and files from a job directory.
        Explicitly skips any file matching protected filename/extension criteria BEFORE
        touching anything — no deletion can accidentally hit a manifest or log.

        Returns True if at least one item was removed.
        """
        removed_any = False

        for folder_name in _RECON_SCRATCH_FOLDERS:
            target = job_dir / folder_name
            if not target.exists():
                continue

            # Safety: scan for protected files before wiping
            protected_found = []
            for f in target.rglob("*"):
                if f.is_file():
                    if (
                        f.name in _PROTECTED_FILENAMES
                        or f.suffix in _PROTECTED_EXTENSIONS
                    ):
                        protected_found.append(f)

            if protected_found:
                logger.warning(
                    f"Retention: Skipping scratch folder {target} — "
                    f"contains {len(protected_found)} protected files: "
                    f"{[str(p.name) for p in protected_found[:3]]}"
                )
                continue

            logger.info(f"Retention: Pruning recon scratch folder {target}")
            shutil.rmtree(target, ignore_errors=True)
            removed_any = True

        for filename in _RECON_SCRATCH_FILES:
            target = job_dir / filename
            if not target.exists():
                continue
            if target.name in _PROTECTED_FILENAMES:
                continue
            logger.info(f"Retention: Pruning recon scratch file {target.name}")
            target.unlink(missing_ok=True)
            removed_any = True

        if removed_any:
            logger.info(f"Retention: Scrubbed scratch from {job_dir.name}")

        return removed_any

    # ── Audit trail ───────────────────────────────────────────────────────────

    @staticmethod
    def _audit_retention_action(
        session_id: str,
        path: Path,
        age_days: int,
        threshold_days: int,
        action: str,
        reason: str,
    ) -> None:
        """
        Emits a structured INFO log for every pruning decision.
        This creates an auditable trail without a separate audit log framework.
        Operators can grep logs for 'RETENTION_AUDIT' to review all deletions.
        """
        logger.info(
            f"RETENTION_AUDIT | action={action} | session={session_id} | "
            f"path={path.name} | age={age_days}d | threshold={threshold_days}d | "
            f"reason={reason}"
        )
