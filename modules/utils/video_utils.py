import subprocess
import os
import json
import fractions
from pathlib import Path
import logging

logger = logging.getLogger("video_utils")

def get_video_metadata(video_path: Path, ffprobe_path: str = "ffprobe", timeout: int = 30):
    """
    Uses ffprobe to extract video metadata including codec and pixel format.
    """
    if not ffprobe_path:
        raise RuntimeError("ffprobe_path is missing or None")

    cmd = [
        ffprobe_path,
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
        data = json.loads(result.stdout)
        
        video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
        if not video_stream:
            return None
            
        return {
            "codec_name": video_stream.get("codec_name"),
            "pix_fmt": video_stream.get("pix_fmt"),
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": float(fractions.Fraction(video_stream.get("r_frame_rate", "0/1"))),
            "duration": float(data.get("format", {}).get("duration", 0)),
            "frame_count": int(video_stream.get("nb_frames") or 0)
        }
    except subprocess.TimeoutExpired:
        logger.error(f"ffprobe timed out for {video_path}")
        return None
    except Exception as e:
        stderr_snippet = getattr(e, "stderr", str(e))[:200]
        logger.error(f"ffprobe failed for {video_path}: {stderr_snippet}")
        return None

def normalize_video(input_path: Path, output_path: Path, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe", timeout: int = 180):
    """
    Normalizes any supported video format to a standard MP4 using ffmpeg.
    Skips transcoding if already H.264 yuv420p MP4.
    """
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg_path is missing or None")

    if input_path.suffix.lower() == ".mp4":
        meta = get_video_metadata(input_path, ffprobe_path)
        if meta and meta["codec_name"] == "h264" and meta["pix_fmt"] == "yuv420p":
            logger.info(f"Skipping normalization for {input_path.name} (already H.264/yuv420p)")
            import shutil
            shutil.copy2(input_path, output_path)
            return True

    cmd = [
        ffmpeg_path,
        "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-strict", "experimental",
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg normalization timed out for {input_path}")
        raise RuntimeError(f"FFmpeg timed out after {timeout}s")
    except subprocess.CalledProcessError as e:
        stderr_snippet = e.stderr[-500:] if e.stderr else "No stderr"
        logger.error(f"FFmpeg normalization failed: {stderr_snippet}")
        raise RuntimeError(f"FFmpeg failed: {stderr_snippet}")
    except FileNotFoundError:
        logger.error("FFmpeg binary not found.")
        raise RuntimeError("FFmpeg not found on the system.")

def validate_video_file(video_path: Path, min_fps=0, min_duration=0, max_duration=9999):
    """
    Comprehensive validation of the final raw_video.mp4.
    Returns (ok, error_message, metadata)
    """
    import cv2
    metadata = {"fps": 0, "frame_count": 0, "duration": 0, "width": 0, "height": 0}
    
    if not video_path.exists():
        return False, "File missing", metadata
        
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return False, "cv2 cannot open video", metadata
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        metadata.update({
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height
        })

        if fps <= 0:
            return False, f"Invalid FPS: {fps}", metadata
        if frame_count <= 0:
            return False, f"Invalid frame count: {frame_count}", metadata
            
        ret, _ = cap.read()
        if not ret:
            return False, "Cannot read first frame", metadata
            
        duration = frame_count / fps
        metadata["duration"] = duration

        if fps < min_fps:
            return False, f"FPS too low: {fps:.1f} < {min_fps}", metadata
        if duration < min_duration:
            return False, f"Duration too short: {duration:.1f}s < {min_duration}s", metadata
        if duration > max_duration:
            return False, f"Duration too long: {duration:.1f}s > {max_duration}s", metadata
            
        return True, None, metadata
    finally:
        cap.release()

def check_video_readable(video_path: Path):
    """Legacy helper."""
    success, _, _ = validate_video_file(video_path)
    return success
