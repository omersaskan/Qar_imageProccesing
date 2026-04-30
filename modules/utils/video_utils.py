import subprocess
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger("video_utils")

def get_video_metadata(video_path: Path, ffprobe_path: str = "ffprobe"):
    """
    Uses ffprobe to extract video metadata including codec and pixel format.
    """
    cmd = [
        ffprobe_path,
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
        if not video_stream:
            return None
            
        return {
            "codec_name": video_stream.get("codec_name"),
            "pix_fmt": video_stream.get("pix_fmt"),
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": eval(video_stream.get("r_frame_rate", "0/1")), # Convert "30/1" or "24000/1001"
            "duration": float(data.get("format", {}).get("duration", 0)),
            "frame_count": int(video_stream.get("nb_frames") or 0)
        }
    except Exception as e:
        logger.error(f"ffprobe failed for {video_path}: {e}")
        return None

def normalize_video(input_path: Path, output_path: Path, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
    """
    Normalizes any supported video format to a standard MP4 using ffmpeg.
    Skips transcoding if already H.264 yuv420p MP4.
    """
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
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg normalization failed: {e.stderr}")
        raise RuntimeError(f"FFmpeg failed: {e.stderr}")
    except FileNotFoundError:
        logger.error("FFmpeg binary not found.")
        raise RuntimeError("FFmpeg not found on the system.")

def validate_video_file(video_path: Path, min_fps=0, min_duration=0, max_duration=9999):
    """
    Comprehensive validation of the final raw_video.mp4.
    """
    import cv2
    if not video_path.exists():
        return False, "File missing"
        
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return False, "cv2 cannot open video"
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0:
            return False, f"Invalid FPS: {fps}"
        if frame_count <= 0:
            return False, f"Invalid frame count: {frame_count}"
            
        ret, _ = cap.read()
        if not ret:
            return False, "Cannot read first frame"
            
        duration = frame_count / fps
        if fps < min_fps:
            return False, f"FPS too low: {fps:.1f} < {min_fps}"
        if duration < min_duration:
            return False, f"Duration too short: {duration:.1f}s < {min_duration}s"
        if duration > max_duration:
            return False, f"Duration too long: {duration:.1f}s > {max_duration}s"
            
        return True, None
    finally:
        cap.release()

def check_video_readable(video_path: Path):
    """Legacy helper."""
    success, _ = validate_video_file(video_path)
    return success
