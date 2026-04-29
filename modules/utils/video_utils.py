import subprocess
import os
from pathlib import Path
import logging

logger = logging.getLogger("video_utils")

def normalize_video(input_path: Path, output_path: Path, ffmpeg_path: str = "ffmpeg"):
    """
    Normalizes any supported video format to a standard MP4 using ffmpeg.
    Standard: H.264 video, AAC audio (or no audio), yuv420p.
    """
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
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg normalization failed: {e.stderr}")
        raise RuntimeError(f"FFmpeg failed: {e.stderr}")
    except FileNotFoundError:
        logger.error("FFmpeg binary not found.")
        raise RuntimeError("FFmpeg not found on the system.")

def check_video_readable(video_path: Path):
    """Checks if cv2 can open and read at least one frame."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        return ret
    finally:
        cap.release()
