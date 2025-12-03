from fastapi import HTTPException, UploadFile
from datetime import datetime
from pathlib import Path
from storage import storage
from models import VideoInfo
import subprocess
import json
import logging
import os

# Variables
logger = logging.getLogger(__name__)

class VideoManager:
    """Handles video file operations and metadata"""

    @staticmethod
    def _get_video_properties(file_path: str) -> dict:
        """
        Use ffprobe to get video width, height, and fps.
        Raises exception if video properties cannot be determined.
        """
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate",
            "-of", "json",
            file_path
        ]
        
        try:
            print("I'm here!")

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            if not data.get("streams") or len(data["streams"]) == 0:
                logger.error(f"ffprobe found no video streams for {file_path}")
                raise ValueError("No video streams found in file")
                
            stream_data = data["streams"][0]
            
            # Parse avg_frame_rate (e.g., "30000/1001" or "30/1")
            fps_str = stream_data.get("avg_frame_rate", "0/1")
            if not fps_str or fps_str == "0/0":
                raise ValueError("Invalid or missing frame rate")
            
            num, den = map(float, fps_str.split('/'))
            if den == 0:
                raise ValueError("Invalid frame rate (division by zero)")
            fps = num / den
            
            width = stream_data.get("width")
            height = stream_data.get("height")
            
            if not width or width <= 0:
                raise ValueError(f"Invalid width: {width}")
            if not height or height <= 0:
                raise ValueError(f"Invalid height: {height}")
            if fps <= 0:
                raise ValueError(f"Invalid fps: {fps}")
            
            logger.info(f"Video properties for {file_path}: {width}x{height} @ {fps:.2f} fps")
            
            return {
                "width": int(width),
                "height": int(height),
                "fps": float(fps)
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"ffprobe failed for {file_path}: {e.stderr}")
            raise ValueError(f"Failed to analyze video file: {e.stderr}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse ffprobe output for {file_path}: {e}")
            raise ValueError("Failed to parse video file metadata")
        except Exception as e:
            logger.error(f"Failed to get video properties for {file_path}: {e}")
            raise ValueError(f"Failed to get video properties: {str(e)}")
    
    @staticmethod
    async def create_video( file: UploadFile, name: str) -> VideoInfo:
        """Upload and register a new video"""
        
        if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(
                status_code=400, 
                detail="Only video files allowed (.mp4, .avi, .mov, .mkv)"
            )
        
        video_id = storage.get_next_video_id()
        video_name = name or file.filename
        file_path = os.getcwd() / storage.video_storage_path / f"{video_id}.mp4"
        
        # Save uploaded file
        try:
            print(file_path)
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

        # Get video properties (strict validation, no fallbacks)
        try:
            properties = VideoManager._get_video_properties(str(file_path))
        except ValueError as e:
            # Clean up uploaded file if property extraction fails
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid video file: {str(e)}"
            )
        
        # Store video metadata
        video_data = {
            "id": video_id,
            "name": video_name,
            "file_path": str(file_path),
            "created_at": datetime.now().isoformat(),
            "is_streaming": False,
            "width": properties["width"],
            "height": properties["height"],
            "fps": properties["fps"]
        }
        
        storage.videos[video_id] = video_data
        storage.bboxes[video_id] = {}
        
        logger.info(f"Video {video_id} created: {video_name} ({properties['width']}x{properties['height']} @ {properties['fps']:.2f} fps)")
        
        return VideoInfo(**video_data)
    
    @staticmethod
    def get_video(video_id: int) -> VideoInfo:
        """Get video by ID"""
        if video_id not in storage.videos:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        return VideoInfo(**storage.videos[video_id])
    
    @staticmethod
    def list_videos() -> list[VideoInfo]:
        """List all videos"""
        # Create a snapshot to avoid race conditions during iteration
        # when other threads modify storage.videos
        videos_snapshot = list(storage.videos.values())
        return [VideoInfo(**v) for v in videos_snapshot]
    
    @staticmethod
    def delete_video(video_id: int) -> dict:
        """Delete a video (only if not streaming)"""
        if video_id not in storage.videos:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        
        # Check if video is currently streaming
        if storage.videos[video_id]["is_streaming"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot delete video {video_id}: stream is currently active. Stop the stream first."
            )
        
        # Check if there's an active stream process (double check)
        if video_id in storage.active_streams:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot delete video {video_id}: stream process is still running. Stop the stream first."
            )
        
        # Delete file
        video_data = storage.videos[video_id]
        file_path = Path(video_data["file_path"])
        if file_path.exists():
            file_path.unlink()
        
        # Remove from storage
        del storage.videos[video_id]
        if video_id in storage.bboxes:
            del storage.bboxes[video_id]
        
        logger.info(f"Video {video_id} deleted successfully")
        
        return {"message": f"Video {video_id} deleted successfully"}