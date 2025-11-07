from fastapi import HTTPException, UploadFile
from datetime import datetime
from pathlib import Path
from storage import storage
from models import VideoInfo

class VideoManager:
    """Handles video file operations and metadata"""
    
    @staticmethod
    async def create_video(file: UploadFile, name: str) -> VideoInfo:
        """Upload and register a new video"""
        
        if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="Only video files allowed (.mp4, .avi, .mov, .mkv)")
        
        video_id = storage.get_next_video_id()
        video_name = name or file.filename
        file_path = storage.video_storage_path / f"{video_id}.mp4"
        
        # Save uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Store video metadata
        video_data = {
            "id": video_id,
            "name": video_name,
            "file_path": str(file_path),
            "created_at": datetime.now().isoformat(),
            "is_streaming": False
        }
        
        storage.videos[video_id] = video_data
        storage.bboxes[video_id] = {}
        
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
        return [VideoInfo(**v) for v in storage.videos.values()]
    
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
        
        return {"message": f"Video {video_id} deleted successfully"}