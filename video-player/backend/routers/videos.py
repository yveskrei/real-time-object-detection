from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional

# Custom modules
from models import VideoInfo
from video_manager import VideoManager

router = APIRouter(prefix="/videos", tags=["videos"])

@router.post("/upload", response_model=VideoInfo)
async def upload_video(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None)
):
    """Upload a video file"""
    return await VideoManager.create_video(file, name or file.filename)

@router.get("/{video_id}", response_model=VideoInfo)
def get_video(video_id: int):
    """Get video by ID"""
    return VideoManager.get_video(video_id)

@router.get("/", response_model=list[VideoInfo])
def list_videos():
    """List all videos"""
    return VideoManager.list_videos()

@router.delete("/{video_id}")
def delete_video(video_id: int):
    """Delete a video"""
    return VideoManager.delete_video(video_id)