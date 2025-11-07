from fastapi import APIRouter

# Custom modules
from models import StreamConfig
from stream_manager import StreamManager

router = APIRouter(prefix="/streams", tags=["streams"])

@router.post("/start")
def start_stream(config: StreamConfig):
    """Start streaming a video"""
    return StreamManager.start_stream(
        config.video_id,
        config.output_format,
        config.resolution
    )

@router.post("/stop/{video_id}")
def stop_stream(video_id: int):
    """Stop streaming a video"""
    return StreamManager.stop_stream(video_id)

@router.get("/status/{video_id}")
def get_stream_status(video_id: int):
    """Get stream status"""
    return StreamManager.get_stream_status(video_id)