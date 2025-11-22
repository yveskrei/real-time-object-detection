from fastapi import APIRouter, Query
from models import BBoxCreate
from bbox_manager import BBoxManager
from websocket_manager import manager as ws_manager

router = APIRouter(prefix="/bboxes", tags=["bboxes"])

@router.post("/")
async def add_bboxes(bbox_data: BBoxCreate):
    """Add bounding boxes for a specific PTS and broadcast via WebSocket"""
    return await BBoxManager.add_bboxes(bbox_data, websocket_manager=ws_manager)

@router.get("/{video_id}")
def get_all_bboxes(
    video_id: int,
    limit: int = Query(None, description="Limit the number of most recent bboxes to return")
):
    """Get all bounding boxes for a video (within retention period), optionally limited to the last X bboxes"""
    return BBoxManager.get_all_bboxes_for_video(video_id, limit=limit)

@router.post("/cleanup")
def cleanup_old_bboxes():
    """Manually trigger cleanup of old bboxes across all videos"""
    return BBoxManager.cleanup_all_old_bboxes()