from fastapi import APIRouter, Query
from models import BBoxCreate
from bbox_manager import BBoxManager

router = APIRouter(prefix="/bboxes", tags=["bboxes"])

@router.post("/")
def add_bboxes(bbox_data: BBoxCreate):
    """Add bounding boxes for a specific PTS (auto-cleanup old data)"""
    return BBoxManager.add_bboxes(bbox_data)

@router.get("/{video_id}/range")
def get_bboxes_range(
    video_id: int,
    from_pts: int = Query(..., description="Start PTS in milliseconds"),
    to_pts: int = Query(..., description="End PTS in milliseconds")
):
    """Get all bboxes between two PTS values (inclusive)"""
    return BBoxManager.get_bboxes_range(video_id, from_pts, to_pts)

@router.get("/{video_id}")
def get_all_bboxes(video_id: int):
    """Get all bounding boxes for a video (within retention period)"""
    return BBoxManager.get_all_bboxes_for_video(video_id)

@router.post("/cleanup")
def cleanup_old_bboxes():
    """Manually trigger cleanup of old bboxes across all videos"""
    return BBoxManager.cleanup_all_old_bboxes()