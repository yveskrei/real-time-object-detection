from fastapi import HTTPException
from storage import storage
from models import BBoxCreate
import time

class BBoxManager:
    """Handles bounding box operations with raw PTS, retention, and WebSocket broadcasting"""
    
    # Retention period in milliseconds (5 minutes)
    RETENTION_PERIOD_MS = 5 * 60 * 1000
    
    # Standard MPEG-TS time base for converting PTS to milliseconds
    STANDARD_TIME_BASE = 90000.0
    
    @staticmethod
    def _pts_to_ms(pts: int) -> int:
        """Convert raw PTS to milliseconds using standard time base"""
        return int((pts / BBoxManager.STANDARD_TIME_BASE) * 1000)
    
    @staticmethod
    def _cleanup_old_bboxes(video_id: int, current_time_ms: int):
        """Remove bboxes older than retention period based on absolute time"""
        if video_id not in storage.bboxes:
            return
        
        cutoff_time = current_time_ms - BBoxManager.RETENTION_PERIOD_MS
        
        pts_to_remove = []
        for pts, bbox_list in storage.bboxes[video_id].items():
            if bbox_list and bbox_list[0].get("absolute_timestamp_ms", 0) < cutoff_time:
                pts_to_remove.append(pts)
        
        for pts in pts_to_remove:
            del storage.bboxes[video_id][pts]
    
    @staticmethod
    async def add_bboxes(bbox_data: BBoxCreate, websocket_manager=None) -> dict:
        """Add bounding boxes with raw PTS and broadcast via WebSocket"""
        
        video_id = bbox_data.stream_id
        
        if video_id not in storage.videos:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        
        if video_id not in storage.active_streams:
            raise HTTPException(status_code=400, detail=f"Video {video_id} is not currently streaming")
        
        stream_start_time_ms = storage.active_streams[video_id]['start_time_ms']
        
        if video_id not in storage.bboxes:
            storage.bboxes[video_id] = {}
        
        added_count = 0
        current_time_ms = int(time.time() * 1000)
        
        # Group bboxes by PTS for efficient broadcasting
        pts_groups = {}
        
        for bbox in bbox_data.bboxes:
            pts = bbox.pts
            pts_ms = BBoxManager._pts_to_ms(pts)
            absolute_timestamp_ms = stream_start_time_ms + pts_ms
            
            if pts not in storage.bboxes[video_id]:
                storage.bboxes[video_id][pts] = []
            
            bbox_dict = {
                "pts": pts,
                "absolute_timestamp_ms": absolute_timestamp_ms,
                "top_left_corner": bbox.top_left_corner,
                "bottom_right_corner": bbox.bottom_right_corner,
                "class_name": bbox.class_name,
                "confidence": bbox.confidence
            }
            
            storage.bboxes[video_id][pts].append(bbox_dict)
            added_count += 1
            
            # Group for broadcasting
            if pts not in pts_groups:
                pts_groups[pts] = []
            pts_groups[pts].append(bbox_dict)
        
        # Cleanup old bboxes
        BBoxManager._cleanup_old_bboxes(video_id, current_time_ms)
        
        # Broadcast to WebSocket clients
        if websocket_manager:
            for pts, bboxes in pts_groups.items():
                await websocket_manager.broadcast_bboxes(video_id, {
                    "type": "bboxes",
                    "video_id": video_id,
                    "pts": pts,
                    "bboxes": bboxes,
                    "timestamp": current_time_ms
                })
        
        return {
            "video_id": video_id,
            "added_count": added_count,
            "remaining_pts_count": len(storage.bboxes[video_id]),
            "retention_period_ms": BBoxManager.RETENTION_PERIOD_MS,
            "stream_start_time_ms": stream_start_time_ms,
            "websocket_clients": websocket_manager.get_connection_count(video_id) if websocket_manager else 0,
            "message": f"Successfully added {added_count} bounding boxes"
        }
    
    @staticmethod
    def get_bboxes_range(video_id: int, from_pts: int, to_pts: int) -> dict:
        """Get all bboxes between two raw PTS values (inclusive)"""
        
        if video_id not in storage.videos:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        
        if video_id not in storage.bboxes:
            return {
                "video_id": video_id,
                "from_pts": from_pts,
                "to_pts": to_pts,
                "results": []
            }
        
        results = []
        for pts, bboxes_list in storage.bboxes[video_id].items():
            if from_pts <= pts <= to_pts:
                results.append({
                    "pts": pts,
                    "bboxes": bboxes_list
                })
        
        results.sort(key=lambda x: x["pts"])
        
        return {
            "video_id": video_id,
            "from_pts": from_pts,
            "to_pts": to_pts,
            "count": len(results),
            "results": results
        }
    
    @staticmethod
    def get_all_bboxes_for_video(video_id: int) -> dict:
        """Get all bounding boxes for a video (within retention period)"""
        
        if video_id not in storage.videos:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        
        pts_with_bboxes = storage.bboxes.get(video_id, {})
        
        all_pts = []
        for pts, bboxes_list in pts_with_bboxes.items():
            all_pts.append({
                "pts": pts,
                "bboxes": bboxes_list
            })
        
        all_pts.sort(key=lambda x: x["pts"])
        
        oldest_pts = min(pts_with_bboxes.keys()) if pts_with_bboxes else None
        newest_pts = max(pts_with_bboxes.keys()) if pts_with_bboxes else None
        
        return {
            "video_id": video_id,
            "total_pts_count": len(pts_with_bboxes),
            "oldest_pts": oldest_pts,
            "newest_pts": newest_pts,
            "retention_period_ms": BBoxManager.RETENTION_PERIOD_MS,
            "results": all_pts
        }
    
    @staticmethod
    def cleanup_all_old_bboxes():
        """Manually trigger cleanup for all videos"""
        current_time_ms = int(time.time() * 1000)
        cleaned_videos = 0
        total_removed = 0
        
        for video_id in list(storage.bboxes.keys()):
            if video_id not in storage.videos:
                del storage.bboxes[video_id]
                cleaned_videos += 1
                continue
            
            initial_count = len(storage.bboxes[video_id])
            BBoxManager._cleanup_old_bboxes(video_id, current_time_ms)
            removed_count = initial_count - len(storage.bboxes[video_id])
            
            if removed_count > 0:
                cleaned_videos += 1
                total_removed += removed_count
        
        return {
            "cleaned_videos": cleaned_videos,
            "total_pts_removed": total_removed,
            "retention_period_ms": BBoxManager.RETENTION_PERIOD_MS
        }