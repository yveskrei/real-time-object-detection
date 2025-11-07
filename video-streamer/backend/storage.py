from typing import Dict, List
import subprocess
from pathlib import Path

class Storage:
    """In-memory storage for videos, streams, and bounding boxes"""
    
    def __init__(self):
        self.videos: Dict[int, dict] = {}
        self.active_streams: Dict[int, subprocess.Popen] = {}
        self.bboxes: Dict[int, Dict[int, List[dict]]] = {}  # {video_id: {pts: [bbox_data]}}
        self.next_video_id: int = 1
        self.video_storage_path = Path("./videos")
        self.video_storage_path.mkdir(exist_ok=True)
    
    def get_next_video_id(self) -> int:
        video_id = self.next_video_id
        self.next_video_id += 1
        return video_id
    
    def reset(self):
        """Reset all storage (useful for testing)"""
        self.videos.clear()
        self.active_streams.clear()
        self.bboxes.clear()
        self.next_video_id = 1

# Global storage instance
storage = Storage()