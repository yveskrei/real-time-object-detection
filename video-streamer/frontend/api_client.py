import requests
from typing import List

class APIClient:
    """Handles all API communication with the backend"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    def upload_video(self, file_path: str, name: str) -> dict:
        """Upload a video file"""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'name': name}
            response = requests.post(f"{self.base_url}/videos/upload", files=files, data=data)
            response.raise_for_status()
            return response.json()
    
    def list_videos(self) -> List[dict]:
        """Get all videos"""
        response = requests.get(f"{self.base_url}/videos/")
        response.raise_for_status()
        return response.json()
    
    def get_video(self, video_id: int) -> dict:
        """Get video by ID"""
        response = requests.get(f"{self.base_url}/videos/{video_id}")
        response.raise_for_status()
        return response.json()
    
    def delete_video(self, video_id: int) -> dict:
        """Delete a video"""
        response = requests.delete(f"{self.base_url}/videos/{video_id}")
        response.raise_for_status()
        return response.json()
    
    def start_stream(self, video_id: int) -> dict:
        """Start streaming a video"""
        data = {"video_id": video_id}
        response = requests.post(f"{self.base_url}/streams/start", json=data)
        response.raise_for_status()
        return response.json()
    
    def stop_stream(self, video_id: int) -> dict:
        """Stop streaming a video"""
        response = requests.post(f"{self.base_url}/streams/stop/{video_id}")
        response.raise_for_status()
        return response.json()
    
    def get_stream_status(self, video_id: int) -> dict:
        """Get stream status"""
        response = requests.get(f"{self.base_url}/streams/status/{video_id}")
        response.raise_for_status()
        return response.json()
    
    def add_bboxes(self, video_id: int, bboxes: List[dict]) -> dict:
        """Add bounding boxes with raw PTS timestamps"""
        data = {
            "stream_id": video_id,
            "bboxes": bboxes
        }
        response = requests.post(f"{self.base_url}/bboxes/", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_bboxes_range(self, video_id: int, from_pts: int, to_pts: int) -> dict:
        """Get all bboxes between two PTS values"""
        response = requests.get(
            f"{self.base_url}/bboxes/{video_id}/range",
            params={"from_pts": from_pts, "to_pts": to_pts}
        )
        response.raise_for_status()
        return response.json()
    
    def get_all_bboxes(self, video_id: int) -> dict:
        """Get all bounding boxes for a video"""
        response = requests.get(f"{self.base_url}/bboxes/{video_id}")
        response.raise_for_status()
        return response.json()