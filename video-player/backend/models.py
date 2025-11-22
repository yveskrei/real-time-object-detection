from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class VideoCreate(BaseModel):
    name: str

class VideoInfo(BaseModel):
    id: int
    name: str
    file_path: str
    created_at: str
    is_streaming: bool
    width: int
    height: int
    fps: float

class BBoxData(BaseModel):
    pts: int = Field(..., description="Presentation timestamp in milliseconds from video start")
    top_left_corner: int = Field(..., description="Top left corner of bbox - pixel index number")
    bottom_right_corner: int = Field(..., description="Bottom right corner of bbox - pixel index number")
    class_name: str = Field(..., description="Object class name")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")

class BBoxCreate(BaseModel):
    stream_id: int
    bboxes: List[BBoxData]

class StreamConfig(BaseModel):
    video_id: int