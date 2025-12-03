from pydantic import BaseModel, Field, Extra
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

class BoundingBox(BaseModel):
    TopLeftPixelNumber: int
    BottomRightPixelNumber: int

class BBoxClientAI(BaseModel):
    StartTime: int
    BoundingBox: BoundingBox
    Class: str
    Confidence: float

    class Config:
        extra = Extra.ignore

class BBoxCreate(BaseModel):
    SourceId: int
    System: str
    Data: List[BBoxClientAI]

    class Config:
        extra = Extra.ignore

class StreamConfig(BaseModel):
    video_id: int