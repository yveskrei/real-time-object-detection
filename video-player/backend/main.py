from fastapi import FastAPI
from contextlib import asynccontextmanager

# Custom modules
from routers import videos, streams, bboxes
from storage import storage
from stream_manager import StreamManager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cleanup on shutdown"""
    yield
    # Stop all active streams on shutdown
    for video_id in list(storage.active_streams.keys()):
        try:
            StreamManager.stop_stream(video_id)
        except:
            pass

app = FastAPI(
    title="Video Stream Management API",
    description="Admin panel for managing video streams with MPEG-TS and bounding box support",
    version="1.0.0",
    lifespan=lifespan
)

# Include routers
app.include_router(videos.router)
app.include_router(streams.router)
app.include_router(bboxes.router)

@app.get("/")
def root():
    return {
        "message": "Video Stream Management API",
        "docs": "/docs",
        "total_videos": len(storage.videos),
        "active_streams": len(storage.active_streams)
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8702)