from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pathlib import Path

from routers import videos, streams, bboxes
from storage import storage
from stream_manager import StreamManager
from websocket_manager import manager as ws_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cleanup on shutdown"""
    yield
    for video_id in list(storage.active_streams.keys()):
        try:
            StreamManager.stop_stream(video_id)
        except:
            pass

app = FastAPI(
    title="Video Stream Management API",
    description="API for managing video streams with DASH and real-time bbox WebSocket support",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(videos.router)
app.include_router(streams.router)
app.include_router(bboxes.router)

# Custom DASH file serving with proper MIME types
@app.get("/dash/{video_id}/{filename:path}")
async def serve_dash_file(video_id: int, filename: str):
    """Serve DASH files with proper MIME types"""
    file_path = Path("dash_streams") / str(video_id) / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Set proper MIME type based on extension
    mime_types = {
        ".mpd": "application/dash+xml",
        ".m4s": "video/iso.segment",
        ".mp4": "video/mp4",
    }
    
    suffix = file_path.suffix.lower()
    media_type = mime_types.get(suffix, "application/octet-stream")
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Expose-Headers": "*"
        }
    )

@app.get("/")
def root():
    return {
        "message": "Video Stream Management API with DASH and WebSocket",
        "docs": "/docs",
        "total_videos": len(storage.videos),
        "active_streams": len(storage.active_streams),
        "websocket_endpoint": "/ws/{video_id}",
        "dash_endpoint": "/dash/{video_id}/manifest.mpd"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.websocket("/ws/{video_id}")
async def websocket_endpoint(websocket: WebSocket, video_id: int):
    """WebSocket endpoint for real-time bbox updates"""
    await ws_manager.connect(websocket, video_id)
    
    try:
        await ws_manager.send_stream_info(websocket, video_id)
        
        while True:
            try:
                data = await websocket.receive_text()
                await websocket.send_json({
                    "type": "pong",
                    "message": "alive"
                })
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"[WebSocket] Error in receive loop: {e}")
                break
    finally:
        await ws_manager.disconnect(websocket, video_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8702)