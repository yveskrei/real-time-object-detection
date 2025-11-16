from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

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

app.mount("/dash", StaticFiles(directory="dash_streams"), name="dash")

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