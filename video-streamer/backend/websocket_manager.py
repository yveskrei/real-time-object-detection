from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import asyncio
from storage import storage

class ConnectionManager:
    """Manages WebSocket connections for real-time bbox broadcasting"""
    
    def __init__(self):
        self.active_connections: Dict[int, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, video_id: int):
        """Register a new WebSocket connection for a video stream"""
        await websocket.accept()
        
        async with self._lock:
            if video_id not in self.active_connections:
                self.active_connections[video_id] = set()
            self.active_connections[video_id].add(websocket)
    
    async def disconnect(self, websocket: WebSocket, video_id: int):
        """Remove a WebSocket connection"""
        async with self._lock:
            if video_id in self.active_connections:
                self.active_connections[video_id].discard(websocket)
                
                if len(self.active_connections[video_id]) == 0:
                    del self.active_connections[video_id]
    
    async def broadcast_bboxes(self, video_id: int, message: dict):
        """Broadcast bbox data to all connected clients for a video"""
        if video_id not in self.active_connections:
            return
        
        stream_start_time_ms = None
        if video_id in storage.active_streams:
            stream_start_time_ms = storage.active_streams[video_id]['start_time_ms']
        
        message['stream_start_time_ms'] = stream_start_time_ms
        
        message_json = json.dumps(message)
        
        disconnected = set()
        for connection in self.active_connections[video_id].copy():
            try:
                await connection.send_text(message_json)
            except WebSocketDisconnect:
                disconnected.add(connection)
            except Exception:
                disconnected.add(connection)
        
        async with self._lock:
            for conn in disconnected:
                self.active_connections[video_id].discard(conn)
    
    def get_connection_count(self, video_id: int) -> int:
        """Get number of active connections for a video"""
        return len(self.active_connections.get(video_id, set()))
    
    async def send_stream_info(self, websocket: WebSocket, video_id: int):
        """Send stream info to client"""
        if video_id in storage.active_streams:
            stream_data = storage.active_streams[video_id]
            tcp_info = stream_data.get('tcp_info', {})
            
            await websocket.send_json({
                "type": "stream_info",
                "video_id": video_id,
                "stream_start_time_ms": stream_data['start_time_ms'],
                "tcp": tcp_info,
                "dash": {
                    "manifest_url": f"/dash/{video_id}/manifest.mpd"
                },
                "message": "Connected to stream"
            })

manager = ConnectionManager()