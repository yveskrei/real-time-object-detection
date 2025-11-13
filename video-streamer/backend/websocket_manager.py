from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import asyncio
from storage import storage

class ConnectionManager:
    """Manages WebSocket connections for real-time bbox broadcasting"""
    
    def __init__(self):
        # {video_id: {websocket1, websocket2, ...}}
        self.active_connections: Dict[int, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, video_id: int):
        """Register a new WebSocket connection for a video stream"""
        await websocket.accept()
        
        async with self._lock:
            if video_id not in self.active_connections:
                self.active_connections[video_id] = set()
            self.active_connections[video_id].add(websocket)
        
        print(f"[WebSocket] Client connected to video {video_id}. Total: {len(self.active_connections[video_id])}")
    
    async def disconnect(self, websocket: WebSocket, video_id: int):
        """Remove a WebSocket connection"""
        async with self._lock:
            if video_id in self.active_connections:
                self.active_connections[video_id].discard(websocket)
                
                # Clean up empty sets
                if len(self.active_connections[video_id]) == 0:
                    del self.active_connections[video_id]
        
        print(f"[WebSocket] Client disconnected from video {video_id}")
    
    async def broadcast_bboxes(self, video_id: int, message: dict):
        """Broadcast bbox data to all connected clients for a video"""
        if video_id not in self.active_connections:
            return
        
        # Get stream start time for clients to compute absolute timestamps
        stream_start_time_ms = None
        if video_id in storage.active_streams:
            stream_start_time_ms = storage.active_streams[video_id]['start_time_ms']
        
        # Add stream context to message
        message['stream_start_time_ms'] = stream_start_time_ms
        
        message_json = json.dumps(message)
        
        # Broadcast to all connected clients (with error handling)
        disconnected = set()
        for connection in self.active_connections[video_id].copy():
            try:
                await connection.send_text(message_json)
            except WebSocketDisconnect:
                disconnected.add(connection)
            except Exception as e:
                print(f"[WebSocket] Error sending to client: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected clients
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
            await websocket.send_json({
                "type": "stream_info",
                "video_id": video_id,
                "port": stream_data['srt_port'],
                "stream_start_time_ms": stream_data['start_time_ms'],
                "message": "Connected to stream"
            })

# Global connection manager
manager = ConnectionManager()