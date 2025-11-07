import websocket
import json
import threading
from typing import Callable, Optional
from PyQt6.QtCore import QObject, pyqtSignal

class WebSocketClient(QObject):
    """WebSocket client for receiving real-time bbox updates"""
    
    # Qt signals for thread-safe communication
    bbox_received = pyqtSignal(dict)  # Emits bbox data
    stream_info_received = pyqtSignal(dict)  # Emits stream info
    connection_status = pyqtSignal(bool, str)  # connected, message
    error_occurred = pyqtSignal(str)
    
    def __init__(self, backend_url: str, video_id: int):
        super().__init__()
        
        # Convert HTTP URL to WebSocket URL
        ws_url = backend_url.replace('http://', 'ws://').replace('https://', 'wss://')
        self.url = f"{ws_url}/ws/{video_id}"
        self.video_id = video_id
        
        self.ws: Optional[websocket.WebSocketApp] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
    
    def connect(self):
        """Start WebSocket connection in background thread"""
        if self.running:
            return
        
        self.running = True
        
        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        
        self.thread = threading.Thread(target=self._run_websocket, daemon=True)
        self.thread.start()
    
    def _run_websocket(self):
        """Run WebSocket in thread"""
        try:
            self.ws.run_forever(ping_interval=30, ping_timeout=10)
        except Exception as e:
            self.error_occurred.emit(f"WebSocket error: {str(e)}")
    
    def _on_open(self, ws):
        """WebSocket connection opened"""
        print(f"[WebSocket] Connected to video {self.video_id}")
        self.connection_status.emit(True, "Connected")
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'bboxes':
                # Real-time bbox update
                self.bbox_received.emit(data)
            
            elif msg_type == 'stream_info':
                # Initial stream information
                self.stream_info_received.emit(data)
            
            elif msg_type == 'pong':
                # Heartbeat response
                pass
            
            elif msg_type == 'error':
                self.error_occurred.emit(data.get('message', 'Unknown error'))
        
        except json.JSONDecodeError:
            self.error_occurred.emit(f"Invalid JSON received: {message}")
        except Exception as e:
            self.error_occurred.emit(f"Message handling error: {str(e)}")
    
    def _on_error(self, ws, error):
        """WebSocket error occurred"""
        error_msg = str(error)
        print(f"[WebSocket] Error: {error_msg}")
        self.error_occurred.emit(error_msg)
    
    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket connection closed"""
        print(f"[WebSocket] Disconnected from video {self.video_id}")
        self.connection_status.emit(False, f"Disconnected: {close_msg or 'Unknown'}")
    
    def disconnect(self):
        """Close WebSocket connection"""
        self.running = False
        if self.ws:
            self.ws.close()
        if self.thread:
            self.thread.join(timeout=2)
    
    def send_ping(self):
        """Send ping to keep connection alive"""
        if self.ws and self.running:
            try:
                self.ws.send(json.dumps({"type": "ping"}))
            except Exception as e:
                print(f"[WebSocket] Ping failed: {e}")