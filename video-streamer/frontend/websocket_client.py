import websocket
import json
import threading
from typing import Optional
from PyQt6.QtCore import QObject, pyqtSignal

class WebSocketClient(QObject):
    """WebSocket client for receiving real-time bbox updates"""
    
    bbox_received = pyqtSignal(dict)
    stream_info_received = pyqtSignal(dict)
    connection_status = pyqtSignal(bool, str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, backend_url: str, video_id: int):
        super().__init__()
        
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
            # Keep-alive pings managed by websocket-client
            self.ws.run_forever(ping_interval=30, ping_timeout=10)
        except Exception as e:
            if self.running:
                self.error_occurred.emit(f"WebSocket error: {str(e)}")
    
    def _on_open(self, ws):
        """WebSocket connection opened"""
        self.connection_status.emit(True, "Connected")
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'bboxes':
                self.bbox_received.emit(data)
            elif msg_type == 'stream_info':
                self.stream_info_received.emit(data)
            elif msg_type == 'pong':
                pass
            elif msg_type == 'error':
                self.error_occurred.emit(data.get('message', 'Unknown error'))
        
        except json.JSONDecodeError:
            self.error_occurred.emit(f"Invalid JSON received: {message}")
        except Exception as e:
            self.error_occurred.emit(f"Message handling error: {str(e)}")
    
    def _on_error(self, ws, error):
        """WebSocket error occurred"""
        # Don't emit simple "Connection reset by peer" if we initiated the close
        if self.running and "reset by peer" not in str(error):
            self.error_occurred.emit(str(error))
    
    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket connection closed"""
        self.running = False
        self.connection_status.emit(False, f"Disconnected: {close_msg or 'Unknown'}")
    
    def disconnect(self):
        """Close WebSocket connection non-blockingly"""
        self.running = False
        if self.ws:
            # Closing the socket will terminate the run_forever loop
            self.ws.close()
        
        # DO NOT call self.thread.join() here, as it blocks the GUI thread
        # The thread is daemonic and will exit
    
    def send_ping(self):
        """Send ping to keep connection alive"""
        if self.ws and self.running:
            try:
                self.ws.send(json.dumps({"type": "ping"}))
            except Exception:
                pass