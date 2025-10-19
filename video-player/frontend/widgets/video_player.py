"""
Video player using WebSocket for real-time bbox synchronization
"""

import av
import time
import numpy as np
from collections import deque
from pathlib import Path
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QFileDialog, QMessageBox
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt, QBuffer, QIODevice
from PyQt6.QtGui import QImage, QPixmap, QPainter
from widgets.bbox_overlay import BBoxOverlay
from websocket_client import WebSocketClient


class ReplaySaveThread(QThread):
    """Background thread for saving replay"""
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, replay_buffer, file_path, resolution, fps):
        super().__init__()
        self.replay_buffer = list(replay_buffer)
        self.file_path = file_path
        self.resolution = resolution
        self.fps = fps
    
    def run(self):
        try:
            width = self.resolution.width()
            height = self.resolution.height()
            total_frames = len(self.replay_buffer)
            
            output = av.open(self.file_path, mode='w')
            stream = output.add_stream('h264', rate=self.fps)
            stream.width = width
            stream.height = height
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': '23', 'preset': 'medium'}
            
            for i, jpeg_data in enumerate(self.replay_buffer):
                qimage = QImage.fromData(jpeg_data, "JPEG")
                qimage = qimage.convertToFormat(QImage.Format.Format_RGB888)
                ptr = qimage.bits()
                ptr.setsize(qimage.sizeInBytes())
                arr = np.array(ptr).reshape(height, width, 3)
                
                frame = av.VideoFrame.from_ndarray(arr, format='rgb24')
                
                for packet in stream.encode(frame):
                    output.mux(packet)
                
                self.progress.emit(i + 1, total_frames)
            
            for packet in stream.encode():
                output.mux(packet)
            
            output.close()
            self.finished.emit(True, f"Replay saved successfully to:\n{self.file_path}")
            
        except Exception as e:
            self.finished.emit(False, f"Failed to save replay:\n{str(e)}")


class VideoStreamThread(QThread):
    """Thread for reading video stream frames"""
    frame_ready = pyqtSignal(object, int, int, float)  # frame, frame_number, pts_raw, time_base
    stream_info = pyqtSignal(float)  # fps
    error = pyqtSignal(str)
    
    def __init__(self, stream_url: str):
        super().__init__()
        self.stream_url = stream_url
        self.running = False
        self.frame_count = 0
    
    def run(self):
        """Read frames from stream"""
        self.running = True
        container = None
        
        while self.running:
            try:
                print(f"[VideoStreamThread] Connecting to {self.stream_url}...")
                
                stream_url = self.stream_url
                options = {}
                
                if stream_url.startswith('udp://'):
                    import re
                    match = re.search(r'udp://(\d+\.\d+\.\d+\.\d+):(\d+)', stream_url)
                    if match:
                        ip = match.group(1)
                        octets = [int(x) for x in ip.split('.')]
                        is_multicast = (224 <= octets[0] <= 239)
                        
                        if is_multicast or ip == '127.0.0.1' or ip == '0.0.0.0':
                            stream_url = stream_url.replace('udp://', 'udp://@')
                    
                    options = {
                        'rtbufsize': '100M',
                        'fifo_size': '1000000',
                        'overrun_nonfatal': '1',
                        'buffer_size': '65536',
                    }
                
                options.update({
                    'fflags': 'nobuffer',
                    'flags': 'low_delay',
                    'max_delay': '500000',
                })
                
                container = av.open(stream_url, options=options, timeout=10.0)
                
                if len(container.streams.video) == 0:
                    self.error.emit("No video stream found")
                    if container:
                        container.close()
                        container = None
                    time.sleep(2)
                    continue
                
                video_stream = container.streams.video[0]
                
                # Get FPS
                stream_fps = None
                if video_stream.average_rate:
                    stream_fps = float(video_stream.average_rate)
                elif video_stream.guessed_rate:
                    stream_fps = float(video_stream.guessed_rate)
                elif video_stream.codec_context.framerate:
                    stream_fps = float(video_stream.codec_context.framerate)
                
                if not stream_fps or stream_fps <= 0:
                    stream_fps = 30.0
                
                self.stream_info.emit(stream_fps)
                time_base = float(video_stream.time_base)
                
                frame_received = False
                waiting_for_keyframe = True
                
                for packet in container.demux(video_stream):
                    if not self.running:
                        break
                    
                    if waiting_for_keyframe and not packet.is_keyframe:
                        continue
                    
                    try:
                        for frame in packet.decode():
                            if not self.running:
                                break
                            
                            if waiting_for_keyframe:
                                waiting_for_keyframe = False
                                print(f"[VideoStreamThread] Found keyframe, starting decode...")
                            
                            if not frame_received:
                                print(f"[VideoStreamThread] First frame received!")
                                frame_received = True
                            
                            self.frame_count += 1
                            pts_raw = frame.pts
                            
                            if pts_raw is None:
                                pts_raw = int((self.frame_count / 30.0) / time_base)
                            
                            img = frame.to_ndarray(format='rgb24')
                            self.frame_ready.emit(img, self.frame_count, pts_raw, time_base)
                    
                    except av.AVError as decode_error:
                        if not frame_received:
                            waiting_for_keyframe = True
                        else:
                            raise
                
                if container:
                    container.close()
                    container = None
                
            except Exception as e:
                error_msg = f"Stream error: {str(e)}"
                self.error.emit(error_msg)
                
                if container:
                    try:
                        container.close()
                    except:
                        pass
                    container = None
                
                time.sleep(2)
        
        if container:
            try:
                container.close()
            except:
                pass
    
    def stop(self):
        """Stop reading frames"""
        self.running = False
        self.wait()


class VideoPlayerWidget(QWidget):
    """Video player with WebSocket-based bbox synchronization"""
    
    closed = pyqtSignal(int)
    
    def __init__(self, video_id: int, stream_url: str, stream_start_time_ms: int, 
                 backend_url: str, replay_duration_seconds: float = 30.0, 
                 buffer_delay_ms: int = 200, parent=None):
        super().__init__(parent)
        self.video_id = video_id
        self.stream_url = stream_url
        self.stream_start_time_ms = stream_start_time_ms
        self.backend_url = backend_url
        
        # Configurable settings
        self.replay_duration_seconds = replay_duration_seconds
        self.buffer_delay_ms = buffer_delay_ms
        
        # Frame buffer for delayed display
        self.frame_buffer = deque(maxlen=200)
        
        # WebSocket-based bbox cache: {raw_pts: [bboxes]}
        self.bbox_cache = {}
        self.bbox_cache_max_age_seconds = 5.0  # Keep bboxes for 5 seconds
        
        # Current state
        self.current_pts_raw = 0
        self.time_base = 1/90000.0
        
        # FPS
        self.stream_fps = 30.0
        self.fps_locked = False
        self.current_display_fps = 0.0
        self.fps_frame_count = 0
        self.fps_last_update = time.time()
        self.display_timer = None
        
        # Frame dimensions
        self.frame_width = 0
        self.frame_height = 0
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=self._calculate_replay_buffer_size())
        self.replay_resolution = None
        self.save_thread = None
        
        # WebSocket client
        self.ws_client = None
        self.ws_connected = False
        
        self.setup_ui()
        self.start_stream()
        self.connect_websocket()
    
    def _calculate_replay_buffer_size(self) -> int:
        """Calculate replay buffer size based on FPS"""
        return max(int(self.stream_fps * self.replay_duration_seconds), 30)
    
    def _update_replay_buffer_size(self):
        """Update replay buffer size"""
        if not self.fps_locked:
            return
        
        new_size = self._calculate_replay_buffer_size()
        if new_size != self.replay_buffer.maxlen:
            old_buffer = list(self.replay_buffer)
            self.replay_buffer = deque(
                old_buffer[-new_size:] if len(old_buffer) > new_size else old_buffer,
                maxlen=new_size
            )
            print(f"[Replay] Buffer size: {new_size} frames ({self.replay_duration_seconds}s @ {self.stream_fps:.1f} FPS)")
    
    def setup_ui(self):
        """Setup UI components"""
        layout = QVBoxLayout(self)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setScaledContents(True)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label)
        
        # BBox overlay
        self.bbox_overlay = BBoxOverlay(self.video_label)
        self.bbox_overlay.setGeometry(self.video_label.geometry())
        
        # Controls
        controls = QHBoxLayout()
        
        self.bbox_toggle = QPushButton("Hide BBoxes")
        self.bbox_toggle.setCheckable(True)
        self.bbox_toggle.clicked.connect(self.toggle_bboxes)
        controls.addWidget(self.bbox_toggle)
        
        # WebSocket status indicator
        self.ws_status = QLabel("🔴 WS: Disconnected")
        self.ws_status.setStyleSheet("color: red; font-weight: bold;")
        controls.addWidget(self.ws_status)
        
        self.info_label = QLabel(f"Video ID: {self.video_id} | FPS: 0.0 | PTS: 0 | Delay: 0 | Cache: 0 |")
        controls.addWidget(self.info_label)
        
        controls.addStretch()
        
        # Save replay button
        self.save_replay_btn = QPushButton("💾 Save Last 0.0s")
        self.save_replay_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_replay_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        self.save_replay_btn.clicked.connect(self.save_replay)
        controls.addWidget(self.save_replay_btn)
        
        # Stop button
        self.stop_button = QPushButton("⏹ Stop Watching")
        self.stop_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.stop_button.clicked.connect(self.stop_watching)
        controls.addWidget(self.stop_button)
        
        layout.addLayout(controls)
    
    def connect_websocket(self):
        """Connect to WebSocket for real-time bbox updates"""
        self.ws_client = WebSocketClient(self.backend_url, self.video_id)
        
        # Connect signals
        self.ws_client.bbox_received.connect(self.handle_bbox_update)
        self.ws_client.stream_info_received.connect(self.handle_stream_info)
        self.ws_client.connection_status.connect(self.handle_ws_status)
        self.ws_client.error_occurred.connect(self.handle_ws_error)
        
        # Start connection
        self.ws_client.connect()
        
        # Setup ping timer to keep connection alive
        self.ping_timer = QTimer()
        self.ping_timer.timeout.connect(self.send_ws_ping)
        self.ping_timer.start(30000)  # Ping every 30 seconds
    
    def handle_bbox_update(self, data: dict):
        """Handle incoming bbox data from WebSocket"""
        pts = data.get('pts')
        bboxes = data.get('bboxes', [])
        
        if pts is not None:
            # Store in cache
            self.bbox_cache[pts] = bboxes
            
            # Clean old cache entries
            self._clean_bbox_cache()
    
    def handle_stream_info(self, data: dict):
        """Handle stream info from WebSocket"""
        print(f"[WebSocket] Received stream info: {data}")
    
    def handle_ws_status(self, connected: bool, message: str):
        """Handle WebSocket connection status"""
        self.ws_connected = connected
        if connected:
            self.ws_status.setText("🟢 WS: Connected")
            self.ws_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.ws_status.setText("🔴 WS: Disconnected")
            self.ws_status.setStyleSheet("color: red; font-weight: bold;")
    
    def handle_ws_error(self, error: str):
        """Handle WebSocket errors"""
        print(f"[WebSocket] Error: {error}")
    
    def send_ws_ping(self):
        """Send ping to keep WebSocket alive"""
        if self.ws_client:
            self.ws_client.send_ping()
    
    def _clean_bbox_cache(self):
        """Remove old bboxes from cache"""
        if not self.current_pts_raw:
            return
        
        cutoff_pts = self.current_pts_raw - int((self.bbox_cache_max_age_seconds / self.time_base))
        old_size = len(self.bbox_cache)
        self.bbox_cache = {k: v for k, v in self.bbox_cache.items() if k > cutoff_pts}
        
        removed = old_size - len(self.bbox_cache)
        if removed > 50:
            print(f"[BBox Cache] Cleaned {removed} entries, kept {len(self.bbox_cache)}")
    
    def start_stream(self):
        """Start video stream"""
        self.stream_thread = VideoStreamThread(self.stream_url)
        self.stream_thread.frame_ready.connect(self.buffer_frame)
        self.stream_thread.stream_info.connect(self.lock_stream_fps)
        self.stream_thread.error.connect(self.handle_stream_error)
        self.stream_thread.start()
    
    def lock_stream_fps(self, fps: float):
        """Lock FPS from stream metadata"""
        if self.fps_locked:
            return
        
        self.stream_fps = fps
        self.fps_locked = True
        print(f"[VideoPlayer] FPS locked to {self.stream_fps:.2f}")
        
        self._update_replay_buffer_size()
        
        display_interval_ms = int(1000.0 / self.stream_fps)
        self.display_timer = QTimer()
        self.display_timer.timeout.connect(self.display_buffered_frame)
        self.display_timer.start(display_interval_ms)
    
    def buffer_frame(self, frame, frame_number, pts_raw, time_base):
        """Add frame to buffer"""
        arrival_time = time.time() * 1000
        self.time_base = time_base
        
        self.frame_buffer.append({
            'frame': frame,
            'frame_number': frame_number,
            'pts_raw': pts_raw,
            'arrival_time': arrival_time
        })
    
    def handle_stream_error(self, error_msg):
        """Handle stream errors"""
        print(f"[VideoPlayer] Stream error: {error_msg}")
        self.info_label.setText(f"Video ID: {self.video_id} | Error: {error_msg}")
    
    def display_buffered_frame(self):
        """Display frame from buffer"""
        if not self.frame_buffer:
            return
        
        current_time = time.time() * 1000
        
        for frame_data in list(self.frame_buffer):
            age = current_time - frame_data['arrival_time']
            
            if age >= self.buffer_delay_ms:
                self.current_pts_raw = frame_data['pts_raw']
                
                # Get bboxes from WebSocket cache
                bboxes = self.get_bboxes_from_cache(frame_data['pts_raw'])
                self.bbox_overlay.set_bboxes(bboxes)
                
                self.display_frame(frame_data['frame'], age)
                self.frame_buffer.remove(frame_data)
                break
    
    def get_bboxes_from_cache(self, pts_raw: int, tolerance_pts: int = None):
        """Get bboxes from WebSocket cache"""
        if tolerance_pts is None:
            tolerance_pts = int((100 / 1000.0) / self.time_base)
        
        # Exact match
        if pts_raw in self.bbox_cache:
            return self.bbox_cache[pts_raw]
        
        # Find closest within tolerance
        for cached_pts, bboxes in self.bbox_cache.items():
            if abs(cached_pts - pts_raw) <= tolerance_pts:
                return bboxes
        
        return []
    
    def display_frame(self, frame, actual_delay):
        """Display frame"""
        h, w, ch = frame.shape
        
        if self.frame_width != w or self.frame_height != h:
            self.frame_width = w
            self.frame_height = h
            self.bbox_overlay.set_frame_dimensions(w, h)
        
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        self.video_label.setPixmap(pixmap)
        self.bbox_overlay.setGeometry(0, 0, self.video_label.width(), self.video_label.height())
        self.bbox_overlay.raise_()
        
        self.capture_replay_frame()
        
        # Calculate display FPS
        current_time = time.time()
        self.fps_frame_count += 1
        time_elapsed = current_time - self.fps_last_update
        
        if time_elapsed >= 1.0:
            self.current_display_fps = self.fps_frame_count / time_elapsed
            self.fps_frame_count = 0
            self.fps_last_update = current_time
        
        # Format PTS
        pts_seconds = self.current_pts_raw * self.time_base
        hours = int(pts_seconds // 3600)
        minutes = int((pts_seconds % 3600) // 60)
        seconds = pts_seconds % 60
        
        if hours > 0:
            pts_str = f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
        else:
            pts_str = f"{minutes:02d}:{seconds:05.2f}"
        
        actual_replay_duration = len(self.replay_buffer) / self.stream_fps
        self.save_replay_btn.setText(f"💾 Save Last {actual_replay_duration:.1f}s")
        
        self.info_label.setText(
            f"Video ID: {self.video_id} | "
            f"FPS: {self.current_display_fps:.1f} | "
            f"PTS: {pts_str} | "
            f"Delay: {actual_delay:.0f}ms | "
            f"Cache: {len(self.bbox_cache)} | "
        )
    
    def capture_replay_frame(self):
        """Capture frame for replay"""
        if self.video_label.pixmap() is None or self.video_label.pixmap().isNull():
            return
        
        displayed_pixmap = self.video_label.pixmap()
        width = displayed_pixmap.width()
        height = displayed_pixmap.height()
        
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.black)
        
        painter = QPainter(pixmap)
        painter.drawPixmap(0, 0, displayed_pixmap)
        
        painter.save()
        scale_x = width / self.bbox_overlay.width() if self.bbox_overlay.width() > 0 else 1.0
        scale_y = height / self.bbox_overlay.height() if self.bbox_overlay.height() > 0 else 1.0
        painter.scale(scale_x, scale_y)
        self.bbox_overlay.render(painter)
        painter.restore()
        
        painter.end()
        
        if self.replay_resolution is None:
            self.replay_resolution = pixmap.size()
        
        if pixmap.size() != self.replay_resolution:
            pixmap = pixmap.scaled(
                self.replay_resolution,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        pixmap.save(buffer, "JPEG", quality=85)
        jpeg_data = buffer.data()
        buffer.close()
        
        self.replay_buffer.append(bytes(jpeg_data))
    
    def save_replay(self):
        """Save replay to file"""
        if len(self.replay_buffer) == 0:
            QMessageBox.warning(self, "No Replay Data", "Replay buffer is empty!")
            return
        
        if not self.fps_locked:
            QMessageBox.warning(self, "FPS Not Ready", "Stream FPS not yet detected.")
            return
        
        if self.save_thread is not None and self.save_thread.isRunning():
            QMessageBox.information(self, "Save in Progress", "Already saving...")
            return
        
        actual_duration = len(self.replay_buffer) / self.stream_fps
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Replay",
            f"replay_video{self.video_id}_{actual_duration:.1f}s_{int(time.time())}.mp4",
            "Video Files (*.mp4)"
        )
        
        if not file_path:
            return
        
        self.save_thread = ReplaySaveThread(
            self.replay_buffer,
            file_path,
            self.replay_resolution,
            int(round(self.stream_fps))
        )
        self.save_thread.progress.connect(self.on_save_progress)
        self.save_thread.finished.connect(self.on_save_finished)
        self.save_thread.start()
        
        self.save_replay_btn.setEnabled(False)
    
    def on_save_progress(self, current, total):
        """Update save progress"""
        percent = (current / total) * 100
        self.save_replay_btn.setText(f"💾 Saving... {percent:.0f}%")
    
    def on_save_finished(self, success, message):
        """Handle save completion"""
        actual_duration = len(self.replay_buffer) / self.stream_fps
        self.save_replay_btn.setText(f"💾 Save Last {actual_duration:.1f}s")
        self.save_replay_btn.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "Replay Saved", message)
        else:
            QMessageBox.critical(self, "Save Error", message)
    
    def toggle_bboxes(self, checked):
        """Toggle bbox visibility"""
        self.bbox_overlay.toggle_visibility(not checked)
        self.bbox_toggle.setText("Show BBoxes" if checked else "Hide BBoxes")
    
    def stop_watching(self):
        """Stop watching stream"""
        self.cleanup()
        self.closed.emit(self.video_id)
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'ws_client') and self.ws_client:
            self.ws_client.disconnect()
        if hasattr(self, 'stream_thread'):
            self.stream_thread.stop()
        if hasattr(self, 'display_timer') and self.display_timer:
            self.display_timer.stop()
        if hasattr(self, 'ping_timer'):
            self.ping_timer.stop()
        if hasattr(self, 'save_thread') and self.save_thread:
            self.save_thread.wait()
    
    def closeEvent(self, event):
        """Handle close event"""
        self.cleanup()
        event.accept()