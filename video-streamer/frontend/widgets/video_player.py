"""
Video player using WebSocket for real-time bbox synchronization (LIVE MODE - NO DELAY)
"""

import av
import time
import numpy as np
import threading  # Import threading
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
        self.container = None
        self._lock = threading.Lock()  # Lock to protect container closing
    
    def run(self):
        """Read frames from SRT stream"""
        self.running = True
        
        while self.running:
            local_container = None
            try:
                print(f"[VideoStreamThread] Connecting to {self.stream_url}...")
                
                # SRT-specific options
                options = {
                    'mode': 'caller', 'latency': '200000', 'maxbw': '0',
                    'timeout': '5000000', 'connect_timeout': '5000000',
                    'rcvbuf': '48234496', 'sndbuf': '48234496',
                    'peerlatency': '0', 'tlpktdrop': '1', 'payload_size': '1316',
                }
                
                # Check running flag *before* blocking call
                if not self.running:
                    break
                
                # av.open() can block. We don't hold the lock here.
                local_container = av.open(self.stream_url, options=options, timeout=10.0)
                
                # Now that we have a container, acquire lock to assign it
                with self._lock:
                    if not self.running:
                        # stop() was called while we were in av.open()
                        local_container.close()
                        break
                    self.container = local_container
                
                if len(self.container.streams.video) == 0:
                    self.error.emit("No video stream found")
                    time.sleep(2)
                    continue # Loop will go to finally, close, and retry
                
                video_stream = self.container.streams.video[0]
                
                # Get FPS
                stream_fps = 30.0
                if video_stream.average_rate:
                    stream_fps = float(video_stream.average_rate)
                elif video_stream.guessed_rate:
                    stream_fps = float(video_stream.guessed_rate)
                elif video_stream.codec_context.framerate:
                    stream_fps = float(video_stream.codec_context.framerate)
                
                self.stream_info.emit(stream_fps)
                time_base = float(video_stream.time_base)
                
                frame_received = False
                waiting_for_keyframe = True
                
                for packet in self.container.demux(video_stream):
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
            
            except Exception as e:
                if not self.running:
                    # This was an intentional stop. The container is (or is being)
                    # closed by stop(). We just need to exit.
                    break
                
                error_msg = f"Stream error: {str(e)}"
                self.error.emit(error_msg)
                time.sleep(2) # Wait before retrying
            
            finally:
                # This block runs on loop exit (break), error, or normal stream end.
                # Use the lock to ensure stop() isn't *also* closing it.
                with self._lock:
                    if self.container:
                        try:
                            self.container.close()
                        except Exception as e:
                            print(f"[VideoStreamThread] Error in finally close: {e}")
                        self.container = None
                    elif local_container:
                        # Case where stop() was called during av.open()
                        # and self.container was never set
                        try:
                            local_container.close()
                        except Exception as e:
                            print(f"[VideoStreamThread] Error in finally (local) close: {e}")
        
        print(f"[VideoStreamThread] Thread for {self.stream_url} has stopped.")
    
    def stop(self):
        """Stop reading frames"""
        print(f"[VideoStreamThread] Stopping thread for {self.stream_url}...")
        
        with self._lock:
            # Set running flag *inside* lock to prevent race condition
            # where run() loop checks running, then stop() sets it,
            # then run() thread proceeds to open a new container.
            self.running = False 
            
            if self.container:
                try:
                    self.container.close()
                except Exception as e:
                    print(f"[VideoStreamThread] Error closing container in stop(): {e}")
                # Set to None to prevent finally block from re-closing
                self.container = None 
        
        self.wait(3000)


class VideoPlayerWidget(QWidget):
    """Video player with WebSocket-based bbox synchronization (LIVE MODE - NO DELAY)"""
    
    closed = pyqtSignal(int)
    
    def __init__(self, video_id: int, port: int, stream_start_time_ms: int, 
                 backend_url: str, replay_duration_seconds: float = 30.0, 
                 parent=None):
        super().__init__(parent)
        self.video_id = video_id
        self.backend_url = backend_url
        self.stream_start_time_ms = stream_start_time_ms
        
        # Extract host
        backend_host = backend_url.replace('http://', '').replace('https://', '')
        if ':' in backend_host:
            backend_host = backend_host.split(':')[0]
        if '/' in backend_host:
            backend_host = backend_host.split('/')[0]
        
        self.stream_url = f"srt://{backend_host}:{port}"
        print(f"[VideoPlayer] Constructed SRT URL: {self.stream_url}")
        
        self.replay_duration_seconds = replay_duration_seconds
        self.bbox_cache = {}
        self.bbox_cache_max_age_seconds = 5.0
        
        self.current_pts_raw = 0
        self.time_base = 1/90000.0
        
        self.stream_fps = 30.0
        self.fps_locked = False
        self.current_display_fps = 0.0
        self.fps_frame_count = 0
        self.fps_last_update = time.time()
        
        self.frame_width = 0
        self.frame_height = 0
        
        self.replay_buffer = deque(maxlen=self._calculate_replay_buffer_size())
        self.replay_resolution = None
        self.save_thread = None
        
        self.ws_client = None
        self.ws_connected = False
        
        self.setup_ui()
        self.start_stream()
        self.connect_websocket()
    
    def _calculate_replay_buffer_size(self) -> int:
        return max(int(self.stream_fps * self.replay_duration_seconds), 30)
    
    def _update_replay_buffer_size(self):
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
        layout = QVBoxLayout(self)
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setScaledContents(True)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label)
        
        self.bbox_overlay = BBoxOverlay(self.video_label)
        self.bbox_overlay.setGeometry(self.video_label.geometry())
        
        controls = QHBoxLayout()
        
        self.ws_status = QLabel("🔴 WS: Disconnected")
        self.ws_status.setStyleSheet("color: red; font-weight: bold;")
        controls.addWidget(self.ws_status)
        
        self.info_label = QLabel(f"Video ID: {self.video_id} | FPS: 0.0 | PTS: 0 | Cache: 0 | LIVE")
        controls.addWidget(self.info_label)
        
        controls.addStretch()
        
        self.save_replay_btn = QPushButton("💾 Save Last 0.0s")
        self.save_replay_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_replay_btn.setStyleSheet("""
            QPushButton { background-color: #27ae60; color: white; font-weight: bold; padding: 8px 16px; border-radius: 4px; }
            QPushButton:hover { background-color: #229954; }
        """)
        self.save_replay_btn.clicked.connect(self.save_replay)
        controls.addWidget(self.save_replay_btn)
        
        self.stop_button = QPushButton("⏹ Stop Watching")
        self.stop_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_button.setStyleSheet("""
            QPushButton { background-color: #e74c3c; color: white; font-weight: bold; padding: 8px 16px; border-radius: 4px; }
            QPushButton:hover { background-color: #c0392b; }
        """)
        self.stop_button.clicked.connect(self.stop_watching)
        controls.addWidget(self.stop_button)
        
        layout.addLayout(controls)
    
    def connect_websocket(self):
        self.ws_client = WebSocketClient(self.backend_url, self.video_id)
        self.ws_client.bbox_received.connect(self.on_websocket_bbox)
        self.ws_client.stream_info_received.connect(self.on_websocket_stream_info)
        self.ws_client.connection_status.connect(self.on_websocket_status)
        self.ws_client.error_occurred.connect(self.on_websocket_error)
        self.ws_client.connect()
        
        self.ping_timer = QTimer()
        self.ping_timer.timeout.connect(self.send_websocket_ping)
        self.ping_timer.start(15000)
    
    def on_websocket_status(self, connected: bool, message: str):
        self.ws_connected = connected
        if connected:
            self.ws_status.setText("🟢 WS: Connected")
            self.ws_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.ws_status.setText(f"🔴 WS: {message}")
            self.ws_status.setStyleSheet("color: red; font-weight: bold;")
    
    def on_websocket_error(self, error_msg: str):
        print(f"[WebSocket] Error: {error_msg}")
    
    def on_websocket_stream_info(self, data: dict):
        print(f"[WebSocket] Stream info received: {data}")
    
    def on_websocket_bbox(self, data: dict):
        pts_raw = data.get('pts')
        bboxes = data.get('bboxes', [])
        
        if pts_raw is not None:
            self.bbox_cache[pts_raw] = bboxes
            self.cleanup_bbox_cache()
    
    def cleanup_bbox_cache(self):
        if not self.bbox_cache or self.time_base == 0:
            return
        
        cutoff_seconds = self.bbox_cache_max_age_seconds
        cutoff_pts = int(cutoff_seconds / self.time_base)
        min_pts = self.current_pts_raw - cutoff_pts
        
        to_remove = [pts for pts in self.bbox_cache if pts < min_pts]
        for pts in to_remove:
            del self.bbox_cache[pts]
    
    def send_websocket_ping(self):
        if self.ws_client and self.ws_connected:
            self.ws_client.send_ping()
    
    def start_stream(self):
        self.stream_thread = VideoStreamThread(self.stream_url)
        self.stream_thread.frame_ready.connect(self.on_frame_ready)
        self.stream_thread.stream_info.connect(self.on_stream_info)
        self.stream_thread.error.connect(self.on_stream_error)
        self.stream_thread.start()
    
    def on_stream_info(self, fps: float):
        if not self.fps_locked:
            self.stream_fps = fps
            self.fps_locked = True
            self._update_replay_buffer_size()
            print(f"[VideoPlayer] Stream FPS locked: {fps:.2f}")
    
    def on_frame_ready(self, frame, frame_number, pts_raw, time_base):
        self.time_base = time_base
        self.current_pts_raw = pts_raw
        
        bboxes = self.get_bboxes_with_retention(pts_raw)
        self.bbox_overlay.set_bboxes(bboxes)
        
        self.display_frame(frame)
    
    def get_bboxes_with_retention(self, pts_raw: int, tolerance_pts: int = None):
        if self.time_base == 0: return []
        if tolerance_pts is None:
            tolerance_pts = int((100 / 1000.0) / self.time_base)
        
        retention_frames = self.bbox_overlay.bbox_retention_frames
        all_bboxes = []
        
        if self.stream_fps > 0:
            pts_per_frame = int((1.0 / self.stream_fps) / self.time_base)
            pts_range = pts_per_frame * retention_frames
        else:
            pts_range = int((retention_frames * 0.033) / self.time_base)
        
        for cached_pts, bboxes in self.bbox_cache.items():
            if pts_raw - pts_range <= cached_pts <= pts_raw + tolerance_pts:
                all_bboxes.extend(bboxes)
        
        return all_bboxes
    
    def on_stream_error(self, error_msg: str):
        print(f"[VideoPlayer] Stream error: {error_msg}")
        self.info_label.setText(f"Video ID: {self.video_id} | Error: {error_msg}")
    
    def display_frame(self, frame):
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
        
        current_time = time.time()
        self.fps_frame_count += 1
        time_elapsed = current_time - self.fps_last_update
        
        if time_elapsed >= 1.0:
            self.current_display_fps = self.fps_frame_count / time_elapsed
            self.fps_frame_count = 0
            self.fps_last_update = current_time
        
        pts_seconds = self.current_pts_raw * self.time_base
        hours = int(pts_seconds // 3600)
        minutes = int((pts_seconds % 3600) // 60)
        seconds = pts_seconds % 60
        
        pts_str = f"{minutes:02d}:{seconds:05.2f}"
        if hours > 0:
            pts_str = f"{hours:02d}:{pts_str}"
        
        actual_replay_duration = len(self.replay_buffer) / max(self.stream_fps, 1.0)
        self.save_replay_btn.setText(f"💾 Save Last {actual_replay_duration:.1f}s")
        
        self.info_label.setText(
            f"Video ID: {self.video_id} | "
            f"FPS: {self.current_display_fps:.1f} | "
            f"PTS: {pts_str} | "
            f"Cache: {len(self.bbox_cache)} | "
            f"LIVE"
        )
    
    def capture_replay_frame(self):
        pixmap_to_save = self.video_label.pixmap()
        if pixmap_to_save is None or pixmap_to_save.isNull():
            return
        
        width = pixmap_to_save.width()
        height = pixmap_to_save.height()
        
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.black)
        
        painter = QPainter(pixmap)
        painter.drawPixmap(0, 0, pixmap_to_save)
        
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
        self.replay_buffer.append(bytes(buffer.data()))
        buffer.close()
    
    def save_replay(self):
        if len(self.replay_buffer) == 0:
            QMessageBox.warning(self, "No Replay Data", "Replay buffer is empty!")
            return
        
        if not self.fps_locked or self.stream_fps <= 0:
            QMessageBox.warning(self, "FPS Not Ready", "Stream FPS not yet detected.")
            return
        
        if self.save_thread is not None and self.save_thread.isRunning():
            QMessageBox.information(self, "Save in Progress", "Already saving...")
            return
        
        actual_duration = len(self.replay_buffer) / self.stream_fps
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Replay",
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
        percent = (current / total) * 100
        self.save_replay_btn.setText(f"💾 Saving... {percent:.0f}%")
    
    def on_save_finished(self, success, message):
        actual_duration = len(self.replay_buffer) / max(self.stream_fps, 1.0)
        self.save_replay_btn.setText(f"💾 Save Last {actual_duration:.1f}s")
        self.save_replay_btn.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "Replay Saved", message)
        else:
            QMessageBox.critical(self, "Save Error", message)
    
    def stop_watching(self):
        self.cleanup()
        self.closed.emit(self.video_id)
    
    def cleanup(self):
        if hasattr(self, 'ping_timer'):
            self.ping_timer.stop()
        
        # Stop stream thread *before* websocket
        if hasattr(self, 'stream_thread'):
            self.stream_thread.stop()
            
        if hasattr(self, 'ws_client') and self.ws_client:
            self.ws_client.disconnect()
        
        if hasattr(self, 'save_thread') and self.save_thread:
            self.save_thread.wait()
    
    def closeEvent(self, event):
        self.cleanup()
        event.accept()