import av
import time
import numpy as np
import threading
from collections import deque
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QFileDialog, QMessageBox
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt, QBuffer, QIODevice
from PyQt6.QtGui import QImage, QPixmap, QPainter
from widgets.bbox_overlay import BBoxOverlay
from websocket_client import WebSocketClient


class ReplaySaveThread(QThread):
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
                
                if qimage.width() != width or qimage.height() != height:
                    qimage = qimage.scaled(width, height, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
                
                ptr = qimage.bits()
                ptr.setsize(qimage.sizeInBytes())
                arr = np.array(ptr).reshape(height, width, 3)
                
                frame = av.VideoFrame.from_ndarray(arr, format='rgb24')
                frame_yuv = frame.reformat(format=stream.pix_fmt)
                
                for packet in stream.encode(frame_yuv):
                    output.mux(packet)
                
                self.progress.emit(i + 1, total_frames)
            
            for packet in stream.encode():
                output.mux(packet)
            
            output.close()
            self.finished.emit(True, f"Replay saved successfully to:\n{self.file_path}")
            
        except Exception as e:
            self.finished.emit(False, f"Failed to save replay:\n{str(e)}")


class VideoStreamThread(QThread):
    frame_ready = pyqtSignal(bytes, tuple, int, int)
    stream_info = pyqtSignal(float)
    
    def __init__(self, dash_url: str):
        super().__init__()
        self.dash_url = dash_url
        self._stop_flag = threading.Event()
        self.frame_count = 0
    
    def run(self):
        container = None
        while not self._stop_flag.is_set():
            try:
                if self._stop_flag.is_set():
                    break
                
                options = {
                    'timeout': '10000000',
                    'reconnect': '1',
                    'reconnect_streamed': '1',
                    'reconnect_delay_max': '2'
                }
                
                container = av.open(self.dash_url, options=options, timeout=10.0)
                
                if self._stop_flag.is_set():
                    break
                
                if len(container.streams.video) == 0:
                    time.sleep(2)
                    continue
                
                video_stream = container.streams.video[0]
                stream_time_base = float(video_stream.time_base)
                
                stream_fps = 30.0
                if video_stream.average_rate:
                    stream_fps = float(video_stream.average_rate)
                elif video_stream.guessed_rate:
                    stream_fps = float(video_stream.guessed_rate)
                elif video_stream.codec_context.framerate:
                    stream_fps = float(video_stream.codec_context.framerate)
                
                if not self._stop_flag.is_set():
                    self.stream_info.emit(stream_fps)
                
                # TIMING FIX: Track the last PTS we processed to detect discontinuities
                last_pts = None
                start_wall_time = None
                start_pts_time = None
                
                for packet in container.demux(video_stream):
                    if self._stop_flag.is_set():
                        break
                    
                    try:
                        frames = packet.decode()
                        if not frames:
                            continue

                        for frame in frames:
                            if self._stop_flag.is_set():
                                break
                            
                            if frame.pts is None:
                                continue

                            current_pts_seconds = float(frame.pts * stream_time_base)
                            
                            # TIMING FIX: Detect PTS discontinuities (jumps/resets)
                            if last_pts is not None:
                                pts_diff = current_pts_seconds - last_pts
                                # If jump is > 2 seconds or negative, reset timing
                                if abs(pts_diff) > 2.0:
                                    start_wall_time = None
                                    start_pts_time = None
                            
                            last_pts = current_pts_seconds
                            
                            # TIMING FIX: Initialize or reset timing anchors
                            if start_wall_time is None:
                                start_wall_time = time.perf_counter()
                                start_pts_time = current_pts_seconds
                            
                            # TIMING FIX: Calculate precise target display time
                            elapsed_stream_time = current_pts_seconds - start_pts_time
                            target_wall_time = start_wall_time + elapsed_stream_time
                            current_wall_time = time.perf_counter()
                            
                            # TIMING FIX: Sleep until it's time to emit this frame
                            wait_time = target_wall_time - current_wall_time
                            
                            # Only sleep if we're ahead, with minimum sleep threshold
                            if wait_time > 0.001:
                                time.sleep(wait_time)
                            elif wait_time < -0.1:
                                # TIMING FIX: If we're more than 100ms behind, skip to catch up
                                # But still emit the frame so display doesn't freeze
                                pass
                            
                            self.frame_count += 1
                            img = frame.to_ndarray(format='rgb24')
                            
                            frame_bytes = img.tobytes()
                            frame_shape = img.shape
                            
                            pts_90khz = int(frame.pts * (90000 * video_stream.time_base))
                            
                            if not self._stop_flag.is_set():
                                self.frame_ready.emit(frame_bytes, frame_shape, self.frame_count, pts_90khz)
                    
                    except av.AVError:
                        pass
            
            except Exception:
                if self._stop_flag.is_set():
                    break
                time.sleep(2)
            
            finally:
                if container:
                    try:
                        container.close()
                    except:
                        pass
                    container = None
    
    def stop(self):
        self._stop_flag.set()


class VideoPlayerWidget(QWidget):
    closed = pyqtSignal(int)
    
    def __init__(self, video_id: int, dash_manifest_url: str, stream_start_time_ms: int, 
                 backend_url: str, replay_duration_seconds: float = 30.0, 
                 parent=None):
        super().__init__(parent)
        self.video_id = video_id
        self.backend_url = backend_url
        self.stream_start_time_ms = stream_start_time_ms
        
        self.dash_url = f"{backend_url}{dash_manifest_url}"
        
        self.replay_duration_seconds = replay_duration_seconds
        self.bbox_cache = {}
        self.bbox_cache_max_age_seconds = 5.0
        
        self.current_pts_90khz = 0
        # self.first_frame_pts = None # No longer needed for PTS display
        
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
        
        self._cleanup_done = False
        
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
    
    def on_websocket_status(self, connected: bool, message: str):
        if self._cleanup_done:
            return
        self.ws_connected = connected
        if connected:
            self.ws_status.setText("🟢 WS: Connected")
            self.ws_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.ws_status.setText(f"🔴 WS: {message}")
            self.ws_status.setStyleSheet("color: red; font-weight: bold;")
    
    def on_websocket_error(self, error_msg: str):
        pass
    
    def on_websocket_stream_info(self, data: dict):
        pass
    
    def on_websocket_bbox(self, data: dict):
        if self._cleanup_done:
            return
        pts_90khz = data.get('pts')
        bboxes = data.get('bboxes', [])
        
        if pts_90khz is not None:
            self.bbox_cache[pts_90khz] = bboxes
            self.cleanup_bbox_cache()
    
    def cleanup_bbox_cache(self):
        if not self.bbox_cache:
            return
        
        # Use a PTS value 5 seconds *before* the current stream PTS
        # as the cutoff.
        cutoff_pts = self.current_pts_90khz - int(self.bbox_cache_max_age_seconds * 90000.0)
        
        to_remove = [pts for pts in self.bbox_cache if pts < cutoff_pts]
        for pts in to_remove:
            del self.bbox_cache[pts]
    
    def start_stream(self):
        self.stream_thread = VideoStreamThread(self.dash_url)
        self.stream_thread.frame_ready.connect(self.on_frame_ready)
        self.stream_thread.stream_info.connect(self.on_stream_info)
        self.stream_thread.start()
    
    def on_stream_info(self, fps: float):
        if self._cleanup_done:
            return
        if not self.fps_locked:
            self.stream_fps = fps
            self.fps_locked = True
            self._update_replay_buffer_size()
    
    def on_frame_ready(self, frame_bytes, frame_shape, frame_number, pts_90khz):
        if self._cleanup_done:
            return
        
        try:
            self.current_pts_90khz = pts_90khz
            
            # if self.first_frame_pts is None: # No longer needed
            #     self.first_frame_pts = pts_90khz
            
            bboxes = self.get_bboxes_for_frame(pts_90khz)
            
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(frame_shape)
            self.bbox_overlay.set_bboxes(bboxes)
            self.display_frame(frame)
            
        except Exception as e:
            print(f"[ERROR] on_frame_ready: {e}")
    
    def get_bboxes_for_frame(self, frame_pts_90khz: int):
        # FIX 3: (From your original code) Re-written retention logic to be unified and correct.
        retention_frames = self.bbox_overlay.bbox_retention_frames
        all_bboxes = []
        
        if self.stream_fps > 0:
            pts_per_frame = 90000.0 / self.stream_fps
        else:
            pts_per_frame = 3000.0  # 30fps default
        
        # Tolerance window is half a frame
        tolerance_pts = int(pts_per_frame * 0.5)
        
        # The window *ends* at the current frame's PTS + tolerance
        # (e.g., show bboxes for this frame or slightly in the future)
        pts_window_end = frame_pts_90khz + tolerance_pts
        
        # The window *starts* (n-1) frames in the past, *also with tolerance*
        # This (n-1) calculation is key for retention.
        pts_range_past = int(pts_per_frame * (retention_frames - 1))
        pts_window_start = frame_pts_90khz - pts_range_past - tolerance_pts
        
        # This unified logic works for n=1 and n>1:
        # n=1: pts_range_past = 0
        #   -> (frame_pts - tol) <= cached_pts <= (frame_pts + tol)
        # n=2: pts_range_past = 1 * pts_per_frame
        #   -> (frame_pts - 1_frame - tol) <= cached_pts <= (frame_pts + tol)
        
        for cached_pts, bboxes in self.bbox_cache.items():
            if pts_window_start <= cached_pts <= pts_window_end:
                all_bboxes.extend(bboxes)
        
        return all_bboxes
    
    def display_frame(self, frame):
        if self._cleanup_done:
            return
        h, w, ch = frame.shape
        
        if self.frame_width != w or self.frame_height != h:
            self.frame_width = w
            self.frame_height = h
            self.bbox_overlay.set_frame_dimensions(w, h)
        
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
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
        
        # FIX 1: (From your original code) Changed PTS calculation to show absolute stream time.
        # Instead of subtracting the 'first_frame_pts' (time since client
        # started watching), we just use the current_pts_90khz directly.
        # This shows the actual media timestamp from the stream.
        elapsed_seconds = max(0.0, self.current_pts_90khz / 90000.0)
        
        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        seconds = elapsed_seconds % 60
        
        # Show hours only if they are non-zero
        if hours > 0:
            pts_str = f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
        else:
            pts_str = f"{minutes:02d}:{seconds:05.2f}"

        actual_replay_duration = len(self.replay_buffer) / max(self.stream_fps, 1.0)
        self.save_replay_btn.setText(f"💾 Save Last {actual_replay_duration:.1f}s")
        
        self.info_label.setText(
            f"Video ID: {self.video_id} | "
            f"Stream FPS: {self.stream_fps:.1f} | "
            f"Display FPS: {self.current_display_fps:.1f} | "
            f"PTS: {pts_str} | "
            f"Cache: {len(self.bbox_cache)} | "
            f"LIVE"
        )
    
    def capture_replay_frame(self):
        if self._cleanup_done:
            return
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
        if self._cleanup_done:
            return
        percent = (current / total)* 100
        self.save_replay_btn.setText(f"💾 Saving... {percent:.0f}%")
    
    def on_save_finished(self, success, message):
        if self._cleanup_done:
            return
        actual_duration = len(self.replay_buffer) / max(self.stream_fps, 1.0)
        self.save_replay_btn.setText(f"💾 Save Last {actual_duration:.1f}s")
        self.save_replay_btn.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "Replay Saved", message)
        else:
            QMessageBox.critical(self, "Save Error", message)
    
    def stop_watching(self):
        if self._cleanup_done:
            return
        
        self._cleanup_done = True
        self.stop_button.setEnabled(False)
        
        if hasattr(self, 'stream_thread'):
            self.stream_thread.stop()
            
        if hasattr(self, 'ws_client') and self.ws_client:
            self.ws_client.disconnect()
        
        QTimer.singleShot(200, self.finish_cleanup)
    
    def finish_cleanup(self):
        if hasattr(self, 'stream_thread'):
            self.stream_thread.wait(3000)
            if self.stream_thread.isRunning():
                self.stream_thread.terminate()
                self.stream_thread.wait()
        
        if hasattr(self, 'save_thread') and self.save_thread and self.save_thread.isRunning():
            self.save_thread.wait()
        
        self.closed.emit(self.video_id)
    
    def cleanup(self):
        if not self._cleanup_done:
            self.stop_watching()
    
    def closeEvent(self, event):
        self.cleanup()
        event.accept()