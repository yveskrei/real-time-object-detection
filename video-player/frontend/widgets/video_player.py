"""
frontend/widgets/video_player.py - Video player using PyAV with raw PTS and instant replay
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


class ReplaySaveThread(QThread):
    """Background thread for saving replay to avoid UI freezing"""
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, replay_buffer, file_path, resolution):
        super().__init__()
        self.replay_buffer = list(replay_buffer)  # Copy buffer
        self.file_path = file_path
        self.resolution = resolution
    
    def run(self):
        try:
            width = self.resolution.width()
            height = self.resolution.height()
            total_frames = len(self.replay_buffer)
            
            # Create output video with PyAV
            output = av.open(self.file_path, mode='w')
            stream = output.add_stream('h264', rate=30)
            stream.width = width
            stream.height = height
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': '23', 'preset': 'medium'}
            
            # Write frames
            for i, jpeg_data in enumerate(self.replay_buffer):
                # Load JPEG
                qimage = QImage.fromData(jpeg_data, "JPEG")
                
                # Convert QImage to numpy array
                qimage = qimage.convertToFormat(QImage.Format.Format_RGB888)
                ptr = qimage.bits()
                ptr.setsize(qimage.sizeInBytes())
                arr = np.array(ptr).reshape(height, width, 3)
                
                # Create VideoFrame
                frame = av.VideoFrame.from_ndarray(arr, format='rgb24')
                
                # Encode and write
                for packet in stream.encode(frame):
                    output.mux(packet)
                
                # Emit progress
                self.progress.emit(i + 1, total_frames)
            
            # Flush encoder
            for packet in stream.encode():
                output.mux(packet)
            
            output.close()
            
            self.finished.emit(True, f"Replay saved successfully to:\n{self.file_path}")
            
        except Exception as e:
            self.finished.emit(False, f"Failed to save replay:\n{str(e)}")



class VideoStreamThread(QThread):
    """Thread for reading video stream frames with real global PTS from FFmpeg"""
    frame_ready = pyqtSignal(object, int, int, float)  # numpy_frame, frame_number, pts_raw, time_base
    error = pyqtSignal(str)
    
    def __init__(self, stream_url: str):
        super().__init__()
        self.stream_url = stream_url
        self.running = False
        self.frame_count = 0
    
    def run(self):
        """Read frames from stream using PyAV and extract real PTS"""
        self.running = True
        container = None
        
        while self.running:
            try:
                print(f"[VideoStreamThread] Connecting to {self.stream_url}...")
                
                # For UDP streams, we need to format the URL correctly
                # Change udp://127.0.0.1:port to udp://@127.0.0.1:port (listen mode)
                stream_url = self.stream_url
                if stream_url.startswith('udp://') and '@' not in stream_url:
                    # Add @ to indicate listen mode
                    stream_url = stream_url.replace('udp://', 'udp://@')
                    print(f"[VideoStreamThread] Adjusted URL to listen mode: {stream_url}")
                
                # Open container with PyAV
                container = av.open(stream_url, options={
                    'rtbufsize': '100M',  # Increase buffer for UDP
                    'fflags': 'nobuffer',  # Minimize buffering delay
                    'max_delay': '500000',  # 0.5 seconds
                    'reorder_queue_size': '0',  # Disable reordering for lower latency
                    'reuse': '1',  # Allow reusing the socket (SO_REUSEADDR)
                }, timeout=10.0)
                
                print(f"[VideoStreamThread] Connected! Container: {container}")
                
                # Get video stream
                if len(container.streams.video) == 0:
                    self.error.emit("No video stream found")
                    if container:
                        container.close()
                        container = None
                    time.sleep(2)
                    continue
                
                video_stream = container.streams.video[0]
                print(f"[VideoStreamThread] Video stream found: {video_stream}")
                print(f"[VideoStreamThread] Codec: {video_stream.codec_context.name}")
                print(f"[VideoStreamThread] Size: {video_stream.codec_context.width}x{video_stream.codec_context.height}")
                print(f"[VideoStreamThread] Time base: {video_stream.time_base}")
                
                # Get time base for PTS conversion
                time_base = float(video_stream.time_base)
                
                # Decode frames
                frame_received = False
                waiting_for_keyframe = True
                
                for packet in container.demux(video_stream):
                    if not self.running:
                        break
                    
                    # If we're waiting for a keyframe, skip non-key packets
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
                                print(f"[VideoStreamThread] Actual size: {frame.width}x{frame.height}")
                                frame_received = True
                            
                            self.frame_count += 1
                            
                            # Get REAL PTS from the frame (global to the stream) - RAW value
                            pts_raw = frame.pts
                            
                            if pts_raw is None:
                                # Fallback: estimate from frame count
                                # Assume 30fps and 90kHz time base if no PTS
                                pts_raw = int((self.frame_count / 30.0) / time_base)
                                if self.frame_count % 30 == 0:
                                    print(f"[VideoStreamThread] Warning: No PTS, using estimated: {pts_raw}")
                            
                            # Convert frame to numpy array (RGB format)
                            img = frame.to_ndarray(format='rgb24')
                            
                            # Emit frame with RAW PTS and time_base for frontend conversion
                            self.frame_ready.emit(img, self.frame_count, pts_raw, time_base)
                    
                    except av.AVError as decode_error:
                        # Decoder error - wait for next keyframe
                        if not frame_received:
                            print(f"[VideoStreamThread] Decode error (waiting for keyframe): {decode_error}")
                            waiting_for_keyframe = True
                        else:
                            # If we were already decoding, this is a real error
                            raise
                
                # If we exit the loop normally, close and try to reconnect
                print(f"[VideoStreamThread] Stream ended, reconnecting...")
                if container:
                    container.close()
                    container = None
                
            except Exception as e:
                error_msg = f"Stream error: {str(e)}"
                print(f"[VideoStreamThread] {error_msg}")
                self.error.emit(error_msg)
                
                # Make sure container is closed
                if container:
                    try:
                        container.close()
                    except:
                        pass
                    container = None
                
                # Wait a bit before reconnecting
                time.sleep(2)
        
        # Final cleanup when stopping
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
    """Video player with bbox overlay using real global PTS for synchronization"""
    
    closed = pyqtSignal(int)  # Signal emitted when player is closed
    
    def __init__(self, video_id: int, stream_url: str, stream_start_time_ms: int, api_client, parent=None):
        super().__init__(parent)
        self.video_id = video_id
        self.stream_url = stream_url
        self.stream_start_time_ms = stream_start_time_ms  # For reference only
        self.api_client = api_client
        
        # Buffer settings
        self.buffer_delay_ms = 500  # 500ms delay to allow bbox generation
        self.frame_buffer = deque(maxlen=60)  # Buffer up to 2 seconds at 30fps
        
        # BBox cache: {raw_pts: [bboxes]}
        self.bbox_cache = {}
        self.last_fetch_pts = 0
        self.fetch_interval_ms = 800  # Fetch every 800ms
        self.fetch_lookback_ms = 1000  # Look back 1 second
        
        # Current state
        self.current_pts_raw = 0  # Raw PTS value
        self.time_base = 1/90000.0  # Will be updated from stream
        self.current_fps = 0.0
        self.fps_frame_count = 0
        self.fps_last_update = time.time()
        
        # Frame dimensions
        self.frame_width = 0
        self.frame_height = 0
        
        # Instant replay buffer (30 seconds @ 30fps = 900 frames)
        self.replay_buffer = deque(maxlen=900)
        self.replay_resolution = None  # Will be set on first frame
        self.save_thread = None  # Background save thread
        
        self.setup_ui()
        self.start_stream()
    
    def setup_ui(self):
        """Setup the UI components"""
        layout = QVBoxLayout(self)
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setScaledContents(True)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label)
        
        # BBox overlay on top of video label
        self.bbox_overlay = BBoxOverlay(self.video_label)
        self.bbox_overlay.setGeometry(self.video_label.geometry())
        
        # Controls
        controls = QHBoxLayout()
        
        self.bbox_toggle = QPushButton("Hide BBoxes")
        self.bbox_toggle.setCheckable(True)
        self.bbox_toggle.clicked.connect(self.toggle_bboxes)
        controls.addWidget(self.bbox_toggle)
        
        self.info_label = QLabel(f"Video ID: {self.video_id} | FPS: 0.0 | PTS: 0 | Cache: 0")
        controls.addWidget(self.info_label)
        
        controls.addStretch()
        
        # Save replay button (with counter in text)
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
        
        # Stop watching button
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
    
    def start_stream(self):
        """Start reading video stream"""
        # Start video stream thread with PyAV
        self.stream_thread = VideoStreamThread(self.stream_url)
        self.stream_thread.frame_ready.connect(self.buffer_frame)
        self.stream_thread.error.connect(self.handle_stream_error)
        self.stream_thread.start()
        
        # Timer to display buffered frames (checks every 33ms)
        self.display_timer = QTimer()
        self.display_timer.timeout.connect(self.display_buffered_frame)
        self.display_timer.start(33)  # ~30fps
        
        # Timer to batch fetch bboxes (every 800ms)
        self.fetch_timer = QTimer()
        self.fetch_timer.timeout.connect(self.fetch_bbox_batch)
        self.fetch_timer.start(self.fetch_interval_ms)
    
    def buffer_frame(self, frame, frame_number, pts_raw, time_base):
        """Add incoming frame to buffer with raw PTS"""
        arrival_time = time.time() * 1000  # When frame arrived at UI
        
        # Update time base from stream
        self.time_base = time_base
        
        self.frame_buffer.append({
            'frame': frame,
            'frame_number': frame_number,
            'pts_raw': pts_raw,  # Store raw PTS
            'arrival_time': arrival_time
        })
    
    def handle_stream_error(self, error_msg):
        """Handle stream errors"""
        print(f"[VideoPlayer] Stream error: {error_msg}")
        # Update UI to show error
        self.info_label.setText(f"Video ID: {self.video_id} | Error: {error_msg}")
    
    def fetch_bbox_batch(self):
        """Fetch a batch of bboxes covering recent raw PTS range"""
        if not self.frame_buffer:
            return
        
        # Get the most recent frame's raw PTS
        latest_frame = self.frame_buffer[-1]
        current_pts_raw = latest_frame['pts_raw']
        
        # Convert interval to PTS units using time_base
        fetch_interval_pts = int((self.fetch_interval_ms / 1000.0) / self.time_base)
        lookback_pts = int((self.fetch_lookback_ms / 1000.0) / self.time_base)
        overlap_pts = int((200 / 1000.0) / self.time_base)
        
        # Don't fetch too frequently (in PTS units)
        if current_pts_raw - self.last_fetch_pts < (fetch_interval_pts - overlap_pts):
            return
        
        try:
            # Calculate range: fetch from (current - lookback) to current (in raw PTS)
            from_pts = max(0, current_pts_raw - lookback_pts)
            to_pts = current_pts_raw
            
            # Fetch bboxes in this raw PTS range
            response = self.api_client.get_bboxes_range(
                self.video_id,
                from_pts=int(from_pts),
                to_pts=int(to_pts)
            )
            
            # Cache all returned bboxes by their raw PTS
            for result in response.get('results', []):
                pts = result['pts']  # Raw PTS from backend
                self.bbox_cache[pts] = result['bboxes']
            
            self.last_fetch_pts = current_pts_raw
            
            # Clean old cache entries (keep last 3 seconds in PTS units)
            cutoff_pts = current_pts_raw - int((3000 / 1000.0) / self.time_base)
            self.bbox_cache = {k: v for k, v in self.bbox_cache.items() if k > cutoff_pts}
            
        except Exception as e:
            # Silently handle errors
            pass
    
    def display_buffered_frame(self):
        """Display frame that has been buffered long enough"""
        if not self.frame_buffer:
            return
        
        current_time = time.time() * 1000
        
        # Look for frames that are at least buffer_delay_ms old
        for frame_data in list(self.frame_buffer):
            age = current_time - frame_data['arrival_time']
            
            if age >= self.buffer_delay_ms:
                # This frame is old enough, display it
                self.current_pts_raw = frame_data['pts_raw']
                
                # Get bboxes from cache for this exact raw PTS
                bboxes = self.get_bboxes_from_cache(frame_data['pts_raw'])
                self.bbox_overlay.set_bboxes(bboxes)
                
                # Display frame
                self.display_frame(frame_data['frame'], age)
                
                # Remove this frame from buffer
                self.frame_buffer.remove(frame_data)
                break
    
    def get_bboxes_from_cache(self, pts_raw: int, tolerance_pts: int = None):
        """Get bboxes from cache for a raw PTS (with tolerance)"""
        # Calculate tolerance in PTS units (default 100ms worth of PTS)
        if tolerance_pts is None:
            tolerance_pts = int((100 / 1000.0) / self.time_base)
        
        # Try exact match first
        if pts_raw in self.bbox_cache:
            return self.bbox_cache[pts_raw]
        
        # Find closest PTS within tolerance
        for cached_pts, bboxes in self.bbox_cache.items():
            if abs(cached_pts - pts_raw) <= tolerance_pts:
                return bboxes
        
        # No bboxes found
        return []
    
    def display_frame(self, frame, actual_delay):
        """Convert numpy array to QPixmap and display"""
        # Frame is already in RGB format from PyAV
        h, w, ch = frame.shape
        
        # Update frame dimensions for bbox overlay
        if self.frame_width != w or self.frame_height != h:
            self.frame_width = w
            self.frame_height = h
            self.bbox_overlay.set_frame_dimensions(w, h)
        
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        self.video_label.setPixmap(pixmap)
        
        # IMPORTANT: Update overlay geometry to match video label
        self.bbox_overlay.setGeometry(0, 0, self.video_label.width(), self.video_label.height())
        self.bbox_overlay.raise_()  # Ensure overlay is on top
        
        # Always capture for replay buffer
        self.capture_replay_frame()
        
        # Calculate FPS
        current_time = time.time()
        self.fps_frame_count += 1
        time_elapsed = current_time - self.fps_last_update
        
        if time_elapsed >= 1.0:  # Update FPS every second
            self.current_fps = self.fps_frame_count / time_elapsed
            self.fps_frame_count = 0
            self.fps_last_update = current_time
        
        # Format PTS for display (convert raw PTS to human readable)
        pts_seconds = self.current_pts_raw * self.time_base
        hours = int(pts_seconds // 3600)
        minutes = int((pts_seconds % 3600) // 60)
        seconds = pts_seconds % 60
        
        if hours > 0:
            pts_str = f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
        else:
            pts_str = f"{minutes:02d}:{seconds:05.2f}"
        
        # Update info
        buffer_seconds = len(self.replay_buffer) / 30.0  # Assuming 30fps
        
        # Update save replay button text with counter
        self.save_replay_btn.setText(f"💾 Save Last {buffer_seconds:.1f}s")
        
        self.info_label.setText(
            f"Video ID: {self.video_id} | "
            f"FPS: {self.current_fps:.1f} | "
            f"PTS: {pts_str} (raw: {self.current_pts_raw}) | "
            f"Delay: {actual_delay:.0f}ms | "
            f"Cache: {len(self.bbox_cache)}"
        )
    
    def capture_replay_frame(self):
        """Capture current frame (video + bboxes only) for replay"""
        # Get the actual size of the displayed pixmap (not the label size)
        if self.video_label.pixmap() is None or self.video_label.pixmap().isNull():
            return
        
        displayed_pixmap = self.video_label.pixmap()
        width = displayed_pixmap.width()
        height = displayed_pixmap.height()
        
        # Create a pixmap with exact video dimensions
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.black)
        
        painter = QPainter(pixmap)
        
        # Draw video at exact size
        painter.drawPixmap(0, 0, displayed_pixmap)
        
        # Draw bbox overlay on top (scale overlay drawing to video size)
        painter.save()
        # Scale factor if overlay has different dimensions
        scale_x = width / self.bbox_overlay.width() if self.bbox_overlay.width() > 0 else 1.0
        scale_y = height / self.bbox_overlay.height() if self.bbox_overlay.height() > 0 else 1.0
        painter.scale(scale_x, scale_y)
        self.bbox_overlay.render(painter)
        painter.restore()
        
        painter.end()
        
        # Lock resolution on first frame
        if self.replay_resolution is None:
            self.replay_resolution = pixmap.size()
            print(f"[Replay] Locked resolution: {self.replay_resolution.width()}x{self.replay_resolution.height()}")
        
        # Scale to locked resolution if needed
        if pixmap.size() != self.replay_resolution:
            pixmap = pixmap.scaled(self.replay_resolution, Qt.AspectRatioMode.IgnoreAspectRatio, 
                                   Qt.TransformationMode.SmoothTransformation)
        
        # Convert to JPEG bytes
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        pixmap.save(buffer, "JPEG", quality=85)
        jpeg_data = buffer.data()
        buffer.close()
        
        # Add to replay buffer
        self.replay_buffer.append(bytes(jpeg_data))
    
    def save_replay(self):
        """Save replay buffer to video file in background"""
        if len(self.replay_buffer) == 0:
            QMessageBox.warning(self, "No Replay Data", "Replay buffer is empty!")
            return
        
        # Check if already saving
        if self.save_thread is not None and self.save_thread.isRunning():
            QMessageBox.information(self, "Save in Progress", "Already saving a replay, please wait...")
            return
        
        # Ask user where to save
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Replay",
            f"replay_video{self.video_id}_{int(time.time())}.mp4",
            "Video Files (*.mp4)"
        )
        
        if not file_path:
            return
        
        # Start background save thread
        self.save_thread = ReplaySaveThread(self.replay_buffer, file_path, self.replay_resolution)
        self.save_thread.progress.connect(self.on_save_progress)
        self.save_thread.finished.connect(self.on_save_finished)
        self.save_thread.start()
        
        # Disable button while saving
        self.save_replay_btn.setEnabled(False)
        print(f"[Replay] Started saving {len(self.replay_buffer)} frames to {file_path}...")
    
    def on_save_progress(self, current, total):
        """Update button with save progress"""
        percent = (current / total) * 100
        self.save_replay_btn.setText(f"💾 Saving... {percent:.0f}%")
    
    def on_save_finished(self, success, message):
        """Handle save completion"""
        buffer_seconds = len(self.replay_buffer) / 30.0
        self.save_replay_btn.setText(f"💾 Save Last {buffer_seconds:.1f}s")
        self.save_replay_btn.setEnabled(True)
        
        if success:
            print(f"[Replay] {message}")
            QMessageBox.information(self, "Replay Saved", message)
        else:
            print(f"[Replay] Error: {message}")
            QMessageBox.critical(self, "Save Error", message)
    
    def toggle_bboxes(self, checked):
        """Toggle bbox visibility"""
        self.bbox_overlay.toggle_visibility(not checked)
        self.bbox_toggle.setText("Show BBoxes" if checked else "Hide BBoxes")
    
    def stop_watching(self):
        """Stop watching this video stream"""
        self.cleanup()
        self.closed.emit(self.video_id)
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'stream_thread'):
            self.stream_thread.stop()
        if hasattr(self, 'display_timer'):
            self.display_timer.stop()
        if hasattr(self, 'fetch_timer'):
            self.fetch_timer.stop()
        if hasattr(self, 'save_thread') and self.save_thread is not None:
            self.save_thread.wait()  # Wait for save to complete
    
    def closeEvent(self, event):
        """Cleanup on close"""
        self.cleanup()
        event.accept()