from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox,
    QLabel, QMessageBox, QGridLayout, QDoubleSpinBox, QSpinBox,
    QFrame
)
from PyQt6.QtCore import QTimer, Qt
from widgets.video_player import VideoPlayerWidget
from typing import Optional

class ViewerTab(QWidget):
    """Tab for viewing one active stream at a time with WebSocket support"""
    
    def __init__(self, api_client, backend_url: str, parent=None):
        super().__init__(parent)
        self.api_client = api_client
        self.backend_url = backend_url  # Store backend URL for WebSocket connections
        self.active_player: Optional[VideoPlayerWidget] = None
        self.setup_ui()
        
        # Auto-refresh active streams every 5 seconds
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_streams)
        self.refresh_timer.start(5000)
    
    def setup_ui(self):
        """Setup the viewer UI"""
        layout = QVBoxLayout(self)
        
        # Top controls row
        controls = QHBoxLayout()
        
        # Left side - Stream selection
        controls.addWidget(QLabel("Select Stream to Watch:"))
        
        self.stream_combo = QComboBox()
        self.stream_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stream_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        controls.addWidget(self.stream_combo)
        
        self.add_stream_btn = QPushButton("Start Stream")
        self.add_stream_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_stream_btn.clicked.connect(self.add_stream_player)
        controls.addWidget(self.add_stream_btn)
        
        refresh_btn = QPushButton("Refresh Streams")
        refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        refresh_btn.clicked.connect(self.refresh_streams)
        controls.addWidget(refresh_btn)
        
        # Spacer to push right-side controls to the right
        controls.addStretch()
        
        # Right side - Confidence control, retention control, and hide bboxes button
        controls.addWidget(QLabel("Min Confidence:"))
        
        self.confidence_spinner = QDoubleSpinBox()
        self.confidence_spinner.setRange(0.0, 1.0)
        self.confidence_spinner.setSingleStep(0.05)
        self.confidence_spinner.setValue(0.0)
        self.confidence_spinner.setDecimals(2)
        self.confidence_spinner.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows)
        self.confidence_spinner.setCursor(Qt.CursorShape.PointingHandCursor)
        self.confidence_spinner.setFixedWidth(80)
        self.confidence_spinner.valueChanged.connect(self.on_confidence_changed)
        self.confidence_spinner.setToolTip("Set minimum confidence threshold for displaying bounding boxes")
        controls.addWidget(self.confidence_spinner)
        
        # BBOX Retention control
        controls.addWidget(QLabel("BBOX Retention:"))
        
        self.retention_spinner = QSpinBox()
        self.retention_spinner.setRange(1, 30)
        self.retention_spinner.setSingleStep(1)
        self.retention_spinner.setValue(1)
        self.retention_spinner.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self.retention_spinner.setCursor(Qt.CursorShape.PointingHandCursor)
        self.retention_spinner.setFixedWidth(80)
        self.retention_spinner.valueChanged.connect(self.on_retention_changed)
        self.retention_spinner.setToolTip("Number of frames to retain bounding boxes on screen")
        controls.addWidget(self.retention_spinner)
        
        self.hide_bboxes_btn = QPushButton("Hide BBoxes")
        self.hide_bboxes_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.hide_bboxes_btn.setCheckable(True)
        self.hide_bboxes_btn.clicked.connect(self.toggle_all_bboxes)
        self.hide_bboxes_btn.setToolTip("Toggle visibility of all bounding boxes")
        controls.addWidget(self.hide_bboxes_btn)
        
        layout.addLayout(controls)
        
        # Container for the *single* video player
        self.player_container = QWidget()
        self.player_container_layout = QVBoxLayout(self.player_container)
        self.player_container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add a styled frame to hold the player
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setLayout(self.player_container_layout)
        
        # Add the frame with a stretch factor of 1 to fill vertical space
        layout.addWidget(frame, 1)
        
        # Removed layout.addStretch() which was causing the issue
        
        # Initial refresh
        self.refresh_streams()
    
    def on_confidence_changed(self, value: float):
        """Update confidence threshold for the active player"""
        if self.active_player and hasattr(self.active_player, 'bbox_overlay'):
            self.active_player.bbox_overlay.set_min_confidence(value)
    
    def on_retention_changed(self, value: int):
        """Update bbox retention for the active player"""
        if self.active_player and hasattr(self.active_player, 'bbox_overlay'):
            self.active_player.bbox_overlay.set_bbox_retention(value)
    
    def toggle_all_bboxes(self, checked: bool):
        """Toggle bbox visibility for the active player"""
        if self.active_player and hasattr(self.active_player, 'bbox_overlay'):
            self.active_player.bbox_overlay.toggle_visibility(not checked)
        
        # Update button text
        self.hide_bboxes_btn.setText("Show BBoxes" if checked else "Hide BBoxes")
    
    def refresh_streams(self):
        """Refresh list of active streams"""
        try:
            videos = self.api_client.list_videos()
            active_streams = [v for v in videos if v['is_streaming']]
            
            # Save current selection
            current_id = self.stream_combo.currentData()
            
            self.stream_combo.clear()
            found_index = -1
            for i, stream in enumerate(active_streams):
                self.stream_combo.addItem(
                    f"ID {stream['id']}: {stream['name']}",
                    stream['id']
                )
                if stream['id'] == current_id:
                    found_index = i
            
            # Restore selection if it still exists
            if found_index != -1:
                self.stream_combo.setCurrentIndex(found_index)
                
        except Exception as e:
            pass  # Silently fail during auto-refresh
    
    def add_stream_player(self):
        """Add a video player for selected stream"""
        if self.stream_combo.count() == 0:
            QMessageBox.warning(self, "No Streams", "No active streams available!")
            return
        
        video_id = self.stream_combo.currentData()
        
        if self.active_player is not None:
            QMessageBox.information(
                self, 
                "Player Active", 
                "A stream is already playing. Please stop the current stream before starting a new one."
            )
            return
        
        try:
            # Get stream info
            status = self.api_client.get_stream_status(video_id)
            if not status['is_streaming']:
                QMessageBox.warning(self, "Not Streaming", "This stream is not active!")
                return
            
            port = status['port']
            stream_start_time = status.get('stream_start_time_ms')
            
            if not stream_start_time:
                QMessageBox.critical(self, "Error", "Stream metadata missing! Restart the stream.")
                return
            
            # Create video player
            player = VideoPlayerWidget(
                video_id,
                port,
                stream_start_time,
                self.backend_url,
                replay_duration_seconds=30.0
            )
            player.closed.connect(self.remove_player)
            
            # Apply current settings to new player
            if hasattr(player, 'bbox_overlay'):
                player.bbox_overlay.set_min_confidence(self.confidence_spinner.value())
                player.bbox_overlay.set_bbox_retention(self.retention_spinner.value())
                player.bbox_overlay.toggle_visibility(not self.hide_bboxes_btn.isChecked())
            
            # Add to layout
            self.player_container_layout.addWidget(player)
            self.active_player = player
            
            # Disable controls
            self.add_stream_btn.setEnabled(False)
            self.stream_combo.setEnabled(False)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add stream player:\n{str(e)}")
    
    def remove_player(self, video_id: int):
        """Remove the video player"""
        if self.active_player and self.active_player.video_id == video_id:
            print(f"[ViewerTab] Removing player for video {video_id}")
            
            self.player_container_layout.removeWidget(self.active_player)
            self.active_player.deleteLater()
            self.active_player = None
            
            # Re-enable controls
            self.add_stream_btn.setEnabled(True)
            self.stream_combo.setEnabled(True)