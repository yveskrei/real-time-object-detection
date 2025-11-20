from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox,
    QLabel, QMessageBox, QDoubleSpinBox, QSpinBox,
    QFrame
)
from PyQt6.QtCore import QTimer, Qt
from widgets.video_player import VideoPlayerWidget
from typing import Optional

class ViewerTab(QWidget):
    
    def __init__(self, api_client, backend_url: str, parent=None):
        super().__init__(parent)
        self.api_client = api_client
        self.backend_url = backend_url
        self.active_player: Optional[VideoPlayerWidget] = None
        self.setup_ui()
        
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_streams)
        self.refresh_timer.start(5000)
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        controls = QHBoxLayout()

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_btn.clicked.connect(self.refresh_streams)
        controls.addWidget(self.refresh_btn)
        
        self.stream_combo = QComboBox()
        self.stream_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stream_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        controls.addWidget(self.stream_combo)
        
        self.add_stream_btn = QPushButton("Start Stream")
        self.add_stream_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_stream_btn.clicked.connect(self.add_stream_player)
        controls.addWidget(self.add_stream_btn)
        
        controls.addStretch()
        
        self.confidence_spinner_label = QLabel("Min Confidence:")
        self.confidence_spinner_label.setVisible(False)
        controls.addWidget(self.confidence_spinner_label)

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
        self.confidence_spinner.setVisible(False)
        controls.addWidget(self.confidence_spinner)
        
        self.retention_spinner_label = QLabel("BBOX Retention:")
        self.retention_spinner_label.setVisible(False)
        controls.addWidget(self.retention_spinner_label)

        self.retention_spinner = QSpinBox()
        self.retention_spinner.setRange(1, 30)
        self.retention_spinner.setSingleStep(1)
        self.retention_spinner.setValue(1)
        self.retention_spinner.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self.retention_spinner.setCursor(Qt.CursorShape.PointingHandCursor)
        self.retention_spinner.setFixedWidth(80)
        self.retention_spinner.valueChanged.connect(self.on_retention_changed)
        self.retention_spinner.setToolTip("Number of frames to retain bounding boxes on screen")
        self.retention_spinner.setVisible(False)
        controls.addWidget(self.retention_spinner)
        
        self.hide_bboxes_btn = QPushButton("Hide BBoxes")
        self.hide_bboxes_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.hide_bboxes_btn.setCheckable(True)
        self.hide_bboxes_btn.clicked.connect(self.toggle_all_bboxes)
        self.hide_bboxes_btn.setToolTip("Toggle visibility of all bounding boxes")
        self.hide_bboxes_btn.setVisible(False)
        controls.addWidget(self.hide_bboxes_btn)
        
        layout.addLayout(controls)
        
        self.player_container = QWidget()
        self.player_container_layout = QVBoxLayout(self.player_container)
        self.player_container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Label to show when no streams are active
        self.no_streams_label = QLabel("No active streams available.")
        self.no_streams_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.no_streams_label.font()
        font.setPointSize(16)
        self.no_streams_label.setFont(font)
        self.no_streams_label.setStyleSheet("color: #888;")
        # Add with stretch 1, so it fills the space (or shares with player)
        self.player_container_layout.addWidget(self.no_streams_label, 1, Qt.AlignmentFlag.AlignCenter)
        self.no_streams_label.setVisible(False) # Initially hidden
        
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setLayout(self.player_container_layout)
        
        layout.addWidget(frame, 1)
        
        self.refresh_streams()

    def update_stream_availability_ui(self):
        """Updates the UI state based on stream availability and player status."""
        has_streams = self.stream_combo.count() > 0
        is_player_active = self.active_player is not None

        # Show "No streams" message only if no streams AND no player
        self.no_streams_label.setVisible(not has_streams and not is_player_active)
        
        # Enable "Start Stream" only if streams available AND no player
        self.add_stream_btn.setEnabled(has_streams and not is_player_active)
        
        # Enable combo box and refresh button only if no player
        self.stream_combo.setEnabled(has_streams and not is_player_active)
        self.refresh_btn.setEnabled(not is_player_active)
    
    def on_confidence_changed(self, value: float):
        if self.active_player and hasattr(self.active_player, 'bbox_overlay'):
            self.active_player.bbox_overlay.set_min_confidence(value)
    
    def on_retention_changed(self, value: int):
        if self.active_player and hasattr(self.active_player, 'bbox_overlay'):
            self.active_player.bbox_overlay.set_bbox_retention(value)
    
    def toggle_all_bboxes(self, checked: bool):
        if self.active_player and hasattr(self.active_player, 'bbox_overlay'):
            self.active_player.bbox_overlay.toggle_visibility(not checked)
        
        self.hide_bboxes_btn.setText("Show BBoxes" if checked else "Hide BBoxes")
    
    def refresh_streams(self):
        try:
            videos = self.api_client.list_videos()
            active_streams = [v for v in videos if v['is_streaming']]
            
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
            
            if found_index != -1:
                self.stream_combo.setCurrentIndex(found_index)
                
        except Exception:
            pass # Silently fail on refresh
        
        # Update UI state after refresh
        self.update_stream_availability_ui()
    
    def add_stream_player(self):
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
            status = self.api_client.get_stream_status(video_id)
            if not status.get('is_streaming'):
                QMessageBox.warning(self, "Not Streaming", "This stream is not active!")
                return
            
            dash_info = status.get('dash')
            if not dash_info or 'manifest_url' not in dash_info:
                QMessageBox.critical(self, "Error", "DASH manifest URL missing from stream status!")
                return
            
            stream_start_time = status.get('stream_start_time_ms')
            if not stream_start_time:
                QMessageBox.critical(self, "Error", "Stream metadata missing! Restart the stream.")
                return
            
            manifest_url = dash_info['manifest_url']
            
            player = VideoPlayerWidget(
                video_id,
                manifest_url,
                stream_start_time,
                self.backend_url,
                replay_duration_seconds=30.0
            )
            player.closed.connect(self.remove_player)
            
            if hasattr(player, 'bbox_overlay'):
                player.bbox_overlay.set_min_confidence(self.confidence_spinner.value())
                player.bbox_overlay.set_bbox_retention(self.retention_spinner.value())
                player.bbox_overlay.toggle_visibility(not self.hide_bboxes_btn.isChecked())
            
            # Add player with stretch 1 so it fills space
            self.player_container_layout.addWidget(player, 1)
            self.active_player = player
            
            # Disable controls and hide "No streams" label
            self.update_stream_availability_ui()

            # Show bbox controls
            self.confidence_spinner_label.setVisible(True)
            self.confidence_spinner.setVisible(True)
            self.retention_spinner.setVisible(True)
            self.retention_spinner_label.setVisible(True)
            self.hide_bboxes_btn.setVisible(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add stream player:\n{str(e)}")
    
    def remove_player(self, video_id: int):
        if self.active_player and self.active_player.video_id == video_id:
            player = self.active_player
            self.active_player = None
            
            self.player_container_layout.removeWidget(player)
            player.hide()
            player.setParent(None)
            player.deleteLater()
            
            # Re-enable controls and show "No streams" label if needed
            self.update_stream_availability_ui()

            # Hide bbox controls
            self.confidence_spinner_label.setVisible(False)
            self.confidence_spinner.setVisible(False)
            self.retention_spinner.setVisible(False)
            self.retention_spinner_label.setVisible(False)
            self.hide_bboxes_btn.setVisible(False)


    def cleanup_active_player(self):
        self.refresh_timer.stop()
        if self.active_player:
            self.active_player.cleanup()
            self.active_player.deleteLater()
            self.active_player = None