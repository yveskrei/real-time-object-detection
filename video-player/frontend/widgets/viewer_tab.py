from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox,
    QLabel, QScrollArea, QMessageBox, QGridLayout
)
from PyQt6.QtCore import QTimer, Qt
from widgets.video_player import VideoPlayerWidget


class ViewerTab(QWidget):
    """Tab for viewing active streams"""
    
    def __init__(self, api_client, parent=None):
        super().__init__(parent)
        self.api_client = api_client
        self.active_players = {}
        self.setup_ui()
        
        # Auto-refresh active streams every 5 seconds
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_streams)
        self.refresh_timer.start(5000)
    
    def setup_ui(self):
        """Setup the viewer UI"""
        layout = QVBoxLayout(self)
        
        # Controls
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Select Stream to Watch:"))
        
        # Dropdown with pointer cursor and auto-adjust width
        self.stream_combo = QComboBox()
        self.stream_combo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stream_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        controls.addWidget(self.stream_combo)
        
        # Add Stream button with pointer cursor
        add_btn = QPushButton("Start Stream")
        add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_btn.clicked.connect(self.add_stream_player)
        controls.addWidget(add_btn)
        
        # Refresh button with pointer cursor
        refresh_btn = QPushButton("Refresh Streams")
        refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        refresh_btn.clicked.connect(self.refresh_streams)
        controls.addWidget(refresh_btn)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        # Scroll area for video players
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        self.players_container = QWidget()
        self.players_layout = QGridLayout(self.players_container)
        scroll.setWidget(self.players_container)
        
        layout.addWidget(scroll)
        
        # Initial refresh
        self.refresh_streams()
    
    def refresh_streams(self):
        """Refresh list of active streams"""
        try:
            videos = self.api_client.list_videos()
            active_streams = [v for v in videos if v['is_streaming']]
            
            self.stream_combo.clear()
            for stream in active_streams:
                self.stream_combo.addItem(
                    f"ID {stream['id']}: {stream['name']}",
                    stream['id']
                )
        except Exception as e:
            pass  # Silently fail during auto-refresh
    
    def add_stream_player(self):
        """Add a video player for selected stream"""
        if self.stream_combo.count() == 0:
            QMessageBox.warning(self, "No Streams", "No active streams available!")
            return
        
        video_id = self.stream_combo.currentData()
        
        if video_id in self.active_players:
            QMessageBox.information(self, "Already Added", f"Stream {video_id} is already being viewed!")
            return
        
        try:
            # Get stream info (includes stream_start_time_ms)
            status = self.api_client.get_stream_status(video_id)
            if not status['is_streaming']:
                QMessageBox.warning(self, "Not Streaming", "This stream is not active!")
                return
            
            stream_url = status['stream_url']
            stream_start_time = status.get('stream_start_time_ms')
            
            if not stream_start_time:
                QMessageBox.critical(self, "Error", "Stream metadata missing! Restart the stream.")
                return
            
            # Create video player with global stream start time
            player = VideoPlayerWidget(video_id, stream_url, stream_start_time, self.api_client)
            player.closed.connect(self.remove_player)  # Connect close signal
            
            # Add to grid (2 columns)
            row = len(self.active_players) // 2
            col = len(self.active_players) % 2
            self.players_layout.addWidget(player, row, col)
            
            self.active_players[video_id] = player
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add stream player:\n{str(e)}")
    
    def remove_player(self, video_id: int):
        """Remove a video player"""
        if video_id in self.active_players:
            player = self.active_players[video_id]
            
            # Remove from layout
            self.players_layout.removeWidget(player)
            player.deleteLater()
            
            # Remove from dictionary
            del self.active_players[video_id]
            
            # Reorganize grid layout
            self.reorganize_players()
    
    def reorganize_players(self):
        """Reorganize players in grid after removal"""
        # Get all remaining players
        players = list(self.active_players.values())
        
        # Clear layout (this doesn't delete widgets, just removes them from layout)
        for i in reversed(range(self.players_layout.count())):
            self.players_layout.itemAt(i).widget().setParent(None)
        
        # Re-add players in grid (2 columns)
        for idx, player in enumerate(players):
            row = idx // 2
            col = idx % 2
            self.players_layout.addWidget(player, row, col)