from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QFileDialog, QMessageBox, QLineEdit, QLabel,
    QTextEdit, QHeaderView
)
from PyQt6.QtCore import Qt, QTimer
from pathlib import Path


class ManagementTab(QWidget):
    """Tab for managing videos and streams"""
    
    def __init__(self, api_client, parent=None):
        super().__init__(parent)
        self.api_client = api_client
        self.setup_ui()
        self.refresh_videos()
        
        # Auto-refresh every 5 seconds
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_videos)
        self.refresh_timer.start(5000)
    
    def setup_ui(self):
        """Setup the management UI"""
        layout = QVBoxLayout(self)
        
        # Upload section
        upload_layout = QHBoxLayout()
        upload_layout.addWidget(QLabel("Upload Video:"))
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Video name")
        upload_layout.addWidget(self.name_input)
        
        upload_btn = QPushButton("Upload Video")
        upload_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        upload_btn.clicked.connect(self.upload_video)
        upload_layout.addWidget(upload_btn)
        
        # Add refresh button to top right
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        refresh_btn.clicked.connect(self.refresh_videos)
        upload_layout.addWidget(refresh_btn)
        
        layout.addLayout(upload_layout)
        
        # Video table
        columns = [
            "ID", "Name", "Status", "Actions"
        ]
        self.table = QTableWidget()
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        # Disable selection and editing
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        
        # Make rows taller to fit buttons
        self.table.verticalHeader().setDefaultSectionSize(50)
        
        layout.addWidget(self.table)
    
    def upload_video(self):
        """Handle video upload"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if not file_path:
            return
        
        name = self.name_input.text() or Path(file_path).stem

        
        try:
            result = self.api_client.upload_video(file_path, name)
            QMessageBox.information(self, "Success", f"Video uploaded! ID: {result['id']}")
            self.refresh_videos()
            self.name_input.clear()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to upload video:\n{str(e)}")
    
    def refresh_videos(self):
        """Refresh video list"""
        try:
            videos = self.api_client.list_videos()
            self.table.setRowCount(len(videos))
            
            for row, video in enumerate(videos):
                # ID - centered
                id_item = QTableWidgetItem(str(video['id']))
                id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, 0, id_item)
                
                # Name - centered
                name_item = QTableWidgetItem(video['name'])
                name_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, 1, name_item)

                # Status - centered with color
                status = "Streaming" if video['is_streaming'] else "Stopped"
                status_item = QTableWidgetItem(status)
                status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                status_item.setForeground(Qt.GlobalColor.green if video['is_streaming'] else Qt.GlobalColor.red)
                self.table.setItem(row, 2, status_item)
                
                # Actions - All buttons in one column (centered in cell)
                actions_widget = QWidget()
                actions_layout = QHBoxLayout(actions_widget)
                actions_layout.setContentsMargins(4, 4, 4, 4)
                actions_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                actions_layout.setSpacing(8)
                
                # Start/Stop Stream button
                if video['is_streaming']:
                    stop_btn = QPushButton("Stop Stream")
                    stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    stop_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #e67e22;
                            color: white;
                            padding: 6px 12px;
                            border: none;
                            border-radius: 4px;
                            font-weight: bold;
                        }
                        QPushButton:hover {
                            background-color: #d35400;
                        }
                    """)
                    stop_btn.clicked.connect(lambda checked, vid=video['id']: self.stop_stream(vid))
                    actions_layout.addWidget(stop_btn)
                else:
                    start_btn = QPushButton("Start Stream")
                    start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                    start_btn.setStyleSheet("""
                        QPushButton {
                            background-color: #27ae60;
                            color: white;
                            padding: 6px 12px;
                            border: none;
                            border-radius: 4px;
                            font-weight: bold;
                        }
                        QPushButton:hover {
                            background-color: #229954;
                        }
                    """)
                    start_btn.clicked.connect(lambda checked, vid=video['id']: self.start_stream(vid))
                    actions_layout.addWidget(start_btn)
                
                # Delete button
                delete_btn = QPushButton("Delete")
                delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                delete_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #e74c3c;
                        color: white;
                        padding: 6px 12px;
                        border: none;
                        border-radius: 4px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #c0392b;
                    }
                """)
                delete_btn.clicked.connect(lambda checked, vid=video['id']: self.delete_video(vid))
                actions_layout.addWidget(delete_btn)
                
                self.table.setCellWidget(row, 3, actions_widget)
        
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to refresh videos:\n{str(e)}")
    
    def delete_video(self, video_id: int):
        """Delete a video"""
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete video {video_id}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.api_client.delete_video(video_id)
                QMessageBox.information(self, "Success", "Video deleted!")
                self.refresh_videos()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete video:\n{str(e)}")
    
    def start_stream(self, video_id: int):
        """Start streaming a video"""
        try:
            result = self.api_client.start_stream(video_id)
            QMessageBox.information(
                self,
                "Stream Started",
                f"Stream started!\nURL: {result['stream_url']}\nVLC: {result.get('vlc_command', '')}"
            )
            self.refresh_videos()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start stream:\n{str(e)}")
    
    def stop_stream(self, video_id: int):
        """Stop streaming a video"""
        try:
            self.api_client.stop_stream(video_id)
            QMessageBox.information(self, "Success", "Stream stopped!")
            self.refresh_videos()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to stop stream:\n{str(e)}")