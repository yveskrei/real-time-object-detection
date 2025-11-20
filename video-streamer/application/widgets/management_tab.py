from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QTableWidgetItem, QFileDialog, QMessageBox, QLineEdit, QLabel,
    QHeaderView, QInputDialog
)
from PyQt6.QtCore import Qt, QTimer
from pathlib import Path
from datetime import datetime
import pytz
from urllib.parse import urlparse


class ManagementTab(QWidget):
    """Tab for managing videos and streams"""
    
    def __init__(self, api_client, backend_url: str, parent=None):
        super().__init__(parent)
        self.api_client = api_client
        self.backend_url = backend_url.rstrip('/') # Store backend URL
        
        # Parse backend host for UDP URL
        parsed_url = urlparse(self.backend_url)
        self.backend_host = parsed_url.hostname
        if not self.backend_host:
            # Failsafe if no scheme (e.g. just "localhost:8000")
            self.backend_host = self.backend_url.split(':')[0]
            
        self.setup_ui()
        self.refresh_videos()
        
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_videos)
        self.refresh_timer.start(5000)
    
    def setup_ui(self):
        """Setup the management UI"""
        layout = QVBoxLayout(self)
        
        upload_layout = QHBoxLayout()
        
        upload_btn = QPushButton("Upload Video")
        upload_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        upload_btn.clicked.connect(self.upload_video)
        upload_layout.addWidget(upload_btn)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        refresh_btn.clicked.connect(self.refresh_videos)
        upload_layout.addWidget(refresh_btn)
        
        upload_layout.addStretch() # Push buttons to the left
        
        layout.addLayout(upload_layout)
        
        columns = ["ID", "Name", "Status", "Actions"]
        self.table = QTableWidget()
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        
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

        # Pop up dialog to ask for a name
        text, ok = QInputDialog.getText(
            self, 
            "Video Name", 
            "Enter a name for the video (optional):", 
            QLineEdit.EchoMode.Normal, 
            "" # Initial text is blank
        )
        
        if not ok:
            # User cancelled the name dialog
            return

        # Use entered name, or file stem if name is blank
        name = text.strip() or Path(file_path).stem
        
        try:
            result = self.api_client.upload_video(file_path, name)
            QMessageBox.information(self, "Success", f"Video uploaded! ID: {result['id']}")
            self.refresh_videos()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to upload video:\n{str(e)}")
    
    def refresh_videos(self):
        """Refresh video list"""
        try:
            videos = self.api_client.list_videos()
            self.table.setRowCount(len(videos))
            
            for row, video in enumerate(videos):
                id_item = QTableWidgetItem(str(video['id']))
                id_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, 0, id_item)
                
                name_item = QTableWidgetItem(video['name'])
                name_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, 1, name_item)
                
                status = "Streaming" if video['is_streaming'] else "Stopped"
                status_item = QTableWidgetItem(status)
                status_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                status_item.setForeground(Qt.GlobalColor.green if video['is_streaming'] else Qt.GlobalColor.red)
                self.table.setItem(row, 2, status_item)
                
                actions_widget = QWidget()
                actions_layout = QHBoxLayout(actions_widget)
                actions_layout.setContentsMargins(4, 4, 4, 4)
                actions_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                actions_layout.setSpacing(8)
                
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
                        QPushButton:disabled {
                            background-color: rgba(230, 126, 34, 128);
                            color: rgba(255, 255, 255, 128);
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
                        QPushButton:disabled {
                            background-color: rgba(39, 174, 96, 128);
                            color: rgba(255, 255, 255, 128);
                        }
                    """)
                    start_btn.clicked.connect(lambda checked, vid=video['id']: self.start_stream(vid))
                    actions_layout.addWidget(start_btn)
                
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
                    QPushButton:disabled {
                        background-color: rgba(231, 76, 60, 128);
                        color: rgba(255, 255, 255, 128);
                    }
                """)
                delete_btn.clicked.connect(lambda checked, vid=video['id']: self.delete_video(vid))
                
                # Disable delete button if streaming
                if video['is_streaming']:
                    delete_btn.setEnabled(False)
                    delete_btn.setToolTip("Cannot delete a running stream")
                
                actions_layout.addWidget(delete_btn)
                
                # Add Info Button
                info_btn = QPushButton("Info")
                info_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                info_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #3498db;
                        color: white;
                        padding: 6px 12px;
                        border: none;
                        border-radius: 4px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #2980b9;
                    }
                    QPushButton:disabled {
                        background-color: rgba(52, 152, 219, 128);
                        color: rgba(255, 255, 255, 128);
                    }
                """)
                info_btn.clicked.connect(lambda checked, vid=video['id']: self.show_stream_info(vid))
                
                # Disable info button if not streaming
                if not video['is_streaming']:
                    info_btn.setEnabled(False)
                    info_btn.setToolTip("Stream is not active")

                actions_layout.addWidget(info_btn)
                
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
    
    def show_stream_info(self, video_id: int):
        """Show detailed information about a stream"""
        try:
            status = self.api_client.get_stream_status(video_id)
            
            if not status.get('is_streaming'):
                QMessageBox.information(self, "Stream Information", f"Stream {video_id} is currently stopped.")
                return

            message = f"Stream ID: {video_id}\n\n"
            
            # Format Start Time
            try:
                start_time_ms = status.get('stream_start_time_ms')
                if start_time_ms:
                    # Convert from ms timestamp to datetime object (UTC)
                    start_time_utc = datetime.fromtimestamp(start_time_ms / 1000.0, tz=pytz.utc)
                    
                    # Convert to Asia/Jerusalem timezone
                    israel_tz = pytz.timezone("Asia/Jerusalem")
                    start_time_local = start_time_utc.astimezone(israel_tz)
                    
                    time_str = start_time_local.strftime('%Y-%m-%d %H:%M:%S %Z')
                    message += f"Stream Started: {time_str}\n"
                else:
                    message += "Stream Started: Unknown\n"
            except Exception as time_e:
                message += f"Stream Started: Error parsing time ({time_e})\n"
            
            # UDP Info
            udp_info = status.get('udp', {})
            udp_port = udp_info.get('port')
            
            if udp_port and self.backend_host:
                udp_url = f"udp://{self.backend_host}:{udp_port}"
                message += f"UDP URL: {udp_url}\n"
            elif udp_port:
                message += f"UDP Port: {udp_port} (Hostname unknown)\n"
            else:
                message += "UDP URL: N/A\n"

            # DASH Info
            dash_info = status.get('dash', {})
            manifest_path = dash_info.get('manifest_url', 'N/A')
            
            if manifest_path.startswith('/') and manifest_path != 'N/A':
                dash_url = f"{self.backend_url}{manifest_path}"
            else:
                dash_url = manifest_path
                
            message += f"DASH Manifest URL: {dash_url}\n"
            
            QMessageBox.information(self, "Stream Information", message)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to get stream status:\n{str(e)}")

    def start_stream(self, video_id: int):
        """Start streaming a video"""
        try:
            self.api_client.start_stream(video_id)
            
            QMessageBox.information(self, "Stream Started", f"Stream {video_id} started!")
            self.refresh_videos()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start stream:\n{str(e)}")
    
    def stop_stream(self, video_id: int):
        """Stop streaming a video"""
        try:
            self.api_client.stop_stream(video_id)
            QMessageBox.information(self, "Stream Stopped", f"Stream {video_id} stopped!")
            self.refresh_videos()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to stop stream:\n{str(e)}")