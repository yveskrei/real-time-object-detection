from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget
from PyQt6.QtCore import Qt
import sys
import argparse

from api_client import APIClient
from widgets.management_tab import ManagementTab
from widgets.viewer_tab import ViewerTab

class MainWindow(QMainWindow):
    """Main application window with WebSocket support"""
    
    def __init__(self, backend_url: str):
        super().__init__()
        self.setWindowTitle("Live Video Stream Management")
        self.setGeometry(100, 100, 1400, 900)
        
        # Store backend URL for WebSocket connections
        self.backend_url = backend_url
        
        # Initialize API client
        self.api_client = APIClient(base_url=backend_url)
        
        # Create tab widget
        tabs = QTabWidget()
        tabs.tabBar().setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Add tabs
        self.management_tab = ManagementTab(self.api_client, self.backend_url)
        self.viewer_tab = ViewerTab(self.api_client, self.backend_url)
        
        tabs.addTab(self.management_tab, "📁 Management")
        tabs.addTab(self.viewer_tab, "📺 Stream Viewer")
        
        self.setCentralWidget(tabs)

    def closeEvent(self, event):
        """
        Handle window close event to clean up all resources
        and prevent segmentation faults.
        """
        
        # Stop timers in management tab
        self.management_tab.refresh_timer.stop()
        
        # Clean up active player in viewer tab
        self.viewer_tab.cleanup_active_player()
        
        event.accept()

def main():
    parser = argparse.ArgumentParser(description='Video Stream Management Application')
    parser.add_argument('backend_url', type=str, help='Backend API URL (e.g., http://localhost:8702)')
    
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow(args.backend_url)
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()