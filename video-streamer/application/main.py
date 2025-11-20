from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget, QInputDialog, QMessageBox
from PyQt6.QtCore import Qt
import sys
import requests

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

def get_backend_url():
    while True:
        url, ok = QInputDialog.getText(None, "Enter Backend URL", "Backend API URL (e.g., http://localhost:8702):")
        if not ok:
            sys.exit(0)
        url = url.strip().rstrip('/')
        if not url:
            continue
        try:
            resp = requests.get(f"{url}/health", timeout=3)
            if resp.status_code == 200:
                return url
            else:
                QMessageBox.warning(None, "Connection Failed", f"Health check failed with status code {resp.status_code}. Please enter a valid backend URL.")
        except Exception as e:
            QMessageBox.warning(None, "Connection Failed", f"Could not connect to backend:\n{e}\nPlease enter a valid backend URL.")

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    backend_url = get_backend_url()
    
    window = MainWindow(backend_url)
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()