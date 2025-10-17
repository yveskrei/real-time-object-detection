from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget
from PyQt6.QtCore import Qt
import sys
import argparse

from api_client import APIClient
from widgets.management_tab import ManagementTab
from widgets.viewer_tab import ViewerTab


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self, backend_url: str):
        super().__init__()
        self.setWindowTitle("Video Stream Management")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize API client with backend URL
        self.api_client = APIClient(base_url=backend_url)
        
        # Create tab widget with pointer cursor on tab bar
        tabs = QTabWidget()
        tabs.tabBar().setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Add tabs
        self.management_tab = ManagementTab(self.api_client)
        self.viewer_tab = ViewerTab(self.api_client)
        
        tabs.addTab(self.management_tab, "📁 Management")
        tabs.addTab(self.viewer_tab, "📺 Stream Viewer")
        
        self.setCentralWidget(tabs)


def main():
    parser = argparse.ArgumentParser(description='Video Stream Management')
    parser.add_argument('backend_url', type=str, help='Backend API URL (e.g., http://localhost:8702)')
    
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    window = MainWindow(args.backend_url)
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()