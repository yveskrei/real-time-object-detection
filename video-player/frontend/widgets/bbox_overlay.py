from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QPainter, QPen, QColor, QFont
from typing import List

class BBoxOverlay(QWidget):
    """Transparent overlay widget for drawing bounding boxes on top of video"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.bboxes: List[dict] = []
        self.show_bboxes = True
        self.frame_width = 0  # Will be set by video player
        self.frame_height = 0
        self.colors = {
            "person": QColor(255, 0, 0),
            "car": QColor(0, 255, 0),
            "truck": QColor(0, 0, 255),
            "dog": QColor(255, 255, 0),
            "cat": QColor(255, 0, 255),
            "default": QColor(255, 255, 0)
        }
    
    def set_frame_dimensions(self, width: int, height: int):
        """Set the frame dimensions for coordinate conversion"""
        self.frame_width = width
        self.frame_height = height
    
    def index_to_coords(self, index: int) -> tuple:
        """Convert 1D pixel index to 2D (x, y) coordinates"""
        if self.frame_width == 0:
            return (0, 0)
        y = index // self.frame_width
        x = index % self.frame_width
        return (x, y)
    
    def set_bboxes(self, bboxes: List[dict]):
        """Update bounding boxes to display"""
        self.bboxes = bboxes if bboxes else []
        self.update()
    
    def toggle_visibility(self, visible: bool):
        """Show or hide bounding boxes"""
        self.show_bboxes = visible
        self.update()
    
    def paintEvent(self, event):
        """Draw bounding boxes with QPainter"""
        if not self.show_bboxes or not self.bboxes or self.frame_width == 0:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        for bbox in self.bboxes:
            # Get color based on class name
            class_name = bbox.get("class_name", "").lower()
            color = self.colors.get(class_name, self.colors["default"])
            
            # Convert 1D indices to 2D coordinates
            top_left_idx = int(bbox["top_left_corner"])
            bottom_right_idx = int(bbox["bottom_right_corner"])
            
            x1, y1 = self.index_to_coords(top_left_idx)
            x2, y2 = self.index_to_coords(bottom_right_idx)
            
            # Calculate width and height
            w = x2 - x1
            h = y2 - y1
            
            # Draw rectangle
            pen = QPen(color, 3)
            painter.setPen(pen)
            painter.drawRect(QRect(x1, y1, w, h))
            
            # Draw label with class name and confidence
            confidence = bbox.get("confidence", 0)
            label = f"{bbox.get('class_name', 'Unknown')} {confidence:.2f}"
            
            # Background for text
            font = QFont("Arial", 10, QFont.Weight.Bold)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            text_width = metrics.horizontalAdvance(label)
            text_height = metrics.height()
            
            # Draw text background
            painter.fillRect(x1, y1 - text_height - 4, text_width + 8, text_height + 4, color)
            
            # Draw text
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(x1 + 4, y1 - 6, label)