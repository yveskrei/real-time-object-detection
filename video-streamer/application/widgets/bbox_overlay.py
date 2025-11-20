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
        self.min_confidence = 0.0  # Minimum confidence threshold
        self.bbox_retention_frames = 1  # Number of frames to retain bbox (default 1)
        
        # Original video frame dimensions (from decoded frame)
        self.frame_width = 0
        self.frame_height = 0
        
        # Updated color scheme with darker, more visible colors
        self.colors = {
            "person": QColor(0, 150, 255),      # Bright blue
            "car": QColor(0, 200, 0),           # Green
            "truck": QColor(255, 100, 0),       # Orange
            "dog": QColor(200, 0, 200),         # Magenta
            "cat": QColor(255, 0, 100),         # Pink-red
            "default": QColor(0, 200, 200)      # Cyan
        }
    
    def set_frame_dimensions(self, width: int, height: int):
        """Set the original frame dimensions for coordinate conversion"""
        self.frame_width = width
        self.frame_height = height
    
    def set_min_confidence(self, confidence: float):
        """Set minimum confidence threshold for displaying bboxes"""
        self.min_confidence = max(0.0, min(1.0, confidence))
        self.update()
    
    def set_bbox_retention(self, frames: int):
        """Set number of frames to retain bboxes"""
        self.bbox_retention_frames = max(1, min(30, frames))
        self.update()
    
    def index_to_coords(self, index: int) -> tuple:
        """Convert 1D pixel index to 2D (x, y) coordinates in original frame space"""
        if self.frame_width == 0:
            return (0, 0)
        y = index // self.frame_width
        x = index % self.frame_width
        return (x, y)
    
    def scale_coords_to_widget(self, x: int, y: int) -> tuple:
        """Scale coordinates from original frame space to widget display space"""
        if self.frame_width == 0 or self.frame_height == 0:
            return (x, y)
        
        # Calculate scale factors
        scale_x = self.width() / self.frame_width
        scale_y = self.height() / self.frame_height
        
        # Scale coordinates
        scaled_x = int(x * scale_x)
        scaled_y = int(y * scale_y)
        
        return (scaled_x, scaled_y)
    
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
            # Filter by confidence
            confidence = bbox.get("confidence", 0)
            if confidence < self.min_confidence:
                continue
            
            # Get color based on class name
            class_name = bbox.get("class_name", "").lower()
            color = self.colors.get(class_name, self.colors["default"])
            
            # Convert 1D indices to 2D coordinates in original frame space
            top_left_idx = int(bbox["top_left_corner"])
            bottom_right_idx = int(bbox["bottom_right_corner"])
            
            x1_orig, y1_orig = self.index_to_coords(top_left_idx)
            x2_orig, y2_orig = self.index_to_coords(bottom_right_idx)
            
            # Scale coordinates to widget display space
            x1, y1 = self.scale_coords_to_widget(x1_orig, y1_orig)
            x2, y2 = self.scale_coords_to_widget(x2_orig, y2_orig)
            
            # Calculate width and height
            w = x2 - x1
            h = y2 - y1
            
            # Draw rectangle
            pen = QPen(color, 3)
            painter.setPen(pen)
            painter.drawRect(QRect(x1, y1, w, h))
            
            # Draw label with class name and confidence
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