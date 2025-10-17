"""Shared constants for the application"""

DEFAULT_API_URL = "http://localhost:8702"
DEFAULT_FPS = 30  # Assumed FPS for frame calculation

# BBox colors by class
BBOX_COLORS = {
    "person": (255, 0, 0),      # Red
    "car": (0, 255, 0),          # Green
    "truck": (0, 0, 255),        # Blue
    "dog": (255, 255, 0),        # Yellow
    "cat": (255, 0, 255),        # Magenta
    "default": (255, 255, 0)     # Yellow
}
