# Video Stream Management with WebSocket

Real-time video streaming and bounding box visualization system using WebSocket for push-based updates.

## Architecture Overview

```
┌─────────────────┐
│   MP4 Upload    │
│    (FastAPI)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stream Manager │ ──UDP Multicast──┐
│    (FFmpeg)     │                  │
└────────┬────────┘                  │
         │                           │
         │                           ▼
         │                  ┌─────────────────┐
         │                  │   Detection     │
         │                  │    Service      │
         │                  │  (YourModel)    │
         │                  └────────┬────────┘
         │                           │
         │                           │ POST /bboxes
         ▼                           ▼
┌──────────────────────────────────────┐
│         Backend (FastAPI)            │
│  - REST API (videos, streams)        │
│  - WebSocket Manager                 │
│  - BBox Storage (in-memory)          │
└───────┬──────────────────────────┬───┘
        │                          │
        │ WebSocket (real-time)    │ UDP (video)
        │                          │
        ▼                          ▼
┌──────────────────────────────────────┐
│      Frontend (PyQt6)                │
│  - Video Player (PyAV)               │
│  - WebSocket Client                  │
│  - BBox Overlay Rendering            │
│  - Instant Replay                    │
└──────────────────────────────────────┘
```

## Key Features

### WebSocket-Based Real-Time Updates
- **Push-based**: Server pushes bboxes to clients instantly when detection service sends them
- **No polling**: Eliminates the need for periodic HTTP requests
- **Low latency**: Bboxes arrive at clients within milliseconds
- **Scalable**: Multiple clients can watch the same stream without additional backend load

### Synchronization Strategy
- **PTS-based**: Uses raw presentation timestamps from video stream
- **Buffer delay**: Small configurable delay (200ms default) for smooth playback
- **Tolerance matching**: Finds bboxes within tolerance window of current PTS

## Setup

### Backend

1. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Start the server:
```bash
python main.py
```

Server runs on `http://localhost:8702`

### Frontend

1. Install dependencies:
```bash
cd frontend
pip install -r requirements.txt
```

2. Start the application:
```bash
python main.py http://localhost:8702
```

### Detection Service

The detection service reads the UDP stream, runs inference, and sends bboxes to the backend:

```bash
python detection_service_example.py http://localhost:8702 1 udp://239.255.0.1:20001
```

Replace the mock detection with your actual model:
```python
def mock_detect_objects(self, frame, frame_number: int) -> List[Dict]:
    # Your model inference here
    results = your_model.predict(frame)
    
    # Convert to bbox format
    bboxes = []
    for detection in results:
        x1, y1, x2, y2 = detection.box
        top_left_idx = y1 * frame_width + x1
        bottom_right_idx = y2 * frame_width + x2
        
        bboxes.append({
            "top_left_corner": top_left_idx,
            "bottom_right_corner": bottom_right_idx,
            "class_name": detection.class_name,
            "confidence": detection.confidence
        })
    
    return bboxes
```

## Usage Flow

1. **Upload Video**: Use Management tab to upload an MP4 file
2. **Start Stream**: Click "Start Stream" to begin UDP multicast streaming
3. **Run Detection**: Start detection service pointing to the stream URL
4. **Watch Stream**: In Viewer tab, select stream and click "Start Stream"
5. **Real-time Visualization**: Bboxes appear automatically via WebSocket

## API Endpoints

### REST API

- `POST /videos/upload` - Upload video file
- `GET /videos/` - List all videos
- `POST /streams/start` - Start streaming a video
- `POST /streams/stop/{video_id}` - Stop stream
- `POST /bboxes/` - Add bboxes (detection service calls this)
- `GET /bboxes/{video_id}/range` - Get bboxes in PTS range (fallback)

### WebSocket

- `WS /ws/{video_id}` - Real-time bbox updates

**WebSocket Message Format**:
```json
{
  "type": "bboxes",
  "video_id": 1,
  "pts": 450000,
  "bboxes": [
    {
      "pts": 450000,
      "top_left_corner": 153600,
      "bottom_right_corner": 345600,
      "class_name": "person",
      "confidence": 0.95,
      "absolute_timestamp_ms": 1234567890
    }
  ],
  "stream_start_time_ms": 1234567000,
  "timestamp": 1234567895
}
```

## Configuration

### Video Player Settings

In `video_player.py`, you can adjust:

```python
player = VideoPlayerWidget(
    video_id=1,
    stream_url="udp://239.255.0.1:20001",
    stream_start_time_ms=1234567000,
    backend_url="http://localhost:8702",
    replay_duration_seconds=30.0,  # Instant replay duration
    buffer_delay_ms=200            # Display buffer delay
)
```

### BBox Cache Settings

```python
self.bbox_cache_max_age_seconds = 5.0  # Keep bboxes in cache for 5 seconds
```