import subprocess
import signal
import time
from fastapi import HTTPException
from storage import storage

class StreamManager:
    """Handles FFmpeg streaming processes"""
    
    @staticmethod
    def start_stream(video_id: int, output_format: str = "mpegts", resolution: str = None) -> dict:
        """Start FFmpeg stream for a video"""
        
        if video_id not in storage.videos:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        
        if video_id in storage.active_streams:
            raise HTTPException(status_code=400, detail=f"Stream {video_id} already active")
        
        video_data = storage.videos[video_id]
        file_path = video_data["file_path"]
        port = 20000 + video_id
        
        # Record stream start time (global) - milliseconds since epoch
        stream_start_time = int(time.time() * 1000)
        
        # Build FFmpeg command with metadata
        cmd = [
            "ffmpeg",
            "-re",  # Read input at native frame rate
            "-stream_loop", "-1",  # Loop indefinitely
            "-i", file_path,
        ]
        
        # Add resolution scaling if specified
        if resolution:
            cmd.extend(["-s", resolution])
        
        # Embed stream start time in metadata
        cmd.extend([
            "-metadata", f"stream_start_time={stream_start_time}",
            "-metadata", f"video_id={video_id}",
        ])
        
        # Output format - UDP allows multiple clients and reconnections
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-f", output_format,
            f"udp://127.0.0.1:{port}"
        ])
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE
            )
            
            # Store both process and metadata
            storage.active_streams[video_id] = {
                'process': process,
                'start_time_ms': stream_start_time
            }
            storage.videos[video_id]["is_streaming"] = True
            
            return {
                "video_id": video_id,
                "status": "streaming",
                "stream_url": f"udp://127.0.0.1:{port}",
                "stream_start_time_ms": stream_start_time,
                "vlc_command": f"vlc udp://@127.0.0.1:{port}",
                "pid": process.pid
            }
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to start stream: {str(e)}")
    
    @staticmethod
    def stop_stream(video_id: int) -> dict:
        """Stop FFmpeg stream for a video"""
        
        if video_id not in storage.active_streams:
            raise HTTPException(status_code=404, detail=f"No active stream for video {video_id}")
        
        stream_data = storage.active_streams[video_id]
        process = stream_data['process']
        
        # Gracefully terminate FFmpeg
        try:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        
        del storage.active_streams[video_id]
        storage.videos[video_id]["is_streaming"] = False
        
        return {
            "video_id": video_id,
            "status": "stopped"
        }
    
    @staticmethod
    def get_stream_status(video_id: int) -> dict:
        """Get stream status"""
        
        if video_id not in storage.videos:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        
        is_active = video_id in storage.active_streams
        port = 20000 + video_id
        
        result = {
            "video_id": video_id,
            "is_streaming": is_active
        }
        
        if is_active:
            stream_data = storage.active_streams[video_id]
            process = stream_data['process']
            
            # Check if process is actually running
            if process.poll() is not None:
                # Process died, clean up
                del storage.active_streams[video_id]
                storage.videos[video_id]["is_streaming"] = False
                result["is_streaming"] = False
                result["error"] = "Stream process died unexpectedly"
            else:
                result["stream_url"] = f"udp://127.0.0.1:{port}"
                result["stream_start_time_ms"] = stream_data['start_time_ms']
                result["vlc_command"] = f"vlc udp://@127.0.0.1:{port}"
                result["pid"] = process.pid
        
        return result