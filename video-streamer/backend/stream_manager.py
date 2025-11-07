import subprocess
import signal
import time
from fastapi import HTTPException
from storage import storage

class StreamManager:
    """Handles FFmpeg streaming processes with multicast support"""
    
    # Multicast base address - each video gets a unique multicast address
    MULTICAST_BASE = "239.255.0"
    
    @staticmethod
    def _get_multicast_address(video_id: int) -> str:
        """Generate multicast address for a video ID
        
        Uses video_id to create unique multicast addresses:
        - 239.255.0.1 to 239.255.0.255 for video IDs 1-255
        - 239.255.1.0 to 239.255.1.255 for video IDs 256-511, etc.
        """
        # For video_id 1-255: use 239.255.0.X
        # For video_id 256+: use 239.255.Y.X where Y = (video_id-1) // 256
        if video_id <= 255:
            return f"239.255.0.{video_id}"
        else:
            octet3 = (video_id - 1) // 256
            octet4 = (video_id - 1) % 256 + 1
            return f"239.255.{octet3}.{octet4}"
    
    @staticmethod
    def start_stream(video_id: int, output_format: str = "mpegts", resolution: str = None) -> dict:
        """Start FFmpeg stream for a video using multicast UDP
        
        Multicast allows unlimited clients to watch simultaneously without
        additional bandwidth or port conflicts.
        """
        
        if video_id not in storage.videos:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        
        if video_id in storage.active_streams:
            raise HTTPException(status_code=400, detail=f"Stream {video_id} already active")
        
        video_data = storage.videos[video_id]
        file_path = video_data["file_path"]
        port = 20000 + video_id
        multicast_addr = StreamManager._get_multicast_address(video_id)
        
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
        
        # Output format - Multicast UDP allows unlimited clients
        # Different handling for different formats
        if output_format == "rtp":
            # RTP has native multicast support
            cmd.extend([
                "-c:v", "libx264",
                "-preset", "veryfast",  # Better quality than ultrafast
                "-tune", "zerolatency",
                "-b:v", "2M",  # Set video bitrate for consistency
                "-maxrate", "2M",
                "-bufsize", "4M",  # Buffer size = 2x bitrate
                "-g", "50",  # GOP size - keyframe every 50 frames
                "-c:a", "aac",  # Explicit audio codec
                "-b:a", "128k",  # Audio bitrate
                "-f", "rtp",
                f"rtp://{multicast_addr}:{port}?ttl=5"  # Increased TTL
            ])
        else:
            # For mpegts and other formats, use UDP with multicast
            cmd.extend([
                "-c:v", "libx264",
                "-preset", "veryfast",  # Better quality than ultrafast
                "-tune", "zerolatency",
                "-b:v", "2M",  # Set video bitrate for consistency
                "-maxrate", "2M",
                "-bufsize", "4M",  # Buffer size = 2x bitrate
                "-g", "50",  # GOP size - keyframe every 50 frames
                "-c:a", "aac",  # Explicit audio codec
                "-b:a", "128k",  # Audio bitrate
                "-f", output_format,
                f"udp://{multicast_addr}:{port}?pkt_size=1316&buffer_size=65535&fifo_size=1000000&ttl=5"
            ])
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE
            )
            
            # Store process and metadata
            storage.active_streams[video_id] = {
                'process': process,
                'start_time_ms': stream_start_time,
                'multicast_addr': multicast_addr,
                'port': port
            }
            storage.videos[video_id]["is_streaming"] = True
            
            return {
                "video_id": video_id,
                "status": "streaming",
                "stream_url": f"udp://{multicast_addr}:{port}",
                "stream_start_time_ms": stream_start_time,
                "vlc_command": f"vlc udp://@{multicast_addr}:{port}",
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
        
        # Send 'q' to FFmpeg stdin for clean exit (better than SIGTERM)
        try:
            process.stdin.write(b'q')
            process.stdin.flush()
            process.wait(timeout=2)
        except (subprocess.TimeoutExpired, Exception):
            # If 'q' didn't work, try SIGTERM
            try:
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                # Force kill if still not dead
                process.kill()
                process.wait()
        
        # Ensure process is fully terminated
        if process.poll() is None:
            process.kill()
            process.wait()
        
        # Clean up storage
        del storage.active_streams[video_id]
        storage.videos[video_id]["is_streaming"] = False
        
        # Give the OS a moment to release resources
        time.sleep(0.1)
        
        return {
            "video_id": video_id,
            "status": "stopped"
        }
    
    @staticmethod
    def get_stream_status(video_id: int) -> dict:
        """Get stream status with multicast information"""
        
        if video_id not in storage.videos:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        
        is_active = video_id in storage.active_streams
        
        result = {
            "video_id": video_id,
            "is_streaming": is_active
        }
        
        if is_active:
            stream_data = storage.active_streams[video_id]
            process = stream_data['process']
            multicast_addr = stream_data['multicast_addr']
            port = stream_data['port']
            
            # Check if process is actually running
            if process.poll() is not None:
                # Process died, clean up
                del storage.active_streams[video_id]
                storage.videos[video_id]["is_streaming"] = False
                result["is_streaming"] = False
                result["error"] = "Stream process died unexpectedly"
            else:
                result["stream_url"] = f"udp://{multicast_addr}:{port}"
                result["stream_start_time_ms"] = stream_data['start_time_ms']
                result["vlc_command"] = f"vlc udp://@{multicast_addr}:{port}"
                result["pid"] = process.pid
        
        return result