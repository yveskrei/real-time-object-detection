import subprocess
import signal
import time
import threading
import logging
from fastapi import HTTPException
from storage import storage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamManager:
    """Handles FFmpeg streaming processes with SRT support and health monitoring"""
    
    # Base port for SRT streams - each video gets a unique port
    SRT_BASE_PORT = 9000
    
    # Health check interval (seconds)
    HEALTH_CHECK_INTERVAL = 30
    
    # Store monitoring threads
    _monitor_threads = {}
    _stderr_threads = {}
    
    @staticmethod
    def _get_srt_port(video_id: int) -> int:
        """Generate SRT port for a video ID
        
        Each video gets a unique port starting from SRT_BASE_PORT
        """
        return StreamManager.SRT_BASE_PORT + video_id
    
    @staticmethod
    def _consume_stderr(video_id: int, process):
        """Continuously consume stderr to prevent buffer blocking"""
        logger.info(f"[Stream {video_id}] Starting stderr consumer thread")
        
        try:
            for line in iter(process.stderr.readline, b''):
                if not line:
                    break
                
                line_str = line.decode('utf-8', errors='ignore').strip()
                
                # Log important messages
                if 'error' in line_str.lower():
                    logger.error(f"[Stream {video_id}] FFmpeg error: {line_str}")
                elif 'warning' in line_str.lower():
                    logger.warning(f"[Stream {video_id}] FFmpeg warning: {line_str}")
                # Uncomment for full debug logging:
                # else:
                #     logger.debug(f"[Stream {video_id}] FFmpeg: {line_str}")
        except Exception as e:
            logger.error(f"[Stream {video_id}] Error consuming stderr: {e}")
        finally:
            logger.info(f"[Stream {video_id}] Stderr consumer thread stopped")
    
    @staticmethod
    def _monitor_stream(video_id: int):
        """Monitor stream health and restart if necessary"""
        logger.info(f"[Stream {video_id}] Starting health monitor")
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        while video_id in storage.active_streams:
            time.sleep(StreamManager.HEALTH_CHECK_INTERVAL)
            
            if video_id not in storage.active_streams:
                break
            
            stream_data = storage.active_streams[video_id]
            process = stream_data['process']
            
            # Check if process is still running
            poll_result = process.poll()
            
            if poll_result is not None:
                logger.error(f"[Stream {video_id}] Process died (exit code: {poll_result})")
                consecutive_failures += 1
                
                # Mark as stopped
                storage.videos[video_id]["is_streaming"] = False
                
                # Try to restart if under failure threshold
                if consecutive_failures <= max_consecutive_failures:
                    try:
                        logger.info(f"[Stream {video_id}] Attempting auto-restart (attempt {consecutive_failures}/{max_consecutive_failures})...")
                        
                        # Clean up old stream data
                        del storage.active_streams[video_id]
                        
                        # Small delay before restart
                        time.sleep(2)
                        
                        # Restart with same parameters
                        StreamManager.start_stream(
                            video_id=video_id,
                            output_format=stream_data.get('output_format', 'mpegts'),
                            resolution=stream_data.get('resolution')
                        )
                        logger.info(f"[Stream {video_id}] Successfully restarted")
                        consecutive_failures = 0  # Reset on successful restart
                    except Exception as e:
                        logger.error(f"[Stream {video_id}] Auto-restart failed: {e}")
                        if video_id in storage.active_streams:
                            del storage.active_streams[video_id]
                        storage.videos[video_id]["is_streaming"] = False
                else:
                    logger.error(f"[Stream {video_id}] Max restart attempts reached, giving up")
                    if video_id in storage.active_streams:
                        del storage.active_streams[video_id]
                    storage.videos[video_id]["is_streaming"] = False
                
                break
            else:
                # Process is healthy
                consecutive_failures = 0
        
        logger.info(f"[Stream {video_id}] Health monitor stopped")
    
    @staticmethod
    def start_stream(video_id: int, output_format: str = "mpegts", resolution: str = None) -> dict:
        """Start FFmpeg stream for a video using SRT (Secure Reliable Transport)
        
        SRT provides reliable, low-latency streaming over standard network connections
        with automatic retransmission and congestion control.
        """
        
        if video_id not in storage.videos:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        
        if video_id in storage.active_streams:
            raise HTTPException(status_code=400, detail=f"Stream {video_id} already active")
        
        video_data = storage.videos[video_id]
        file_path = video_data["file_path"]
        srt_port = StreamManager._get_srt_port(video_id)
        
        # Record stream start time (global) - milliseconds since epoch
        stream_start_time = int(time.time() * 1000)
        
        logger.info(f"[Stream {video_id}] Starting stream from {file_path} on port {srt_port}")
        
        # Build FFmpeg command with improved settings for long-running streams
        cmd = [
            "ffmpeg",
            "-re",  # Read input at native frame rate
            "-stream_loop", "-1",  # Loop indefinitely
            "-fflags", "+genpts+igndts",  # Generate PTS and ignore input DTS
            "-avoid_negative_ts", "make_zero",  # Avoid negative timestamps
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
        
        # Video encoding settings optimized for long-running streams
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-profile:v", "baseline",  # Better compatibility
            "-level", "3.1",
            "-pix_fmt", "yuv420p",  # Explicit pixel format
            "-b:v", "2M",
            "-maxrate", "2M",
            "-bufsize", "4M",
            "-g", "50",  # GOP size - keyframe every 50 frames
            "-keyint_min", "50",  # Minimum GOP size
            "-sc_threshold", "0",  # Disable scene change detection for consistent GOP
            "-forced-idr", "1",  # Force IDR frames
        ])
        
        # Audio encoding settings
        cmd.extend([
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "48000",
            "-ac", "2",  # Stereo
        ])
        
        # Output format settings
        cmd.extend([
            "-f", "mpegts",
            "-mpegts_flags", "resend_headers",  # Periodically resend headers for reliability
            "-flush_packets", "1",  # Flush packets immediately
            "-max_delay", "0",  # Minimize delay
            "-muxrate", "2500000",  # Mux rate slightly higher than video bitrate
        ])
        
        # SRT output with robust settings for long streams
        # Increased buffer sizes and latency for stability
        srt_url = (
            f"srt://0.0.0.0:{srt_port}"
            f"?mode=listener"  # FFmpeg listens, clients connect
            f"&pkt_size=1316"  # Standard packet size
            f"&latency=1000000"  # 1 second latency (1,000,000 microseconds)
            f"&rcvbuf=12058624"  # 12MB receive buffer
            f"&sndbuf=12058624"  # 12MB send buffer
            f"&lossmaxttl=0"  # Infinite retransmission attempts
            f"&maxbw=5000000"  # 5MB/s max bandwidth
            f"&streamid=video_{video_id}"  # Stream identifier
        )
        
        cmd.append(srt_url)
        
        logger.debug(f"[Stream {video_id}] FFmpeg command: {' '.join(cmd)}")
        
        try:
            # Start FFmpeg with unbuffered pipes
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                bufsize=0  # Unbuffered
            )
            
            logger.info(f"[Stream {video_id}] FFmpeg process started (PID: {process.pid})")
            
            # Store process and metadata
            storage.active_streams[video_id] = {
                'process': process,
                'start_time_ms': stream_start_time,
                'srt_port': srt_port,
                'output_format': output_format,
                'resolution': resolution
            }
            storage.videos[video_id]["is_streaming"] = True
            
            # Start stderr consumer thread (CRITICAL - prevents buffer blocking)
            stderr_thread = threading.Thread(
                target=StreamManager._consume_stderr,
                args=(video_id, process),
                daemon=True
            )
            stderr_thread.start()
            StreamManager._stderr_threads[video_id] = stderr_thread
            
            # Start health monitoring thread
            monitor_thread = threading.Thread(
                target=StreamManager._monitor_stream,
                args=(video_id,),
                daemon=True
            )
            monitor_thread.start()
            StreamManager._monitor_threads[video_id] = monitor_thread
            
            # Give FFmpeg a moment to initialize
            time.sleep(0.5)
            
            # Check if process is still alive after startup
            if process.poll() is not None:
                logger.error(f"[Stream {video_id}] Process died immediately after start")
                raise Exception("FFmpeg process died immediately after start")
            
            logger.info(f"[Stream {video_id}] Stream started successfully")
            
            return {
                "video_id": video_id,
                "status": "streaming",
                "port": srt_port,
                "stream_url": f"srt://127.0.0.1:{srt_port}?mode=caller",
                "stream_start_time_ms": stream_start_time,
                "pid": process.pid
            }
        
        except Exception as e:
            logger.error(f"[Stream {video_id}] Failed to start stream: {e}")
            
            # Clean up if startup failed
            if video_id in storage.active_streams:
                del storage.active_streams[video_id]
            if video_id in storage.videos:
                storage.videos[video_id]["is_streaming"] = False
            
            raise HTTPException(status_code=500, detail=f"Failed to start stream: {str(e)}")
    
    @staticmethod
    def stop_stream(video_id: int) -> dict:
        """Stop FFmpeg stream for a video"""
        
        if video_id not in storage.active_streams:
            raise HTTPException(status_code=404, detail=f"No active stream for video {video_id}")
        
        stream_data = storage.active_streams[video_id]
        process = stream_data['process']
        
        logger.info(f"[Stream {video_id}] Stopping stream (PID: {process.pid})")
        
        # Send 'q' to FFmpeg stdin for clean exit
        try:
            process.stdin.write(b'q\n')
            process.stdin.flush()
            process.wait(timeout=3)
            logger.info(f"[Stream {video_id}] Process terminated cleanly")
        except (subprocess.TimeoutExpired, BrokenPipeError, OSError) as e:
            logger.warning(f"[Stream {video_id}] Clean exit failed: {e}, trying SIGTERM")
            # If 'q' didn't work, try SIGTERM
            try:
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=3)
                logger.info(f"[Stream {video_id}] Process terminated with SIGTERM")
            except subprocess.TimeoutExpired:
                logger.warning(f"[Stream {video_id}] SIGTERM failed, force killing")
                # Force kill if still not dead
                process.kill()
                process.wait()
                logger.info(f"[Stream {video_id}] Process force killed")
        
        # Ensure process is fully terminated
        if process.poll() is None:
            logger.warning(f"[Stream {video_id}] Process still alive, force killing")
            process.kill()
            process.wait()
        
        # Clean up storage
        del storage.active_streams[video_id]
        storage.videos[video_id]["is_streaming"] = False
        
        # Wait for threads to finish
        if video_id in StreamManager._monitor_threads:
            monitor_thread = StreamManager._monitor_threads[video_id]
            monitor_thread.join(timeout=2)
            del StreamManager._monitor_threads[video_id]
        
        if video_id in StreamManager._stderr_threads:
            stderr_thread = StreamManager._stderr_threads[video_id]
            stderr_thread.join(timeout=2)
            del StreamManager._stderr_threads[video_id]
        
        # Give the OS a moment to release resources
        time.sleep(0.2)
        
        logger.info(f"[Stream {video_id}] Stream stopped and cleaned up")
        
        return {
            "video_id": video_id,
            "status": "stopped"
        }
    
    @staticmethod
    def get_stream_status(video_id: int) -> dict:
        """Get stream status with SRT port information"""
        
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
            srt_port = stream_data['srt_port']
            
            # Check if process is actually running
            poll_result = process.poll()
            
            if poll_result is not None:
                # Process died, clean up
                logger.warning(f"[Stream {video_id}] Detected dead process during status check (exit code: {poll_result})")
                del storage.active_streams[video_id]
                storage.videos[video_id]["is_streaming"] = False
                result["is_streaming"] = False
                result["error"] = f"Stream process died unexpectedly (exit code: {poll_result})"
            else:
                result["port"] = srt_port
                result["stream_url"] = f"srt://127.0.0.1:{srt_port}?mode=caller"
                result["stream_start_time_ms"] = stream_data['start_time_ms']
                result["pid"] = process.pid
                result["uptime_seconds"] = (int(time.time() * 1000) - stream_data['start_time_ms']) / 1000
        
        return result
    
    @staticmethod
    def cleanup_all_streams():
        """Stop all active streams - useful for shutdown"""
        logger.info("Cleaning up all active streams...")
        video_ids = list(storage.active_streams.keys())
        
        for video_id in video_ids:
            try:
                StreamManager.stop_stream(video_id)
            except Exception as e:
                logger.error(f"[Stream {video_id}] Error during cleanup: {e}")
        
        logger.info("All streams cleaned up")