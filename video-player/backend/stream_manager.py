import subprocess
import signal
import time
import threading
import logging
import os
from pathlib import Path
from fastapi import HTTPException
from storage import storage
from websocket_manager import manager as ws_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamManager:
    """Handles FFmpeg streaming with DASH output and UDP raw frames for AI"""
    
    DASH_OUTPUT_DIR = Path("./dash_streams")
    UDP_BASE_PORT = 9000  # Base port for raw streams
    
    # DASH segment retention settings
    DASH_SEGMENT_DURATION = 2  # seconds
    DASH_WINDOW_SIZE = 5  # keep last 5 segments
    
    _stderr_threads = {}
    _stream_locks = {}
    _client_counts = {}
    
    def __init__(self):
        self.DASH_OUTPUT_DIR.mkdir(exist_ok=True)
    
    @staticmethod
    def _get_udp_port(video_id: int) -> int:
        return StreamManager.UDP_BASE_PORT + video_id
    
    @staticmethod
    def _get_lock(video_id: int) -> threading.Lock:
        if video_id not in StreamManager._stream_locks:
            StreamManager._stream_locks[video_id] = threading.Lock()
        return StreamManager._stream_locks[video_id]
    
    @staticmethod
    def _consume_stderr(video_id: int, process):
        """Consume stderr to prevent buffer blocking"""
        logger.info(f"[Stream {video_id}] Starting stderr consumer thread")
        try:
            for line in iter(process.stderr.readline, b''):
                if not line:
                    break
                line_str = line.decode('utf-8', errors='ignore').strip()
                if 'error' in line_str.lower() and 'configuration:' not in line_str.lower():
                    logger.error(f"[Stream {video_id}] FFmpeg error: {line_str}")
                elif 'warning' in line_str.lower():
                    logger.warning(f"[Stream {video_id}] FFmpeg warning: {line_str}")
        except Exception as e:
            # This can happen normally when process is killed
            logger.info(f"[Stream {video_id}] Stderr consumer thread finished: {e}")
        finally:
            logger.info(f"[Stream {video_id}] Stderr consumer thread stopped")
    
    @staticmethod
    def _validate_video_properties(video_data: dict) -> None:
        """Validate that video has all required properties with valid values"""
        width = video_data.get("width")
        height = video_data.get("height")
        fps = video_data.get("fps")
        
        if not width or width <= 0:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid video properties: width={width}. Video must have valid dimensions."
            )
        
        if not height or height <= 0:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid video properties: height={height}. Video must have valid dimensions."
            )
        
        if not fps or fps <= 0:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid video properties: fps={fps}. Video must have valid frame rate."
            )
    
    @staticmethod
    def _start_ffmpeg_process(video_id: int) -> dict:
        """Start FFmpeg process for both DASH and UDP streaming"""
        video_data = storage.videos[video_id]
        file_path = video_data["file_path"]
        
        # Validate video properties
        StreamManager._validate_video_properties(video_data)
        
        udp_port = StreamManager._get_udp_port(video_id)
        stream_start_time = int(time.time() * 1000)
        
        output_width = video_data["width"]
        output_height = video_data["height"]
        output_fps = video_data["fps"]
        
        udp_info = {
            "port": udp_port,
            "width": output_width,
            "height": output_height,
            "pix_fmt": "rgb24",
            "fps": output_fps,
            "bytes_per_pixel": 3,
            "frame_size_bytes": output_width * output_height * 3
        }
        
        logger.info(f"[Stream {video_id}] Starting stream from {file_path}")
        logger.info(f"[Stream {video_id}] AI (UDP) Info: {udp_info}")
        
        dash_dir = StreamManager.DASH_OUTPUT_DIR / str(video_id)
        dash_dir.mkdir(parents=True, exist_ok=True)
        dash_manifest = dash_dir / "manifest.mpd"
        
        cmd = [
            "ffmpeg",
            "-re",
            "-stream_loop", "-1",
            "-i", file_path,
            "-filter_complex", f"[0:v]fps=fps={output_fps}[v_base]; [v_base]split=2[v_dash][v_udp]",
            
            "-map", "[v_dash]", "-map", "0:a",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-b:v", "2M",
            "-maxrate", "2M",
            "-bufsize", "4M",
            "-g", str(int(output_fps * 2)),
            "-c:a", "aac",
            "-b:a", "128k",
            "-f", "dash",
            "-seg_duration", str(StreamManager.DASH_SEGMENT_DURATION),
            "-window_size", str(StreamManager.DASH_WINDOW_SIZE),
            "-extra_window_size", str(StreamManager.DASH_WINDOW_SIZE),
            "-remove_at_exit", "1",
            "-streaming", "1",
            "-ldash", "1",
            str(dash_manifest),
            
            # UDP output (for AI analytics)
            "-map", "[v_udp]",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-f", "mpegts",
            "-mpegts_copyts", "1",
            f"udp://127.0.0.1:{udp_port}"
        ]
        
        logger.debug(f"[Stream {video_id}] FFmpeg command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            bufsize=0,
            preexec_fn=os.setsid
        )
        
        logger.info(f"[Stream {video_id}] FFmpeg process started (PID: {process.pid})")
        
        storage.active_streams[video_id] = {
            'process': process,
            'start_time_ms': stream_start_time,
            'udp_info': udp_info,
            'dash_manifest': str(dash_manifest)
        }
        storage.videos[video_id]["is_streaming"] = True
        
        stderr_thread = threading.Thread(
            target=StreamManager._consume_stderr,
            args=(video_id, process),
            daemon=True
        )
        stderr_thread.start()
        StreamManager._stderr_threads[video_id] = stderr_thread
        
        time.sleep(2)
        
        if process.poll() is not None:
            logger.error(f"[Stream {video_id}] Process died immediately after start")
            del storage.active_streams[video_id]
            storage.videos[video_id]["is_streaming"] = False
            raise HTTPException(
                status_code=500, 
                detail="FFmpeg process died immediately after start. Check video file integrity and logs."
            )
        
        logger.info(f"[Stream {video_id}] Stream started successfully")
        
        return {
            "video_id": video_id,
            "status": "streaming",
            "stream_start_time_ms": stream_start_time,
            "pid": process.pid,
            "udp": udp_info,
            "dash": {
                "manifest_url": f"/dash/{video_id}/manifest.mpd"
            }
        }
    
    @staticmethod
    def start_stream(video_id: int) -> dict:
        """
        Start streaming for a video. 
        Always uses original video resolution and DASH format.
        """
        if video_id not in storage.videos:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        
        lock = StreamManager._get_lock(video_id)
        
        with lock:
            # Track client count
            if video_id not in StreamManager._client_counts:
                StreamManager._client_counts[video_id] = 0
            
            StreamManager._client_counts[video_id] += 1
            logger.info(f"[Stream {video_id}] Client connected. Total clients: {StreamManager._client_counts[video_id]}")
            
            # Check if stream already exists
            if video_id in storage.active_streams:
                stream_data = storage.active_streams[video_id]
                process = stream_data['process']
                
                if process.poll() is None:
                    # Stream is active, return existing info
                    logger.info(f"[Stream {video_id}] Reusing existing stream")
                    return {
                        "video_id": video_id,
                        "status": "streaming",
                        "stream_start_time_ms": stream_data['start_time_ms'],
                        "pid": process.pid,
                        "udp": stream_data['udp_info'],
                        "dash": {
                            "manifest_url": f"/dash/{video_id}/manifest.mpd"
                        },
                        "clients": StreamManager._client_counts[video_id]
                    }
                else:
                    # Process died, clean up and restart
                    logger.warning(f"[Stream {video_id}] Found dead stream, cleaning up and restarting")
                    del storage.active_streams[video_id]
                    storage.videos[video_id]["is_streaming"] = False
                    # Note: _client_counts[video_id] is preserved
            
            # Start new stream
            try:
                result = StreamManager._start_ffmpeg_process(video_id)
                result["clients"] = StreamManager._client_counts[video_id]
                return result
            except HTTPException:
                # Re-raise HTTP exceptions as-is
                StreamManager._client_counts[video_id] -= 1
                if StreamManager._client_counts[video_id] <= 0:
                    del StreamManager._client_counts[video_id]
                raise
            except Exception as e:
                # Cleanup on failure
                StreamManager._client_counts[video_id] -= 1
                if StreamManager._client_counts[video_id] <= 0:
                    del StreamManager._client_counts[video_id]
                
                logger.error(f"[Stream {video_id}] Failed to start stream: {e}")
                
                if video_id in storage.active_streams:
                    del storage.active_streams[video_id]
                if video_id in storage.videos:
                    storage.videos[video_id]["is_streaming"] = False
                
                raise HTTPException(status_code=500, detail=f"Failed to start stream: {str(e)}")
    
    @staticmethod
    async def stop_stream(video_id: int) -> dict:
        """Stop streaming for a video (reference counting for multiple clients)"""
        lock = StreamManager._get_lock(video_id)
        
        with lock:
            # Decrement client count
            if video_id in StreamManager._client_counts:
                StreamManager._client_counts[video_id] -= 1
                logger.info(f"[Stream {video_id}] Client disconnected. Remaining clients: {StreamManager._client_counts[video_id]}")
                
                # Keep stream alive if other clients are connected
                if StreamManager._client_counts[video_id] > 0:
                    return {
                        "video_id": video_id,
                        "status": "streaming",
                        "clients": StreamManager._client_counts[video_id],
                        "message": "Stream continues for other clients"
                    }
                
                del StreamManager._client_counts[video_id]
            else:
                logger.warning(f"[Stream {video_id}] Stop called but no clients were tracked")

            
            # Check if stream exists (it should if client_counts just hit zero)
            if video_id not in storage.active_streams:
                logger.warning(f"[Stream {video_id}] Stop called, client count is zero, but stream not in active_streams")
                storage.videos[video_id]["is_streaming"] = False # Ensure consistency
                return {
                    "video_id": video_id,
                    "status": "already_stopped"
                }
            
            stream_data = storage.active_streams[video_id]
            process = stream_data['process']
            
            logger.info(f"[Stream {video_id}] Stopping stream (PID: {process.pid})")
            
            # Terminate FFmpeg process
            try:
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signal.SIGTERM)
                
                try:
                    process.wait(timeout=5)
                    logger.info(f"[Stream {video_id}] Process terminated cleanly")
                except subprocess.TimeoutExpired:
                    logger.warning(f"[Stream {video_id}] SIGTERM timeout, force killing")
                    os.killpg(pgid, signal.SIGKILL)
                    process.wait()
                    logger.info(f"[Stream {video_id}] Process force killed")
            except (ProcessLookupError, OSError) as e:
                logger.warning(f"[Stream {video_id}] Process cleanup error: {e}")
            
            # Clean up storage
            del storage.active_streams[video_id]
            storage.videos[video_id]["is_streaming"] = False
            
            # Clean up stderr thread
            if video_id in StreamManager._stderr_threads:
                stderr_thread = StreamManager._stderr_threads[video_id]
                stderr_thread.join(timeout=2)
                del StreamManager._stderr_threads[video_id]
            
            # Clean up DASH directory
            dash_dir = StreamManager.DASH_OUTPUT_DIR / str(video_id)
            if dash_dir.exists():
                import shutil
                try:
                    shutil.rmtree(dash_dir)
                    logger.info(f"[Stream {video_id}] DASH directory cleaned up")
                except Exception as e:
                    logger.warning(f"[Stream {video_id}] Could not remove DASH dir: {e}")
            
            time.sleep(0.5)
            
            logger.info(f"[Stream {video_id}] Stream stopped and cleaned up")
            
            # Force close WebSockets
            try:
                await ws_manager.close_connections(video_id)
            except Exception as e:
                logger.warning(f"[Stream {video_id}] Failed to close websockets: {e}")

            return {
                "video_id": video_id,
                "status": "stopped"
            }
    
    @staticmethod
    def get_stream_status(video_id: int) -> dict:
        """Get current status of a stream"""
        if video_id not in storage.videos:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        
        lock = StreamManager._get_lock(video_id)
        
        with lock:
            is_active = video_id in storage.active_streams
            
            result = {
                "video_id": video_id,
                "is_streaming": is_active,
                "clients": StreamManager._client_counts.get(video_id, 0)
            }
            
            if is_active:
                stream_data = storage.active_streams[video_id]
                process = stream_data['process']
                
                poll_result = process.poll()
                
                if poll_result is not None:
                    # Process died unexpectedly
                    logger.warning(f"[Stream {video_id}] Detected dead process during status check")
                    del storage.active_streams[video_id]
                    storage.videos[video_id]["is_streaming"] = False
                    
                    # Clean up client count if it's still > 0
                    if video_id in StreamManager._client_counts:
                        logger.warning(f"[Stream {video_id}] Removing {StreamManager._client_counts[video_id]} clients from dead stream")
                        del StreamManager._client_counts[video_id]
                        
                    result["is_streaming"] = False
                    result["clients"] = 0
                    result["error"] = f"Stream process died unexpectedly (exit code: {poll_result})"
                else:
                    # Stream is active
                    result["status"] = "streaming"
                    result["stream_start_time_ms"] = stream_data['start_time_ms']
                    result["pid"] = process.pid
                    result["udp"] = stream_data['udp_info']
                    result["dash"] = {
                        "manifest_url": f"/dash/{video_id}/manifest.mpd"
                    }
            else:
                # If it's not active, client count should be 0
                if StreamManager._client_counts.get(video_id, 0) > 0:
                    logger.warning(f"[Stream {video_id}] Stream is not active but client count is {StreamManager._client_counts[video_id]}. Resetting.")
                    del StreamManager._client_counts[video_id]
                    result["clients"] = 0
                
                storage.videos[video_id]["is_streaming"] = False
        
        return result
    
    @staticmethod
    def cleanup_all_streams():
        """Cleanup all active streams (called on shutdown)"""
        logger.info("Cleaning up all active streams...")
        video_ids = list(storage.active_streams.keys())
        
        for video_id in video_ids:
            try:
                lock = StreamManager._get_lock(video_id)
                with lock:
                    # Set client count to 1 to ensure cleanup happens
                    # regardless of tracked clients
                    StreamManager._client_counts[video_id] = 1
                StreamManager.stop_stream(video_id)
            except Exception as e:
                logger.error(f"[Stream {video_id}] Error during cleanup: {e}")
        
        logger.info("All streams cleaned up")

stream_manager = StreamManager()