use anyhow::{Context, Result};
use ffmpeg_next as ffmpeg;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::task::JoinHandle;
use tokio::time::sleep;
use reqwest::Url;
use serde::{Deserialize, Serialize};

use crate::player_proxy::PlayerSession;
use crate::get_runtime;
use crate::{SourceFramesCallback, SourceStoppedCallback, SourceNameCallback, SourceStatusCallback};
use crate::{log_info, log_error, log_debug};

// Stream timeout constant
const STREAM_TIMEOUT: Duration = Duration::from_secs(10);

// Info for the raw video stream
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RawStreamInfo {
    pub protocol: Option<String>,
    pub host: Option<String>,
    pub port: u16,
    pub width: u32,
    pub height: u32,
    pub pix_fmt: String,
    pub fps: f64,
    pub bytes_per_pixel: u16,
    pub frame_size_bytes: u32,
}

// Source status codes for C FFI
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum SourceStatus {
    Ok = 0,
    NotStreaming = 1,
    NotFound = 2,
    ConnectionError = 3,
    DecodeError = 4,
}

// Global state for managing streams
pub struct StreamManager {
    streams: Mutex<HashMap<i32, JoinHandle<()>>>,
    callbacks: Mutex<Option<Callbacks>>,
    player_session: PlayerSession,
}

#[derive(Clone, Copy)]
struct Callbacks {
    source_frames: SourceFramesCallback,
    source_stopped: SourceStoppedCallback,
    source_name: SourceNameCallback,
    source_status: SourceStatusCallback,
}

// Function pointers are Send and Sync by nature
unsafe impl Send for Callbacks {}
unsafe impl Sync for Callbacks {}

pub static STREAM_MANAGER: OnceLock<Arc<StreamManager>> = OnceLock::new();

pub fn get_stream_manager() -> &'static Arc<StreamManager> {
    STREAM_MANAGER.get_or_init(|| {
        match StreamManager::new() {
            Ok(manager) => Arc::new(manager),
            Err(e) => {
                eprintln!("Failed to initialize StreamManager: {}", e);
                panic!("Failed to initialize StreamManager: {}", e);
            }
        }
    })
}

impl StreamManager {
    fn new() -> Result<Self> {
        Ok(Self {
            streams: Mutex::new(HashMap::new()),
            callbacks: Mutex::new(None),
            player_session: PlayerSession::new()?,
        })
    }

    pub fn set_callbacks(
        &self,
        source_frames: SourceFramesCallback,
        source_stopped: SourceStoppedCallback,
        source_name: SourceNameCallback,
        source_status: SourceStatusCallback,
    ) {
        let callbacks = Callbacks {
            source_frames,
            source_stopped,
            source_name,
            source_status,
        };
        *self.callbacks.lock().unwrap() = Some(callbacks);
        log_info!("Callbacks registered");
    }

    pub fn are_callbacks_set(&self) -> bool {
        self.callbacks.lock().unwrap().is_some()
    }

    pub fn init_sources(&self, source_ids: Vec<i32>) {
        for source_id in source_ids {
            self.start_source_monitor(source_id);
            log_info!("[Source {}] Initialized!", source_id);
        }
    }

    fn start_source_monitor(&self, source_id: i32) {
        let manager = get_stream_manager().clone();
        
        let handle = get_runtime().spawn(async move {
            log_debug!("[Source {}] Starting monitor task", source_id);
            
            // Get host from base_url. Assumes backend is on same host.
            let host = match Url::parse(manager.player_session.base_url()) {
                Ok(url) => url
                    .host_str()
                    .unwrap_or("127.0.0.1")
                    .to_string(),
                Err(_) => "127.0.0.1".to_string(),
            };
            
            log_debug!("[Source {}] Using backend host: {}", source_id, host);
            
            loop {
                // Check if we have callbacks registered
                let callbacks = {
                    let cb_lock = manager.callbacks.lock().unwrap();
                    match *cb_lock {
                        None => {
                            log_error!("[Source {}] Callbacks not set, waiting...", source_id);
                            None
                        }
                        Some(cbs) => Some(cbs)
                    }
                };
                
                let callbacks = match callbacks {
                    None => {
                        sleep(STREAM_TIMEOUT).await;
                        continue;
                    }
                    Some(cbs) => cbs
                };

                // Check stream status
                match manager.player_session.get_stream_status(source_id).await {
                    Ok(status) => {
                        if !status.is_streaming {
                            log_error!("[Source {}] Not streaming, waiting...", source_id);
                            (callbacks.source_status)(source_id, SourceStatus::NotStreaming as i32);
                            sleep(STREAM_TIMEOUT).await;
                            continue;
                        }

                        // UPDATED: Get raw stream info from 'udp' block
                        let raw_stream_info = match status.udp {
                            Some(info) => info,
                            None => {
                                // UPDATED: Log message
                                log_error!("[Source {}] No raw stream info ('udp' block) available from backend", source_id);
                                (callbacks.source_status)(source_id, SourceStatus::ConnectionError as i32);
                                sleep(STREAM_TIMEOUT).await;
                                continue;
                            }
                        };

                        // Get video name from backend
                        if let Ok(video_info) = manager.get_video_info(source_id).await {
                            let name_cstr = std::ffi::CString::new(video_info.name)
                                .unwrap_or_else(|_| std::ffi::CString::new("unknown").unwrap());
                            (callbacks.source_name)(source_id, name_cstr.into_raw());
                        }

                        // UPDATED: Log for UDP
                        log_info!("[Source {}] Stream active, connecting to udp://{}:{}", 
                                 source_id, host, raw_stream_info.port);
                        (callbacks.source_status)(source_id, SourceStatus::Ok as i32);

                        // Start consuming stream
                        if let Err(e) = manager.consume_stream(source_id, raw_stream_info.clone(), host.clone(), callbacks, status.pid).await {
                            log_error!("[Source {}] Stream error: {}", source_id, e);
                            (callbacks.source_stopped)(source_id);
                        }
                    }
                    Err(e) => {
                        log_error!("[Source {}] Failed to get status: {}", source_id, e);
                        (callbacks.source_status)(source_id, SourceStatus::ConnectionError as i32);
                    }
                }

                // Wait before retry
                log_debug!("[Source {}] Retrying in {:?}...", source_id, STREAM_TIMEOUT);
                sleep(STREAM_TIMEOUT).await;
            }
        });

        self.streams.lock().unwrap().insert(source_id, handle);
    }

    async fn get_video_info(&self, video_id: i32) -> Result<VideoInfo> {
        let url = format!("{}/videos/{}", self.player_session.base_url(), video_id);
        let response = reqwest::get(&url)
            .await?;
        let info: VideoInfo = response
            .json()
            .await?;
        Ok(info)
    }

    async fn consume_stream(
        &self,
        source_id: i32,
        stream_info: RawStreamInfo,
        host: String,
        callbacks: Callbacks,
        stream_pid: Option<i32>,
    ) -> Result<()> {
        let session = self.player_session.clone();
        let stop_signal = Arc::new(AtomicBool::new(false));
        let stop_signal_decode = stop_signal.clone();
        
        // Spawn a task to periodically check if stream is still active on backend
        let mut keepalive_handle = tokio::spawn(async move {
            loop {
                sleep(Duration::from_secs(2)).await;
                
                match session.get_stream_status(source_id).await {
                    Ok(status) => {
                        if !status.is_streaming {
                            log_info!("[Source {}] Backend reports stream stopped, triggering reconnect", source_id);
                            return; // Stream is not active anymore
                        }
                        if status.pid != stream_pid {
                            log_info!("[Source {}] Stream PID changed (old: {:?}, new: {:?}), triggering reconnect", source_id, stream_pid, status.pid);
                            return;
                        }
                    }
                    Err(e) => {
                        log_error!("[Source {}] Keepalive check failed: {}", source_id, e);
                    }
                }
            }
        });
        
        // Spawn blocking task for FFmpeg operations
        let mut decode_handle = tokio::task::spawn_blocking(move || {
            if let Err(e) = decode_stream(source_id, stream_info, host, callbacks, stop_signal_decode) {
                log_error!("[Source {}] Decode error: {}", source_id, e);
                (callbacks.source_status)(source_id, SourceStatus::DecodeError as i32);
            }
        });
        
        // Wait for either decode to finish or keepalive to detect stream stopped
        tokio::select! {
            _ = &mut decode_handle => {
                keepalive_handle.abort();
            }
            _ = &mut keepalive_handle => {
                // Keepalive detected stream stopped
                // Signal decode to stop
                stop_signal.store(true, Ordering::Relaxed);
                
                // Wait for decode task to finish with timeout to ensure cleanup
                log_info!("[Source {}] Stream stopped, waiting for decode task to cleanup...", source_id);
                
                let timeout_result = tokio::time::timeout(
                    Duration::from_secs(5),
                    decode_handle
                ).await;
                
                match timeout_result {
                    Ok(_) => {
                        log_debug!("[Source {}] Decode task completed cleanup successfully", source_id);
                    }
                    Err(_) => {
                        log_error!("[Source {}] Decode task cleanup timed out after 5s", source_id);
                    }
                }
            }
        }

        Ok(())
    }
}

#[derive(serde::Deserialize)]
struct VideoInfo {
    pub name: String,
}

fn decode_stream(
    source_id: i32, 
    stream_info: RawStreamInfo, 
    host: String,
    callbacks: Callbacks, 
    stop_signal: Arc<AtomicBool>,
) -> Result<()> {
    // UPDATED: Connect to UDP stream with reuse=1
    let connection_url = format!("udp://{}:{}?reuse=1", host, stream_info.port);

    log_info!("[Source {}] Connecting to UDP stream: {}", source_id, connection_url);

    let mut input_opts = ffmpeg::Dictionary::new();
    input_opts.set("analyzeduration", "500000"); // 0.5s
    input_opts.set("probesize", "500000"); // 500KB
    input_opts.set("fflags", "nobuffer");
    input_opts.set("flags", "low_delay");
    // Set UDP timeout to 3 seconds (in microseconds)
    // This ensures that if the stream stops sending data, we don't block forever
    // and can release the port.
    input_opts.set("timeout", "3000000");
    // We let FFmpeg auto-detect format (mpegts) and codec (h264)

    let mut last_error = None;
    for attempt in 1..=3 {
        log_info!("[Source {}] Connection attempt {}/3", source_id, attempt);

        // We pass options, but don't force rawvideo
        match ffmpeg::format::input_with_dictionary(&connection_url, input_opts.clone()) {
            Ok(mut ictx) => {
                log_info!("[Source {}] Successfully connected to UDP stream", source_id);
                // process_stream will decode, scale to RGB24, and call callbacks
                let result = process_stream(source_id, &mut ictx, callbacks, stop_signal.clone());
                
                // Explicitly drop the input context to ensure UDP socket is released
                drop(ictx);
                log_debug!("[Source {}] FFmpeg input context dropped, UDP port released", source_id);
                
                return result;
            }
            Err(e) => {
                last_error = Some(e);
                log_error!("[Source {}] Connection attempt {} failed: {}", source_id, attempt, last_error.as_ref().unwrap());
                if attempt < 3 {
                    std::thread::sleep(std::time::Duration::from_secs(2));
                }
            }
        }
    }
    
    // UPDATED: Error message
    Err(last_error.unwrap()).context(format!("Failed to open UDP stream after 3 attempts"))
}

// This function decodes the mpegts/h264 stream and scales it to RGB24
fn process_stream(
    source_id: i32,
    ictx: &mut ffmpeg::format::context::Input,
    callbacks: Callbacks,
    stop_signal: Arc<AtomicBool>,
) -> Result<()> {
    let input = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .context("No video stream found")?;
    
    let video_stream_index = input.index();

    // Get FPS from stream
    let fps = input.avg_frame_rate();
    let fps_float = if fps.denominator() != 0 {
        fps.numerator() as f64 / fps.denominator() as f64
    } else {
        0.0
    };
    
    // UPDATED: log_debug uses static log level
    log_debug!("[Source {}] Found video stream, attempting to decode...", source_id);

    let context_decoder = ffmpeg::codec::context::Context::from_parameters(input.parameters())
        .context("Failed to create codec context")?;
    
    let mut decoder = context_decoder
        .decoder()
        .video()
        .context("Failed to create video decoder")?;

    log_debug!("[Source {}] Waiting for first frame from stream...", source_id);
    
    let mut first_frame = ffmpeg::util::frame::video::Video::empty();
    let mut got_first_frame = false;
    
    for (stream, packet) in ictx.packets() {
        if stop_signal.load(Ordering::Relaxed) {
            log_info!("[Source {}] Stop signal received during initial decode, exiting", source_id);
            break;
        }
        if stream.index() == video_stream_index {
            
            if decoder.send_packet(&packet).is_ok() {
                if decoder.receive_frame(&mut first_frame).is_ok() {
                    got_first_frame = true;
                    break;
                }
            }
        }
    }
    
    if !got_first_frame {
        anyhow::bail!("Could not decode first frame from stream");
    }
    
    let width = first_frame.width();
    let height = first_frame.height();
    // This format will be YUV420P (or similar), which is correct for the stream
    let format = first_frame.format();
    
    log_debug!("[Source {}] Got response from stream ({}x{}), {:.2} FPS, format: {:?}", 
             source_id, width, height, fps_float, format);
    
    if width == 0 || height == 0 {
        anyhow::bail!("Invalid frame dimensions from ffmpeg: {}x{}", width, height);
    }

    // Create scaler to convert from stream format (e.g., YUV420P) to RGB24
    let mut scaler = ffmpeg::software::scaling::context::Context::get(
        format, // Input format from stream
        width,
        height,
        ffmpeg::format::Pixel::RGB24,  // Output format: rgb24
        width,
        height,
        ffmpeg::software::scaling::Flags::BILINEAR,
    )
    .context("Failed to create scaler")?;
    
    // Process the first frame we already decoded
    let mut rgb_frame = ffmpeg::util::frame::video::Video::empty();
    if scaler.run(&first_frame, &mut rgb_frame).is_ok() {
        let pts = first_frame.pts().unwrap_or(0);
        let data_ptr = rgb_frame.data(0).as_ptr();
        // Callback with RGB24 frame data
        (callbacks.source_frames)(source_id, data_ptr, width as i32, height as i32, pts as u64);
        
        log_info!("[Source {}] Started receiving frames ({}x{}), PTS: {}", 
                     source_id, width, height, pts);
    }

    let mut last_pts: Option<i64> = first_frame.pts();

    // Continue processing remaining frames
    for (stream, packet) in ictx.packets() {
        if stop_signal.load(Ordering::Relaxed) {
            log_info!("[Source {}] Stop signal received, exiting stream loop", source_id);
            break;
        }
        if stream.index() == video_stream_index {
            
            if let Err(e) = decoder.send_packet(&packet) {
                log_error!("[Source {}] Error sending packet: {}", source_id, e);
                break;
            }

            let mut decoded_frame = ffmpeg::util::frame::video::Video::empty();
            
            while decoder.receive_frame(&mut decoded_frame).is_ok() {
                
                let mut rgb_frame = ffmpeg::util::frame::video::Video::empty();
                
                // Scale to RGB24
                if let Err(e) = scaler.run(&decoded_frame, &mut rgb_frame) {
                    log_error!("[Source {}] Scaling error: {}", source_id, e);
                    continue;
                }

                // Get PTS - raw value from stream
                let pts = decoded_frame.pts().unwrap_or(0);
                
                if let Some(last) = last_pts {
                    if pts <= last && pts != 0 {
                        log_debug!("[Source {}] PTS issue detected (last: {}, current: {})", 
                                source_id, last, pts);
                    }
                }
                last_pts = Some(pts);

                let width = rgb_frame.width() as i32;
                let height = rgb_frame.height() as i32;
                let data_ptr = rgb_frame.data(0).as_ptr();

                // Call frames callback with RGB24 data
                (callbacks.source_frames)(source_id, data_ptr, width, height, pts as u64);
            }
        }
    }

    // If we exit the loop, stream ended
    log_info!("[Source {}] Stream ended", source_id);
    (callbacks.source_stopped)(source_id);

    Ok(())
}

/// Initialize FFmpeg library (call once at startup)
pub fn init_ffmpeg() -> Result<()> {
    unsafe {
        // Setting a less verbose log level
        ffmpeg_next::sys::av_log_set_level(ffmpeg_next::sys::AV_LOG_ERROR);
    }
    
    ffmpeg::init().context("Failed to initialize FFmpeg")?;
    
    log_info!("FFmpeg Initialized successfully");
    Ok(())
}

