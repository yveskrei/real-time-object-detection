use anyhow::{Context, Result};
use ffmpeg_next as ffmpeg;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::task::JoinHandle;
use tokio::time::sleep;
use reqwest::Url;

use crate::player_proxy::{PlayerSession, RawStreamInfo};
use crate::TOKIO_RUNTIME;
use crate::{SourceFramesCallback, SourceStoppedCallback, SourceNameCallback, SourceStatusCallback};
use crate::{log_info, log_error, log_debug}; // log_debug now uses static state

// Stream timeout constant
const STREAM_TIMEOUT: Duration = Duration::from_secs(10);

// Logging level for C FFI
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogLevel {
    Regular = 0,
    Debug = 1,
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
    // This is the static log level you wanted
    pub log_level: Mutex<LogLevel>,
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

lazy_static::lazy_static! {
    pub static ref STREAM_MANAGER: Arc<StreamManager> = Arc::new(StreamManager::new());
}

impl StreamManager {
    fn new() -> Self {
        Self {
            streams: Mutex::new(HashMap::new()),
            callbacks: Mutex::new(None),
            player_session: PlayerSession::default(),
            log_level: Mutex::new(LogLevel::Regular),
        }
    }

    pub fn set_log_level(&self, level: LogLevel) {
        *self.log_level.lock().unwrap() = level;
        log_info!("Log level set to: {:?}", level);
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
        let manager = Arc::clone(&STREAM_MANAGER);
        
        let handle = TOKIO_RUNTIME.spawn(async move {
            // UPDATED: log_debug no longer needs 'manager'
            log_debug!("[Source {}] Starting monitor task", source_id);
            
            // Get host from base_url. Assumes backend is on same host.
            let host = match Url::parse(manager.player_session.base_url()) {
                Ok(url) => url.host_str().unwrap_or("127.0.0.1").to_string(),
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
                        // CHANGED: Pass host
                        if let Err(e) = manager.consume_stream(source_id, raw_stream_info.clone(), host.clone(), callbacks).await {
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
        let response = reqwest::get(&url).await?;
        let info: VideoInfo = response.json().await?;
        Ok(info)
    }

    async fn consume_stream(
        &self,
        source_id: i32,
        stream_info: RawStreamInfo,
        host: String,
        callbacks: Callbacks,
    ) -> Result<()> {
        let session = self.player_session.clone();
        
        // Spawn a task to periodically check if stream is still active on backend
        let mut keepalive_handle = tokio::spawn(async move {
            loop {
                sleep(Duration::from_secs(15)).await;
                
                match session.get_stream_status(source_id).await {
                    Ok(status) => {
                        if !status.is_streaming {
                            log_info!("[Source {}] Backend reports stream stopped, triggering reconnect", source_id);
                            return; // Stream is not active anymore
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
            // CHANGED: Pass stream_info and host
            // Note: 'manager' is no longer needed here for logging
            if let Err(e) = decode_stream(source_id, stream_info, host, callbacks) {
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
                // Keepalive detected stream stopped, this will cause decode to error out eventually
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
) -> Result<()> {
    // UPDATED: Connect to UDP stream
    let connection_url = format!("udp://{}:{}", host, stream_info.port);

    log_info!("[Source {}] Connecting to UDP stream: {}", source_id, connection_url);

    // UPDATED: Removed rawvideo options, added options for UDP/MPEGTS
    let mut input_opts = ffmpeg::Dictionary::new();
    input_opts.set("analyzeduration", "500000"); // 0.5s
    input_opts.set("probesize", "500000"); // 500KB
    input_opts.set("fflags", "nobuffer");
    input_opts.set("flags", "low_delay");
    // We let FFmpeg auto-detect format (mpegts) and codec (h264)

    let mut last_error = None;
    for attempt in 1..=3 {
        log_info!("[Source {}] Connection attempt {}/3", source_id, attempt);

        // We pass options, but don't force rawvideo
        match ffmpeg::format::input_with_dictionary(&connection_url, input_opts.clone()) {
            Ok(mut ictx) => {
                log_info!("[Source {}] Successfully connected to UDP stream", source_id);
                // process_stream will decode, scale to RGB24, and call callbacks
                return process_stream(source_id, &mut ictx, callbacks);
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
    
    // UPDATED: log_debug uses static log level
    if *STREAM_MANAGER.log_level.lock().unwrap() == LogLevel::Debug {
        log_info!("[Source {}] Got response from stream ({}x{}), {:.2} FPS, format: {:?}", 
                 source_id, width, height, fps_float, format);
    }
    
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

/// Check if callbacks are set
pub fn are_callbacks_set() -> bool {
    STREAM_MANAGER.are_callbacks_set()
}

/// Start monitoring and consuming a single stream
pub fn start_stream(source_id: i32) -> Result<()> {
    STREAM_MANAGER.init_sources(vec![source_id]);
    Ok(())
}

/// Start monitoring and consuming multiple streams
pub fn start_streams(source_ids: Vec<i32>, log_level: LogLevel) -> Result<()> {
    STREAM_MANAGER.set_log_level(log_level);
    STREAM_MANAGER.init_sources(source_ids);
    Ok(())
}

/// Set the callbacks for stream events
pub fn set_callbacks(
    source_frames: SourceFramesCallback,
    source_stopped: SourceStoppedCallback,
    source_name: SourceNameCallback,
    source_status: SourceStatusCallback,
) {
    STREAM_MANAGER.set_callbacks(source_frames, source_stopped, source_name, source_status);
}