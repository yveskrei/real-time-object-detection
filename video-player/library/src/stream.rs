use anyhow::{Context, Result};
use ffmpeg_next as ffmpeg;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::task::JoinHandle;
use tokio::time::sleep;

use crate::player_proxy::PlayerSession;
use crate::TOKIO_RUNTIME;
use crate::{FramesCallback, SourceStoppedCallback, SourceNameCallback, SourceStatusCallback};
use crate::{log_info, log_error, log_debug};

// Stream timeout constant
const STREAM_TIMEOUT: Duration = Duration::from_secs(5);

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
    pub log_level: Mutex<LogLevel>,
}

#[derive(Clone, Copy)]
struct Callbacks {
    frames: FramesCallback,
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
        frames: FramesCallback,
        source_stopped: SourceStoppedCallback,
        source_name: SourceNameCallback,
        source_status: SourceStatusCallback,
    ) {
        let callbacks = Callbacks {
            frames,
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
            log_debug!(manager, "[Source {}] Starting monitor task", source_id);
            
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
                    // cb_lock is dropped here automatically at end of scope
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

                        let stream_url = match status.stream_url {
                            Some(url) => url,
                            None => {
                                log_error!("[Source {}] No stream URL available", source_id);
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

                        log_info!("[Source {}] Stream active, connecting to {}", source_id, stream_url);
                        (callbacks.source_status)(source_id, SourceStatus::Ok as i32);

                        // Start consuming stream
                        if let Err(e) = manager.consume_stream(source_id, &stream_url, callbacks).await {
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
                log_debug!(manager, "[Source {}] Retrying in {:?}...", source_id, STREAM_TIMEOUT);
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
        stream_url: &str,
        callbacks: Callbacks,
    ) -> Result<()> {
        let stream_url = stream_url.to_string();
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
        
        let manager = Arc::clone(&STREAM_MANAGER);
        // Spawn blocking task for FFmpeg operations
        let mut decode_handle = tokio::task::spawn_blocking(move || {
            if let Err(e) = decode_stream(source_id, &stream_url, callbacks, manager) {
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

fn decode_stream(source_id: i32, stream_url: &str, callbacks: Callbacks, manager: Arc<StreamManager>) -> Result<()> {
    // Set input options for UDP stream
    let mut input_opts = ffmpeg::Dictionary::new();
    input_opts.set("analyzeduration", "10000000"); // 10 seconds
    input_opts.set("probesize", "50000000"); // 50MB
    input_opts.set("fflags", "nobuffer"); // Reduce latency
    input_opts.set("max_delay", "0"); // Minimum delay
    
    // Convert Duration to microseconds for FFmpeg
    let timeout_micros = (STREAM_TIMEOUT.as_secs() * 1_000_000).to_string();
    input_opts.set("timeout", &timeout_micros);
    input_opts.set("rw_timeout", &timeout_micros);
    
    let mut ictx = ffmpeg::format::input_with_dictionary(&stream_url, input_opts)
        .context("Failed to open stream")?;

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
    
    log_debug!(manager, "[Source {}] Found video stream, attempting to decode...", source_id);

    let context_decoder = ffmpeg::codec::context::Context::from_parameters(input.parameters())
        .context("Failed to create codec context")?;
    
    let mut decoder = context_decoder
        .decoder()
        .video()
        .context("Failed to create video decoder")?;

    // Wait a bit and try to decode first packet to get actual frame parameters
    log_debug!(manager, "[Source {}] Waiting for first frame to determine dimensions...", source_id);
    
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
    let format = first_frame.format();
    
    let is_debug = *manager.log_level.lock().unwrap() == LogLevel::Debug;
    
    if is_debug {
        log_info!("[Source {}] Got response from stream ({}x{}), {:.2} FPS, format: {:?}", 
                 source_id, width, height, fps_float, format);
    }
    
    if width == 0 || height == 0 {
        anyhow::bail!("Invalid frame dimensions: {}x{}", width, height);
    }

    let mut scaler = ffmpeg::software::scaling::context::Context::get(
        format,
        width,
        height,
        ffmpeg::format::Pixel::RGB24,
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
        (callbacks.frames)(source_id, data_ptr, width as i32, height as i32, pts as u64);
        
        log_info!("[Source {}] Started receiving frames ({}x{}), PTS: {}", 
                     source_id, width, height, pts);
    }

    let mut last_pts: Option<i64> = first_frame.pts();

    for (stream, packet) in ictx.packets() {
        if stream.index() == video_stream_index {
            if let Err(e) = decoder.send_packet(&packet) {
                log_error!("[Source {}] Error sending packet: {}", source_id, e);
                continue;
            }

            let mut decoded_frame = ffmpeg::util::frame::video::Video::empty();
            
            while decoder.receive_frame(&mut decoded_frame).is_ok() {
                let mut rgb_frame = ffmpeg::util::frame::video::Video::empty();
                
                if let Err(e) = scaler.run(&decoded_frame, &mut rgb_frame) {
                    log_error!("[Source {}] Scaling error: {}", source_id, e);
                    continue;
                }

                // Get PTS - raw value from stream
                let pts = decoded_frame.pts().unwrap_or(0);
                
                // Check for pts progression issues
                if let Some(last) = last_pts {
                    if pts <= last {
                        // Stream might have looped or issues
                        log_debug!(manager, "[Source {}] PTS reset detected (last: {}, current: {})", 
                                source_id, last, pts);
                    }
                }
                last_pts = Some(pts);

                let width = rgb_frame.width() as i32;
                let height = rgb_frame.height() as i32;
                let data_ptr = rgb_frame.data(0).as_ptr();

                // Call frames callback with raw PTS
                (callbacks.frames)(source_id, data_ptr, width, height, pts as u64);
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
    // Set log level to quiet to suppress all FFmpeg logs
    unsafe {
        ffmpeg_next::sys::av_log_set_level(ffmpeg_next::sys::AV_LOG_QUIET);
    }
    
    ffmpeg::init().context("Failed to initialize FFmpeg")?;
    
    // Set again after init to be sure
    ffmpeg::log::set_level(ffmpeg::log::Level::Quiet);
    
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
    frames: FramesCallback,
    source_stopped: SourceStoppedCallback,
    source_name: SourceNameCallback,
    source_status: SourceStatusCallback,
) {
    STREAM_MANAGER.set_callbacks(frames, source_stopped, source_name, source_status);
}