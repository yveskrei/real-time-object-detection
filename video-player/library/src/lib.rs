use libc::{c_int, c_ulonglong, c_char, c_void};
use std::ffi::CStr;
use std::slice;
use std::sync::{Mutex, OnceLock};
use tokio::runtime::Runtime;

// Custom modules
pub mod player_proxy;
pub mod stream;

// Logging level for C FFI
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogLevel {
    Regular = 0,
    Debug = 1,
}

// Global log level
pub static LOG_LEVEL: Mutex<LogLevel> = Mutex::new(LogLevel::Regular);

// Global Tokio runtime
pub static TOKIO_RUNTIME: OnceLock<Runtime> = OnceLock::new();

pub fn get_runtime() -> &'static Runtime {
    TOKIO_RUNTIME.get_or_init(|| Runtime::new().expect("Failed to create Tokio runtime"))
}

pub fn set_log_level(level: LogLevel) {
    *LOG_LEVEL.lock().unwrap() = level;
    log_info!("Log level set to: {:?}", level);
}

// Logging macros - used globally
#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {{
        println!("[CLIENT_STREAM][INFO] {}", format!($($arg)*))
    }};
}

#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {{
        println!("[CLIENT_STREAM][ERR] {}", format!($($arg)*))
    }};
}

#[macro_export]
macro_rules! log_debug {
    ($($arg:tt)*) => {{
        if *$crate::LOG_LEVEL.lock().unwrap() == $crate::LogLevel::Debug {
            println!("[CLIENT_STREAM][DBG] {}", format!($($arg)*))
        }
    }};
}

// C Types
pub type SourceFramesCallback = extern "C" fn(source_id: c_int, frame: *const u8, width: c_int, height: c_int, pts: c_ulonglong);
pub type SourceStoppedCallback = extern "C" fn(source_id: c_int);
pub type SourceNameCallback = extern "C" fn(source_id: c_int, source_name: *const c_char);
pub type SourceStatusCallback = extern "C" fn(source_id: c_int, source_status: c_int);

#[no_mangle]
pub extern "C" fn SetCallbacks(
    source_frames: SourceFramesCallback,
    source_stopped: SourceStoppedCallback,
    source_name: SourceNameCallback,
    source_status: SourceStatusCallback,
) {
    log_info!("SetCallbacks called");
    stream::get_stream_manager().set_callbacks(source_frames, source_stopped, source_name, source_status);
}

#[no_mangle]
pub extern "C" fn InitMultipleSources(source_ids: *const c_int, size: c_int, log_level: c_int) {
    log_info!("InitMultipleSources called with {} sources, log_level: {}", size, log_level);
    
    if source_ids.is_null() || size <= 0 {
        log_error!("Invalid parameters: null pointer or invalid size");
        return;
    }

    // Check if callbacks are set
    if !stream::get_stream_manager().are_callbacks_set() {
        log_error!("Callbacks not set. Call SetCallbacks before InitMultipleSources");
        return;
    }

    // Convert C log level to Rust enum
    let log_level = match log_level {
        0 => LogLevel::Regular,
        1 => LogLevel::Debug,
        _ => {
            log_error!("Invalid log level: {}, defaulting to Regular", log_level);
            LogLevel::Regular
        }
    };

    // Convert C array to Rust Vec
    let ids = unsafe {
        slice::from_raw_parts(source_ids, size as usize)
            .iter()
            .map(|&id| id as i32)
            .collect::<Vec<i32>>()
    };

    log_info!("Initializing {} sources: {:?}", ids.len(), ids);
    
    // Initialize FFmpeg
    if let Err(e) = stream::init_ffmpeg() {
        log_error!("Failed to initialize FFmpeg: {}", e);
        return;
    }
    
    // Set the global log level
    set_log_level(log_level);
    
    // Start streams
    stream::get_stream_manager().init_sources(ids);

    // Keeping the library alive by sleeping in this thread
    get_runtime().block_on(async {
        // Keep the runtime alive indefinitely
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await;
        }
    });
}

#[no_mangle]
#[allow(unused_variables)]
pub extern "C" fn PostResults(source_id: c_int, result_json: *const c_char) -> c_int {
    if result_json.is_null() {
        log_error!("PostResults: null JSON pointer");
        return -1;
    }

    let json_str = unsafe {
        match CStr::from_ptr(result_json).to_str() {
            Ok(s) => s,
            Err(e) => {
                log_error!("PostResults: invalid UTF-8 in JSON: {}", e);
                return -1;
            }
        }
    };
    
    // Spawn async task to post results
    get_runtime().spawn(async move {
        match post_results_async(json_str.to_string()).await {
            Ok(_) => log_info!("PostResults: Successfully posted bboxes"),
            Err(e) => log_error!("PostResults: Failed to post bboxes: {}", e),
        }
    });

    // Return immediately (non-blocking)
    0
}

async fn post_results_async(json_str: String) -> anyhow::Result<()> {
    use anyhow::Context;
    
    let session = player_proxy::PlayerSession::new()?;
    let url = format!("{}/bboxes/", session.base_url());
    
    // Parse JSON to validate it's valid JSON
    let _: serde_json::Value = serde_json::from_str(&json_str)
        .context("Invalid JSON format")?;
    
    let client = reqwest::Client::new();
    let response = client
        .post(&url)
        .header("Content-Type", "application/json")
        .body(json_str)
        .send()
        .await
        .context("Failed to send POST request")?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        anyhow::bail!("Backend rejected bboxes (status {}): {}", status, error_text);
    }
    
    Ok(())
}

#[no_mangle]
pub extern "C" fn FreeCPtr(ptr: *const c_void) {
    if ptr.is_null() {
        log_error!("FreeCPtr: attempted to free null pointer");
        return;
    }

    unsafe {
        // Reconstruct the CString and drop it
        let _ = std::ffi::CString::from_raw(ptr as *mut c_char);
    }
}