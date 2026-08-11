//! FFI boundary to the compiled `libclient_video.so`.
//!
//! The `.so` is `dlopen`ed once into [`ClientVideo`], which lives on [`crate::services::Services`].
//! Frames arrive on the library's decoder threads through `_source_frames_callback` and are
//! handed to the runtime; detections go back through [`ClientVideo::populate_bboxes`].

use anyhow::{Context, Result};
use libc::{c_char, c_int, c_longlong, c_uint, c_void};
use libloading::{Library, Symbol};
use serde_json::json;
use std::ffi::CString;
use std::sync::Arc;

// Custom modules
pub mod utils;
use crate::client_video::utils::{Ownership, get_c_array, get_c_string};
use crate::services;
use crate::utils::config::AppConfig;
// Re-exported so callers get the API and its types from one module.
pub use crate::client_video::utils::{Detection, RawFrame, ResultBBOX};

// Path to the compiled library, relative to the process working directory
const SO_PATH: &str = "secrets/libclient_video.so";

// C ABI types
// Host callbacks - the library calls into these
pub type SourceFramesCb = extern "C" fn(
    source_id: c_uint,
    frame: *const u8,
    width: c_int,
    height: c_int,
    pts: c_longlong,
);
pub type SourceMetadataCb = extern "C" fn(
    source_id: c_uint,
    source_name: *const c_char,
    width: c_int,
    height: c_int,
    fps: c_int,
);
pub type SourceStatusCb = extern "C" fn(source_id: c_uint, status: c_int);
pub type PostResultsCb = extern "C" fn(
    source_id: c_uint,
    results_count: c_int,
    results_ids: *const *const c_char,
    results_timestamps: *const c_longlong,
);

// Exported library functions - we call into these
pub type SetCallbacksFn = extern "C" fn(
    frames: SourceFramesCb,
    metadata: SourceMetadataCb,
    status: SourceStatusCb,
    results: PostResultsCb,
);
pub type SetSettingsFn = extern "C" fn(run_mode: c_int);
pub type InitSourcesFn = extern "C" fn(source_ids: *const c_uint, size: c_int);
pub type StopSourcesFn = extern "C" fn(source_ids: *const c_uint, size: c_int);
pub type PostResultsFn = extern "C" fn(
    source_id: c_uint,
    results_count: c_int,
    results_ids: *const *const c_char,
    result_body: *const c_char,
) -> c_int;
pub type FreeCPtrFn = extern "C" fn(ptr: *const c_void);

/// Library log level
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunMode {
    Regular = 0,
    Debug = 1,
}

impl RunMode {
    pub fn as_i32(self) -> c_int {
        self as c_int
    }
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceStatus {
    Idle = 0,
    Initializing = 1,
    Streaming = 2,
    Terminating = 3,
}

impl SourceStatus {
    pub fn as_i32(self) -> i32 {
        self as i32
    }

    fn from_i32(value: c_int) -> Self {
        match value {
            1 => SourceStatus::Initializing,
            2 => SourceStatus::Streaming,
            3 => SourceStatus::Terminating,
            _ => SourceStatus::Idle,
        }
    }
}

pub struct ClientVideo {
    library: Library,
}

impl ClientVideo {
    pub fn new() -> Result<Self> {
        // Load dynamic library
        let library =
            unsafe { Library::new(SO_PATH).with_context(|| format!("Error loading '{SO_PATH}'"))? };

        Ok(Self { library })
    }

    /// Boots the library: register the callbacks, then apply the run mode.
    ///
    /// Order matters - `SetCallbacks` boots the library's global state, and
    /// `SetSettings` is a no-op until that exists.
    pub async fn init_state(&self, run_mode: RunMode) -> Result<()> {
        ClientVideo::set_callbacks()
            .await
            .context("Error setting Client Video callbacks")?;

        ClientVideo::set_settings(run_mode)
            .await
            .context("Error setting Client Video settings")?;

        Ok(())
    }

    /// Starts a decoder per configured source id. Requires [`ClientVideo::init_state`]
    /// to have run first - frames begin arriving as soon as this returns.
    pub async fn init_sources(&self, app_config: &AppConfig) -> Result<()> {
        let source_ids: Vec<c_uint> = app_config
            .sources_config()
            .sources
            .keys()
            .copied()
            .collect();
        ClientVideo::start_sources(source_ids)
            .await
            .context("Error starting Client Video sources")?;

        Ok(())
    }

    // Library function
    async fn set_callbacks() -> Result<()> {
        tokio::task::spawn_blocking(move || -> Result<()> {
            let services = services::get_services()?;

            unsafe {
                let lib_set_callbacks: Symbol<SetCallbacksFn> = services
                    .client_video()
                    .library()
                    .get(b"SetCallbacks")
                    .context("Cannot get 'SetCallbacks' function")?;

                lib_set_callbacks(
                    ClientVideo::_source_frames_callback,
                    ClientVideo::_source_metadata_callback,
                    ClientVideo::_source_status_callback,
                    ClientVideo::_post_results_callback,
                )
            }

            Ok(())
        })
        .await
        .context("Error trying to set callbacks in video client")?
        .context("Error setting callbacks in video client")?;

        Ok(())
    }

    /// Sets the library run mode (log level)
    async fn set_settings(run_mode: RunMode) -> Result<()> {
        tokio::task::spawn_blocking(move || -> Result<()> {
            let services = services::get_services()?;

            unsafe {
                let lib_set_settings: Symbol<SetSettingsFn> = services
                    .client_video()
                    .library()
                    .get(b"SetSettings")
                    .context("Cannot get 'SetSettings' function")?;

                lib_set_settings(run_mode.as_i32())
            }

            Ok(())
        })
        .await
        .context("Error trying to set settings in video client")?
        .context("Error setting settings in video client")?;

        Ok(())
    }

    async fn start_sources(source_ids: Vec<c_uint>) -> Result<()> {
        if source_ids.is_empty() {
            anyhow::bail!("No valid sources are avaliable");
        }

        tokio::task::spawn_blocking(move || -> Result<()> {
            let services = services::get_services()?;
            let client_video = services.client_video();

            unsafe {
                let lib_init_sources: Symbol<InitSourcesFn> = client_video
                    .library()
                    .get(b"InitSources")
                    .context("Cannot get 'InitSources' function")?;

                lib_init_sources(source_ids.as_ptr(), source_ids.len() as c_int)
            }

            Ok(())
        })
        .await
        .context("Error trying to initiate sources in video client")?
        .context("Error initiating source in video client")?;

        Ok(())
    }

    #[allow(dead_code)]
    async fn stop_sources(source_ids: Vec<c_uint>) -> Result<()> {
        if source_ids.is_empty() {
            anyhow::bail!("No valid sources are avaliable");
        }

        tokio::task::spawn_blocking(move || -> Result<()> {
            let services = services::get_services()?;
            let client_video = services.client_video();

            unsafe {
                let lib_stop_sources: Symbol<StopSourcesFn> = client_video
                    .library()
                    .get(b"StopSources")
                    .context("Cannot get 'StopSources' function")?;

                lib_stop_sources(source_ids.as_ptr(), source_ids.len() as c_int)
            }

            Ok(())
        })
        .await
        .context("Error trying to stop sources in video client")?
        .context("Error stopping source in video client")?;

        Ok(())
    }

    /// Posts detections for a frame's source back to the backend
    ///
    /// Each bbox's `id` is sent both in the results-id array and inside the JSON body,
    /// so the backend can correlate what it stored.
    pub async fn populate_bboxes(frame: &RawFrame, bboxes: &[ResultBBOX]) -> Result<()> {
        let source_id = frame.source_id;

        // Format BBOXes output for sending it back to the client
        let bboxes_json: Vec<_> = bboxes
            .iter()
            .map(|bbox| {
                // Get bbox corners - indexes of pixels in frame, as if it was a 1d array
                let (top_left_corner, bottom_right_corner) = bbox.corners_coordinates(frame);

                json!({
                    "id": bbox.id,
                    "pts": frame.pts,
                    "top_left_corner": top_left_corner,
                    "bottom_right_corner": bottom_right_corner,
                    "class_name": bbox.class_name(),
                    "confidence": bbox.score
                })
            })
            .collect();

        let bboxes_result_json = json!({
            "stream_id": source_id,
            "bboxes": bboxes_json
        })
        .to_string();

        // Per-bbox ids for the results-id array
        let results_ids: Vec<CString> = bboxes
            .iter()
            .map(|bbox| CString::new(bbox.id.as_str()))
            .collect::<std::result::Result<_, _>>()
            .context("A bbox id contains an interior NUL")?;
        let results_body =
            CString::new(bboxes_result_json).context("Error converting bboxes to C string")?;

        tokio::task::spawn_blocking(move || -> Result<()> {
            let services = services::get_services()?;

            // The library copies these synchronously, so we only need to keep them alive
            // across the call - Rust drops them afterwards. Nothing is handed off.
            let id_ptrs: Vec<*const c_char> = results_ids.iter().map(|id| id.as_ptr()).collect();

            unsafe {
                let lib_post_results: Symbol<PostResultsFn> = services
                    .client_video()
                    .library()
                    .get(b"PostResults")
                    .context("Cannot get 'PostResults' function")?;

                let result = lib_post_results(
                    source_id as c_uint,
                    results_ids.len() as c_int,
                    id_ptrs.as_ptr(),
                    results_body.as_ptr(),
                );

                // Check whether posting failed
                if result != 0 {
                    anyhow::bail!("Failed to post bboxes, got code {}", result)
                }
            }

            Ok(())
        })
        .await
        .context("Error trying to post bboxes")?
        .context("Error posting bboxes")?;

        Ok(())
    }

    // Callbacks
    extern "C" fn _source_frames_callback(
        source_id: c_uint,
        frame: *const u8,
        width: c_int,
        height: c_int,
        pts: c_longlong,
    ) {
        let width = width as u32;
        let height = height as u32;
        let frame_size = (width as usize) * (height as usize) * 3;

        // The buffer is borrowed - valid only for this call - so copy it now
        let Ok(rgb_frame) = get_c_array(frame, frame_size, Ownership::Borrowed) else {
            tracing::error!(source_id = source_id, "RGB Frame is invalid");
            return;
        };

        let Ok(services_object) = services::get_services() else {
            tracing::error!(
                source_id = source_id,
                "Cannot process frame on application runtime"
            );
            return;
        };

        let raw_frame = RawFrame {
            source_id,
            data: rgb_frame,
            width,
            height,
            pts,
        };

        // Spawn task on our runtime to prevent blocking the C callback
        let services_task = Arc::clone(&services_object);
        services_object.runtime().spawn(async move {
            match services_task
                .source_processors()
                .read()
                .await
                .processor(source_id)
            {
                Err(e) => {
                    tracing::error!(
                        error = e.to_string(),
                        source_id = source_id,
                        "Source processor is not available"
                    )
                }
                Ok(processor) => {
                    processor.add_to_queue(raw_frame).await;
                }
            }
        });
    }

    extern "C" fn _source_metadata_callback(
        source_id: c_uint,
        source_name: *const c_char,
        width: c_int,
        height: c_int,
        fps: c_int,
    ) {
        let source_name =
            get_c_string(source_name, Ownership::Owned).unwrap_or("UNKNOWN".to_string());

        tracing::info!(
            source_id = source_id,
            source_name = source_name,
            width = width,
            height = height,
            fps = fps,
            "Got source metadata"
        );
    }

    extern "C" fn _source_status_callback(source_id: c_uint, source_status: c_int) {
        let source_status = SourceStatus::from_i32(source_status);

        tracing::info!(source_id = source_id, ?source_status, "Got source status");
    }

    #[allow(unused_variables)]
    extern "C" fn _post_results_callback(
        source_id: c_uint,
        results_count: c_int,
        results_ids: *const *const c_char,
        results_timestamps: *const c_longlong,
    ) {
        let count = results_count.max(0) as usize;

        // All library-owned: N + 2 frees. An unreadable id becomes an empty string rather
        // than being dropped, since `ids` and `timestamps` are index-aligned.
        let ids: Vec<String> = get_c_array(results_ids, count, Ownership::Owned)
            .unwrap_or_default()
            .into_iter()
            .map(|ptr| get_c_string(ptr, Ownership::Owned).unwrap_or_default())
            .collect();
        let timestamps =
            get_c_array(results_timestamps, count, Ownership::Owned).unwrap_or_default();

        // tracing::info!(
        //     source_id=source_id,
        //     results_count=results_count,
        //     ids=?ids,
        //     timestamps=?timestamps,
        //     "Got post results response"
        // );
    }
}

impl ClientVideo {
    fn library(&self) -> &Library {
        &self.library
    }
}

// Helper functions
/// Hands a library-owned pointer back to the `.so`. Callers go through `Ownership`.
pub(crate) fn free_c_ptr<T>(ptr: *const T) -> Result<()> {
    let services = services::get_services()?;

    unsafe {
        let lib_free_c_ptr: Symbol<FreeCPtrFn> = services
            .client_video()
            .library()
            .get(b"FreeCPtr")
            .context("Cannot get 'FreeCPtr' function")?;

        // Call library function
        lib_free_c_ptr(ptr as *const c_void);
    }

    Ok(())
}
