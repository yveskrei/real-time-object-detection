use libc::{c_int, c_ulonglong, c_char, c_void};

// Custom modules
pub mod player_proxy;
pub mod stream;

// C Types
pub type FramesCallback = extern "C" fn(source_id: c_int, frame: *const u8, width: c_int, height: c_int, timestamp: c_ulonglong);
pub type SourceStoppedCallback = extern "C" fn(source_id: c_int);
pub type SourceErrorCallback = extern "C" fn(source_id: c_int);
pub type SourceNameCallback = extern "C" fn(source_id: c_int, source_name: *const c_char);
pub type SourceStatusCallback = extern "C" fn(source_id: c_int, source_status: c_int); 

#[unsafe(no_mangle)]
pub extern "C" fn SetCallbacks(
    frames: FramesCallback, 
    source_stopped: SourceStoppedCallback, 
    source_error: SourceErrorCallback,
    source_name: SourceNameCallback,
    source_status: SourceStatusCallback
) {

}

#[unsafe(no_mangle)]
pub extern "C" fn InitMultipleSources(source_ids: *const c_int, size: c_int) {

}

#[unsafe(no_mangle)]
pub extern "C" fn PostResults(source_id: c_int, result_json: *const c_char) -> c_int {
    0 as c_int
}

#[unsafe(no_mangle)]
pub extern "C" fn FreeCPtr(ptr: *const c_void) {
    
}