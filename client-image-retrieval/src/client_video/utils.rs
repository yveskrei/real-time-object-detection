//! Data types crossing the client video boundary, and the C-pointer readers that
//! materialize them.

use std::borrow::Cow;
use std::ffi::CStr;

use anyhow::Result;
use libc::c_char;

// Custom modules
use crate::client_video::free_c_ptr;

// Pointer ownership
/// Whether a pointer from the library must be handed back to `FreeCPtr` after reading.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ownership {
    Borrowed,
    Owned,
}

impl Ownership {
    fn release<T>(self, ptr: *const T) {
        if self == Ownership::Borrowed {
            return;
        }

        if let Err(e) = free_c_ptr(ptr) {
            tracing::warn!(error = %e, "failed to free a library-owned pointer");
        }
    }
}

// Data crossing to the user
/// One decoded frame, tightly-packed RGB24 (stride = width*3). `pts` is 90 kHz.
#[derive(Clone, Debug)]
pub struct RawFrame {
    pub source_id: u32,
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub pts: i64,
}

/// A single detection as it comes out of the model.
///
/// Deliberately `Copy` and id-free: decoding and NMS churn through thousands of
/// candidates per frame, and only the survivors are worth giving an identity to.
#[derive(Clone, Copy, Debug)]
pub struct Detection {
    pub bbox: [f32; 4],
    pub class: u32,
    pub score: f32,
}

impl Detection {
    /// Promotes a surviving detection into a result, giving it its backend id.
    pub fn into_result(self, id: String) -> ResultBBOX {
        ResultBBOX {
            id,
            bbox: self.bbox,
            class: self.class,
            score: self.score,
        }
    }
}

/// A single detection we report. `id` is a caller-supplied UUID, echoed back by the backend.
#[derive(Clone, Debug)]
pub struct ResultBBOX {
    pub id: String,
    pub bbox: [f32; 4],
    pub class: u32,
    pub score: f32,
}

impl ResultBBOX {
    pub fn class_name(&self) -> Cow<'static, str> {
        match self.class {
            0 => Cow::Borrowed("person"),
            1 => Cow::Borrowed("bicycle"),
            2 => Cow::Borrowed("car"),
            3 => Cow::Borrowed("motorcycle"),
            4 => Cow::Borrowed("airplane"),
            5 => Cow::Borrowed("bus"),
            _ => Cow::Owned(self.class.to_string()),
        }
    }

    /// The two flat pixel indices (`y*width + x`) for `[x1, y1, x2, y2]`.
    pub fn corners_coordinates(&self, frame: &RawFrame) -> (u32, u32) {
        let x1 = self.bbox[0] as u32;
        let y1 = self.bbox[1] as u32;
        let x2 = self.bbox[2] as u32;
        let y2 = self.bbox[3] as u32;

        let top_left_corner = y1 * frame.width + x1;
        let bottom_right_corner = y2 * frame.width + x2;

        (top_left_corner, bottom_right_corner)
    }
}

// C pointer readers
pub(crate) fn get_c_array<T: Clone>(
    ptr: *const T,
    len: usize,
    ownership: Ownership,
) -> Result<Vec<T>> {
    if ptr.is_null() {
        anyhow::bail!("received a null C array");
    }

    let slice = unsafe {
        std::slice::from_raw_parts(ptr, len)
    };

    // Copy before releasing: for an array of pointers the elements outlive their array.
    let values = slice.to_vec();
    ownership.release(ptr);

    Ok(values)
}

pub(crate) fn get_c_string(ptr: *const c_char, ownership: Ownership) -> Result<String> {
    if ptr.is_null() {
        anyhow::bail!("received a null C string");
    }

    let string = unsafe {
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    };
    ownership.release(ptr);

    Ok(string)
}
