//! Responsible for pre/post processing images before inference.
//! Performs operations on raw frames/inference results with SIMD optimizations

use anyhow::Result;
use std::sync::OnceLock;

// Custom modules
pub mod yolo;
use crate::utils::config::InferencePrecision;

/// Normalization constants
const PAD_GRAY_COLOR: usize = 114;

/// Lookup table for converting values from FP16 to FP32
pub static F16_TO_F32_LUT: OnceLock<Box<[f32; 65536]>> = OnceLock::new();
/// Lookup table for converting pixel values to FP16
pub static F16_LUT: OnceLock<Box<[u16; 256]>> = OnceLock::new();
/// Lookup table for converting pixel values to FP32
pub static F32_LUT: OnceLock<Box<[f32; 256]>> = OnceLock::new();

/// Create static lookup table for high speed conversion
fn create_f16_to_f32_lut() -> Box<[f32; 65536]> {
    let mut lut = Box::new([0.0f32; 65536]);

    for i in 0u16..=65535 {
        let sign = (i >> 15) & 0x1;
        let exp = (i >> 10) & 0x1f;
        let frac = i & 0x3ff;

        lut[i as usize] = if exp == 0 {
            if frac == 0 {
                if sign == 1 { -0.0 } else { 0.0 }
            } else {
                // Denormal
                let mut val = frac as f32 / 1024.0 / 16384.0;
                if sign == 1 { val = -val; }
                val
            }
        } else if exp == 31 {
            // Infinity or NaN
            if frac == 0 {
                if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
            } else {
                f32::NAN
            }
        } else {
            // Normal numbers
            let exp_f32 = (exp as i32 - 15 + 127) as u32;
            let frac_f32 = (frac as u32) << 13;
            let bits = (sign as u32) << 31 | exp_f32 << 23 | frac_f32;
            f32::from_bits(bits)
        };
    }

    lut
}

pub fn get_f16_to_f32_lut(val: u16) -> f32 {
    F16_TO_F32_LUT
        .get_or_init(create_f16_to_f32_lut)[val as usize]
}

/// Create static lookup table for high speed conversion
fn create_f16_lut() -> Box<[u16; 256]> {
    let mut lut = Box::new([0u16; 256]);
    for i in 0..256 {
        let normalized = i as f32 / 255.0;
        let bits = normalized.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xff) as i32;
        let mantissa = bits & 0x7fffff;
        lut[i] = if exp == 0 {
            sign as u16
        } else {
            let exp_adj = exp - 127 + 15;
            if exp_adj >= 31 {
                (sign | 0x7c00) as u16
            } else if exp_adj <= 0 {
                sign as u16
            } else {
                let mantissa_adj = mantissa >> 13;
                (sign | ((exp_adj as u32) << 10) | mantissa_adj) as u16
            }
        };
    }
    lut
}

pub fn get_f16_lut() -> &'static [u16; 256] {
    F16_LUT
        .get_or_init(create_f16_lut)
}

/// Create static lookup table for high speed conversion
fn create_f32_lut() -> Box<[f32; 256]> {
    let mut lut = Box::new([0.0f32; 256]);
    for i in 0..256 {
        lut[i] = i as f32 / 255.0;
    }
    lut
}

pub fn get_f32_lut() -> &'static [f32; 256] {
    F32_LUT
        .get_or_init(create_f32_lut)
}

#[derive(Debug, Clone, Copy)]
pub struct LetterboxParams {
    pub new_width: u32,
    pub new_height: u32,
    pub pad_x: u32,
    pub pad_y: u32,
    pub scale: f32,
    pub inv_scale: f32,
}

/// Calculate letterbox parameters for image resizing
pub fn calculate_letterbox(in_h: u32, in_w: u32, target_size: u32) -> LetterboxParams {
    let scale = (target_size as f32) / (in_h.max(in_w) as f32);
    let new_width = (in_w as f32 * scale) as u32;
    let new_height = (in_h as f32 * scale) as u32;
    let pad_x = (target_size - new_width) / 2;
    let pad_y = (target_size - new_height) / 2;

    LetterboxParams {
        new_width,
        new_height,
        pad_x,
        pad_y,
        scale,
        inv_scale: 1.0 / scale,
    }
}

///
/// Performs a single-pass, fused nearest-neighbor resize, letterbox,
/// and pixel normalization (x / 255.0).
///
/// * `input`: Raw `u8` RGB interleaved pixel data.
/// * `in_h`, `in_w`: Dimensions of the `input` image.
/// * `target_h`, `target_w`: Dimensions of the `output` buffer.
/// * `precision`: The desired output precision (FP32 or FP16).
///
/// Returns a new `Vec<u8>` containing the final FP32 or FP16 planar data.
///
pub fn resize_letterbox_and_normalize(
    input: &[u8],
    in_h: u32,
    in_w: u32,
    target_h: u32,
    target_w: u32,
    precision: InferencePrecision,
) -> Result<Vec<u8>> {
    // 1. Calculate letterbox params
    let letterbox = calculate_letterbox(in_h, in_w, target_h.max(target_w));
    let num_pixels = (target_h * target_w) as usize;

    // 2. Allocate the *FINAL* output buffer ONCE
    let mut output: Vec<u8> = match precision {
        InferencePrecision::FP16 => vec![0u8; num_pixels * 3 * 2],
        InferencePrecision::FP32 => vec![0u8; num_pixels * 3 * 4],
    };

    // 3. Pre-calculate x-offsets for the source image
    let mut x_offsets: Vec<u32> = Vec::with_capacity(letterbox.new_width as usize);
    for x in 0..letterbox.new_width {
        x_offsets.push(((x as f32 * letterbox.inv_scale) as u32).min(in_w - 1) * 3);
    }

    let in_ptr = input.as_ptr();

    // 4. Process based on precision
    match precision {
        InferencePrecision::FP16 => {
            // Get the U8 -> F16 LUT (fast, L1-cache resident)
            let norm_lut_f16 = get_f16_lut();
            let pad_val_f16 = norm_lut_f16[PAD_GRAY_COLOR];

            let out_ptr = output.as_mut_ptr() as *mut u16;
            let (out_r, out_g, out_b) = unsafe {
                (
                    std::slice::from_raw_parts_mut(out_ptr, num_pixels),
                    std::slice::from_raw_parts_mut(out_ptr.add(num_pixels), num_pixels),
                    std::slice::from_raw_parts_mut(out_ptr.add(num_pixels * 2), num_pixels),
                )
            };

            // 5. Pre-fill the *entire* buffer with the *normalized* padding color
            out_r.fill(pad_val_f16);
            out_g.fill(pad_val_f16);
            out_b.fill(pad_val_f16);

            // 6. Iterate *only* over the target image area and write real pixels
            for y in 0..letterbox.new_height {
                let src_y = ((y as f32 * letterbox.inv_scale) as u32).min(in_h - 1);
                let src_row_offset = src_y * in_w * 3;
                let dst_y = y + letterbox.pad_y;

                for x in 0..letterbox.new_width {
                    let src_idx = (src_row_offset + x_offsets[x as usize]) as usize;
                    let dst_idx = (dst_y * target_w + (x + letterbox.pad_x)) as usize;

                    unsafe {
                        out_r[dst_idx] = norm_lut_f16[*in_ptr.add(src_idx) as usize];
                        out_g[dst_idx] = norm_lut_f16[*in_ptr.add(src_idx + 1) as usize];
                        out_b[dst_idx] = norm_lut_f16[*in_ptr.add(src_idx + 2) as usize];
                    }
                }
            }
        }
        InferencePrecision::FP32 => {
            // Get the U8 -> F32 LUT (fast, L1-cache resident)
            let norm_lut_f32 = get_f32_lut();
            let pad_val_f32 = norm_lut_f32[PAD_GRAY_COLOR];

            let out_ptr = output.as_mut_ptr() as *mut f32;
            let (out_r, out_g, out_b) = unsafe {
                (
                    std::slice::from_raw_parts_mut(out_ptr, num_pixels),
                    std::slice::from_raw_parts_mut(out_ptr.add(num_pixels), num_pixels),
                    std::slice::from_raw_parts_mut(out_ptr.add(num_pixels * 2), num_pixels),
                )
            };

            // 5. Pre-fill the *entire* buffer with the *normalized* padding color
            out_r.fill(pad_val_f32);
            out_g.fill(pad_val_f32);
            out_b.fill(pad_val_f32);

            // 6. Iterate *only* over the target image area and write real pixels
            for y in 0..letterbox.new_height {
                let src_y = ((y as f32 * letterbox.inv_scale) as u32).min(in_h - 1);
                let src_row_offset = src_y * in_w * 3;
                let dst_y = y + letterbox.pad_y;

                for x in 0..letterbox.new_width {
                    let src_idx = (src_row_offset + x_offsets[x as usize]) as usize;
                    let dst_idx = (dst_y * target_w + (x + letterbox.pad_x)) as usize;

                    unsafe {
                        // Fetch U8, normalize with LUT, write to F32 planar buffer
                        out_r[dst_idx] = norm_lut_f32[*in_ptr.add(src_idx) as usize];
                        out_g[dst_idx] = norm_lut_f32[*in_ptr.add(src_idx + 1) as usize];
                        out_b[dst_idx] = norm_lut_f32[*in_ptr.add(src_idx + 2) as usize];
                    }
                }
            }
        }
    }

    Ok(output)
}
