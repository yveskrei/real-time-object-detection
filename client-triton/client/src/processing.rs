//! Responsible for pre/post processing images before inference.
//! Performs operations on raw frames/inference results with SIMD optimizations

use anyhow::Result;
use std::sync::OnceLock;
use tokio::time::Instant;
use serde::Serialize;

// Custom modules
pub mod yolo;
pub mod dino;
use crate::utils::config::InferencePrecision;

/// Normalization constants
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
const PAD_GRAY_COLOR: usize = 114;

/// Represents raw frame before performing inference on it
#[derive(Clone, Debug)]
pub struct RawFrame {
    pub data: Vec<u8>,
    pub height: u32,
    pub width: u32,
    pub pts: u64,
    pub added: Instant
}

/// Represents a single bbox instance from the model inference results
#[derive(Clone, Copy, Serialize)]
pub struct ResultBBOX {
    pub bbox: [f32; 4],
    pub class: u32, 
    pub score: f32
}

impl ResultBBOX {
    pub fn class_name(&self) -> &'static str {
        match self.class {
            0 => "person",
            1 => "bicycle",
            2 => "car",
            3 => "motorcycle",
            4 => "airplane",
            5 => "bus",
            _ => Box::leak(self.class.to_string().into_boxed_str())
        }
    }

    pub fn corners_coordinates(&self, frame: &RawFrame) -> (u32, u32) {
        // Extract bbox coordinates [x1, y1, x2, y2]
        let x1 = self.bbox[0] as u32;
        let y1 = self.bbox[1] as u32;
        let x2 = self.bbox[2] as u32;
        let y2 = self.bbox[3] as u32;
        
        // Calculate 1D array indices
        let top_left_corner = y1 * frame.width + x1;
        let bottom_right_corner = y2 * frame.width + x2;

        return (top_left_corner, bottom_right_corner)
    }
}

/// Represents embedding output from the model inference results
#[derive(Clone, Serialize)]
pub struct ResultEmbedding {
    pub data: Vec<f32>
}

impl ResultEmbedding {
    pub fn get_raw_bytes(&self) -> Vec<u8> {
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * std::mem::size_of::<f32>()
            )
        }.to_vec()
    }
}

/// Lookup table for converting values from FP16 to FP32
pub static F16_TO_F32_LUT: OnceLock<Box<[f32; 65536]>> = OnceLock::new();
/// Lookup table for F32 to F16 conversion
pub static F32_TO_F16_LUT: OnceLock<Box<[u16; 32768]>> = OnceLock::new();
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

/// Create static lookup table for F32 to F16 conversion
fn create_f32_to_f16_lut() -> Box<[u16; 32768]> {
    let mut lut = Box::new([0u16; 32768]);
    
    const MIN_VAL: f32 = -4.0;
    const MAX_VAL: f32 = 4.0;
    const RANGE: f32 = MAX_VAL - MIN_VAL;
    const STEP: f32 = RANGE / 32768.0;
    
    for i in 0..32768 {
        let val = MIN_VAL + (i as f32) * STEP;
        let bits = val.to_bits();
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

fn get_f32_to_f16_lut(val: f32) -> u16 {
    const MIN_VAL: f32 = -4.0;
    const MAX_VAL: f32 = 4.0;
    const RANGE: f32 = MAX_VAL - MIN_VAL;
    
    let clamped_val = val.clamp(MIN_VAL, MAX_VAL);
    let index = ((clamped_val - MIN_VAL) / RANGE * 32767.0) as usize;
    let index = index.min(32767);
    
    F32_TO_F16_LUT
        .get_or_init(create_f32_to_f16_lut)[index]
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


#[derive(Copy, Clone, Debug)]
pub struct LetterboxParams {
    pub pad_x: u32,
    pub pad_y: u32,
    pub new_width: u32,
    pub new_height: u32,
    pub inv_scale: f32,
}

/// Calculates values for letterbox padding
pub fn calculate_letterbox(
    height: u32,
    width: u32,
    target_size: u32,
) -> LetterboxParams {
    let max_dim = height.max(width) as f32;
    let scale = (target_size as f32) / max_dim;
    let inv_scale = max_dim / (target_size as f32);

    let new_width = ((width as f32 * scale) as u32).min(target_size);
    let new_height = ((height as f32 * scale) as u32).min(target_size);

    let pad_x = (target_size - new_width) >> 1; // Bit shift for / 2
    let pad_y = (target_size - new_height) >> 1;

    LetterboxParams {
        pad_x,
        pad_y,
        new_width,
        new_height,
        inv_scale,
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

    // 4. Perform fused resize, normalization, and planar conversion
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

///
/// Performs a single-pass, fused nearest-neighbor resize, letterbox,
/// pixel normalization (x / 255.0) and ImageNet normalization.
///
/// * `input`: Raw `u8` RGB interleaved pixel data.
/// * `in_h`, `in_w`: Dimensions of the `input` image.
/// * `target_h`, `target_w`: Dimensions of the `output` buffer.
/// * `precision`: The desired output precision (FP32 or FP16).
///
/// Returns a new `Vec<u8>` containing the final FP32 or FP16 planar data.
///
pub fn resize_letterbox_and_normalize_imagenet(
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

    // 3. Get normalization constants
    let r_mean = IMAGENET_MEAN[0];
    let g_mean = IMAGENET_MEAN[1];
    let b_mean = IMAGENET_MEAN[2];
    let r_std_inv = 1.0 / IMAGENET_STD[0];
    let g_std_inv = 1.0 / IMAGENET_STD[1];
    let b_std_inv = 1.0 / IMAGENET_STD[2];
    let norm_lut_f32 = get_f32_lut(); // u8 -> f32 (0-1)

    // 4. Pre-calculate x-offsets for the source image
    let mut x_offsets: Vec<u32> = Vec::with_capacity(letterbox.new_width as usize);
    for x in 0..letterbox.new_width {
        x_offsets.push(((x as f32 * letterbox.inv_scale) as u32).min(in_w - 1) * 3);
    }

    let in_ptr = input.as_ptr();

    // 5. Calculate padding values (normalized with ImageNet)
    let pad_val_r = (norm_lut_f32[PAD_GRAY_COLOR] - r_mean) * r_std_inv;
    let pad_val_g = (norm_lut_f32[PAD_GRAY_COLOR] - g_mean) * g_std_inv;
    let pad_val_b = (norm_lut_f32[PAD_GRAY_COLOR] - b_mean) * b_std_inv;

    // 6. Perform fused resize, normalization (pixel + ImageNet), and planar conversion
    match precision {
        InferencePrecision::FP16 => {
            let pad_val_r_f16 = get_f32_to_f16_lut(pad_val_r);
            let pad_val_g_f16 = get_f32_to_f16_lut(pad_val_g);
            let pad_val_b_f16 = get_f32_to_f16_lut(pad_val_b);
            
            let out_ptr = output.as_mut_ptr() as *mut u16;
            let (out_r, out_g, out_b) = unsafe {
                (
                    std::slice::from_raw_parts_mut(out_ptr, num_pixels),
                    std::slice::from_raw_parts_mut(out_ptr.add(num_pixels), num_pixels),
                    std::slice::from_raw_parts_mut(out_ptr.add(num_pixels * 2), num_pixels),
                )
            };

            // Pre-fill with normalized padding color
            out_r.fill(pad_val_r_f16);
            out_g.fill(pad_val_g_f16);
            out_b.fill(pad_val_b_f16);

            // Write real pixels with ImageNet normalization
            for y in 0..letterbox.new_height {
                let src_y = ((y as f32 * letterbox.inv_scale) as u32).min(in_h - 1);
                let src_row_offset = src_y * in_w * 3;
                let dst_y = y + letterbox.pad_y;

                for x in 0..letterbox.new_width {
                    let src_idx = (src_row_offset + x_offsets[x as usize]) as usize;
                    let dst_idx = (dst_y * target_w + (x + letterbox.pad_x)) as usize;

                    unsafe {
                        let r_norm = (norm_lut_f32[*in_ptr.add(src_idx) as usize] - r_mean) * r_std_inv;
                        let g_norm = (norm_lut_f32[*in_ptr.add(src_idx + 1) as usize] - g_mean) * g_std_inv;
                        let b_norm = (norm_lut_f32[*in_ptr.add(src_idx + 2) as usize] - b_mean) * b_std_inv;

                        out_r[dst_idx] = get_f32_to_f16_lut(r_norm);
                        out_g[dst_idx] = get_f32_to_f16_lut(g_norm);
                        out_b[dst_idx] = get_f32_to_f16_lut(b_norm);
                    }
                }
            }
        }
        InferencePrecision::FP32 => {
            let out_ptr = output.as_mut_ptr() as *mut f32;
            let (out_r, out_g, out_b) = unsafe {
                (
                    std::slice::from_raw_parts_mut(out_ptr, num_pixels),
                    std::slice::from_raw_parts_mut(out_ptr.add(num_pixels), num_pixels),
                    std::slice::from_raw_parts_mut(out_ptr.add(num_pixels * 2), num_pixels),
                )
            };

            // Pre-fill with normalized padding color
            out_r.fill(pad_val_r);
            out_g.fill(pad_val_g);
            out_b.fill(pad_val_b);

            // Write real pixels with ImageNet normalization
            for y in 0..letterbox.new_height {
                let src_y = ((y as f32 * letterbox.inv_scale) as u32).min(in_h - 1);
                let src_row_offset = src_y * in_w * 3;
                let dst_y = y + letterbox.pad_y;

                for x in 0..letterbox.new_width {
                    let src_idx = (src_row_offset + x_offsets[x as usize]) as usize;
                    let dst_idx = (dst_y * target_w + (x + letterbox.pad_x)) as usize;

                    unsafe {
                        out_r[dst_idx] = (norm_lut_f32[*in_ptr.add(src_idx) as usize] - r_mean) * r_std_inv;
                        out_g[dst_idx] = (norm_lut_f32[*in_ptr.add(src_idx + 1) as usize] - g_mean) * g_std_inv;
                        out_b[dst_idx] = (norm_lut_f32[*in_ptr.add(src_idx + 2) as usize] - b_mean) * b_std_inv;
                    }
                }
            }
        }
    }

    Ok(output)
}