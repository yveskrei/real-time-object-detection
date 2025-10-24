//! Contains modular, sequential preprocessing routines.
//! The public functions are "fused," performing resize, padding, and
//! normalization in a single pass to minimize memory I/O.
use anyhow::Result;

// Custom modules
use crate::processing;
use crate::utils::config::InferencePrecision;

/// ImageNet normalization constants
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
const PAD_GRAY_COLOR: usize = 114;

#[derive(Copy, Clone, Debug)]
pub struct LetterboxParams {
    pub pad_x: usize,
    pub pad_y: usize,
    pub new_width: usize,
    pub new_height: usize,
    pub inv_scale: f32,
}

/// Calculates values for letterbox padding
pub fn calculate_letterbox(
    height: usize,
    width: usize,
    target_size: usize,
) -> LetterboxParams {
    let max_dim = height.max(width) as f32;
    let scale = (target_size as f32) / max_dim;
    let inv_scale = max_dim / (target_size as f32);

    let new_width = ((width as f32 * scale) as usize).min(target_size);
    let new_height = ((height as f32 * scale) as usize).min(target_size);

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
/// and YOLO normalization (x / 255.0).
///
/// * `input`: Raw `u8` RGB interleaved pixel data.
/// * `in_h`, `in_w`: Dimensions of the `input` image.
/// * `target_h`, `target_w`: Dimensions of the `output` buffer.
/// * `precision`: The desired output precision (FP32 or FP16).
/// * `layout`: The desired output layout (MUST be Planar for this op).
///
/// Returns a new `Vec<u8>` containing the final FP32 or FP16 planar data.
///
pub fn resize_letterbox_and_normalize(
    input: &[u8],
    in_h: usize,
    in_w: usize,
    target_h: usize,
    target_w: usize,
    precision: InferencePrecision,
) -> Result<Vec<u8>> {
    // 1. Calculate letterbox params
    let letterbox = calculate_letterbox(in_h, in_w, target_h.max(target_w));
    let num_pixels = target_h * target_w;

    // 2. Allocate the *FINAL* output buffer ONCE
    let mut output: Vec<u8> = match precision {
        InferencePrecision::FP16 => vec![0u8; num_pixels * 3 * 2],
        InferencePrecision::FP32 => vec![0u8; num_pixels * 3 * 4],
    };

    // 3. Pre-calculate x-offsets for the source image
    let mut x_offsets: Vec<usize> = Vec::with_capacity(letterbox.new_width);
    for x in 0..letterbox.new_width {
        x_offsets.push(((x as f32 * letterbox.inv_scale) as usize).min(in_w - 1) * 3);
    }

    let in_ptr = input.as_ptr();

    // 4. Perform fused resize, normalization, and planar conversion
    match precision {
        InferencePrecision::FP16 => {
            // Get the U8 -> F16 LUT (fast, L1-cache resident)
            let norm_lut_f16 = processing::get_f16_lut();
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
                let src_y = ((y as f32 * letterbox.inv_scale) as usize).min(in_h - 1);
                let src_row_offset = src_y * in_w * 3;
                let dst_y = y + letterbox.pad_y;

                for x in 0..letterbox.new_width {
                    let src_idx = src_row_offset + x_offsets[x];
                    let dst_idx = dst_y * target_w + (x + letterbox.pad_x);

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
            let norm_lut_f32 = processing::get_f32_lut();
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
                let src_y = ((y as f32 * letterbox.inv_scale) as usize).min(in_h - 1);
                let src_row_offset = src_y * in_w * 3;
                let dst_y = y + letterbox.pad_y;

                for x in 0..letterbox.new_width {
                    let src_idx = src_row_offset + x_offsets[x];
                    let dst_idx = dst_y * target_w + (x + letterbox.pad_x);

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
/// Performs a single-pass, fused shortest-edge resize, center crop,
/// and ImageNet normalization.
///
/// * `input`: Raw `u8` RGB interleaved pixel data.
/// * `in_h`, `in_w`: Dimensions of the `input` image.
/// * `crop_h`, `crop_w`: Dimensions of the `output` buffer (e.g., 224x224).
/// * `target_shortest_edge`: The size to resize the shortest edge to *before* cropping (e.g., 256).
/// * `precision`: The desired output precision (FP32 or FP16).
/// * `layout`: The desired output layout (MUST be Planar for this op).
///
/// Returns a new `Vec<u8>` containing the final FP32 or FP16 planar data.
///
pub fn resize_center_crop_and_normalize(
    input: &[u8],
    in_h: usize,
    in_w: usize,
    crop_h: usize,
    crop_w: usize,
    target_shortest_edge: usize,
    precision: InferencePrecision,
) -> Result<Vec<u8>, &'static str> {
    // 1. Allocate the *FINAL* output buffer ONCE
    let num_pixels = crop_h * crop_w;
    let mut output: Vec<u8> = match precision {
        InferencePrecision::FP16 => vec![0u8; num_pixels * 3 * 2],
        InferencePrecision::FP32 => vec![0u8; num_pixels * 3 * 4],
    };

    // 2. Calculate resize parameters
    let min_dim = in_h.min(in_w) as f32;
    let scale = target_shortest_edge as f32 / min_dim;
    let new_width = (in_w as f32 * scale).round() as usize;
    let new_height = (in_h as f32 * scale).round() as usize;

    let scale_x = new_width as f32 / in_w as f32;
    let scale_y = new_height as f32 / in_h as f32;

    // 3. Calculate center crop parameters
    let crop_x_start = (new_width.saturating_sub(crop_w)) / 2;
    let crop_y_start = (new_height.saturating_sub(crop_h)) / 2;

    // 4. Get normalization constants
    let r_mean = IMAGENET_MEAN[0];
    let g_mean = IMAGENET_MEAN[1];
    let b_mean = IMAGENET_MEAN[2];
    let r_std_inv = 1.0 / IMAGENET_STD[0];
    let g_std_inv = 1.0 / IMAGENET_STD[1];
    let b_std_inv = 1.0 / IMAGENET_STD[2];
    let norm_lut_f32 = processing::get_f32_lut(); // u8 -> f32 (0-1)

    // 5. Iterate and perform fused op
    match precision {
        InferencePrecision::FP16 => {
            let out_ptr = output.as_mut_ptr() as *mut u16;
            let (out_r, out_g, out_b) = unsafe {
                (
                    std::slice::from_raw_parts_mut(out_ptr, num_pixels),
                    std::slice::from_raw_parts_mut(out_ptr.add(num_pixels), num_pixels),
                    std::slice::from_raw_parts_mut(out_ptr.add(num_pixels * 2), num_pixels),
                )
            };
            
            for y in 0..crop_h {
                for x in 0..crop_w {
                    // Find the corresponding pixel in the *original* image
                    let src_x_f = (x + crop_x_start) as f32 / scale_x;
                    let src_y_f = (y + crop_y_start) as f32 / scale_y;

                    // Use `floor` for nearest-neighbor to prevent off-by-one
                    let src_x = (src_x_f.floor() as usize).min(in_w - 1);
                    let src_y = (src_y_f.floor() as usize).min(in_h - 1);

                    // If we're cropping "outside" the resized image, let the
                    // default 0.0s (from vec! init) act as padding.
                    if src_x_f >= new_width as f32 || src_y_f >= new_height as f32 {
                        continue;
                    }
                    
                    let src_idx = (src_y * in_w + src_x) * 3;
                    let dst_idx = y * crop_w + x;

                    unsafe {
                        let r_norm = (norm_lut_f32[*input.get_unchecked(src_idx) as usize] - r_mean) * r_std_inv;
                        let g_norm = (norm_lut_f32[*input.get_unchecked(src_idx + 1) as usize] - g_mean) * g_std_inv;
                        let b_norm = (norm_lut_f32[*input.get_unchecked(src_idx + 2) as usize] - b_mean) * b_std_inv;

                        out_r[dst_idx] = processing::get_f32_to_f16_lut(r_norm);
                        out_g[dst_idx] = processing::get_f32_to_f16_lut(g_norm);
                        out_b[dst_idx] = processing::get_f32_to_f16_lut(b_norm);
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

            for y in 0..crop_h {
                for x in 0..crop_w {
                    let src_x_f = (x + crop_x_start) as f32 / scale_x;
                    let src_y_f = (y + crop_y_start) as f32 / scale_y;

                    let src_x = (src_x_f.floor() as usize).min(in_w - 1);
                    let src_y = (src_y_f.floor() as usize).min(in_h - 1);

                    if src_x_f >= new_width as f32 || src_y_f >= new_height as f32 {
                        continue;
                    }
                    
                    let src_idx = (src_y * in_w + src_x) * 3;
                    let dst_idx = y * crop_w + x;
                    
                    unsafe {
                        out_r[dst_idx] = (norm_lut_f32[*input.get_unchecked(src_idx) as usize] - r_mean) * r_std_inv;
                        out_g[dst_idx] = (norm_lut_f32[*input.get_unchecked(src_idx + 1) as usize] - g_mean) * g_std_inv;
                        out_b[dst_idx] = (norm_lut_f32[*input.get_unchecked(src_idx + 2) as usize] - b_mean) * b_std_inv;
                    }
                }
            }
        }
    }

    Ok(output)
}
