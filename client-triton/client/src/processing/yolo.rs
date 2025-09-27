use anyhow::{Result, Context};
use std::time::Instant;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::inference::source::SourceProcessor;
// Custom modules
use crate::inference::{
    source::FrameProcessStats, 
    InferenceModel, 
    InferencePrecision
};
use crate::processing::{self, RawFrame, ResultBBOX};
use crate::utils::config::SourceConfig;

#[derive(Copy, Clone)]
struct LetterboxParams {
    pad_x: f32,
    pad_y: f32,
    new_width: usize,
    new_height: usize,
    _scale: f32,
    inv_scale: f32,
}

/// Calculates values for letterbox padding
/// 
/// Calculates necessary values to preserve the the aspect ratio of a given image
/// by adding additional blank pixels
fn calculate_letterbox(height: usize, width: usize, target_size: usize) -> LetterboxParams {
    let max_dim = height.max(width) as f32;
    let scale = (target_size as f32) / max_dim;
    let inv_scale = max_dim / (target_size as f32);
    
    let new_width = ((width as f32 * scale) as usize).min(target_size);
    let new_height = ((height as f32 * scale) as usize).min(target_size);
    
    // Use bit shift for division by 2 when possible
    let pad_x = ((target_size - new_width) >> 1) as f32;
    let pad_y = ((target_size - new_height) >> 1) as f32;
    
    LetterboxParams {
        pad_x,
        pad_y,
        new_width,
        new_height,
        _scale: scale,
        inv_scale,
    }
}

/// Perform NMS reduction of bboxes
#[inline(never)] // Don't inline to keep instruction cache hot for main loop
fn bbox_nms(detections: &mut Vec<ResultBBOX>, nms_threshold: f32) {
    let len = detections.len();
    if len <= 1 {
        return;
    }
    
    // Sort in-place by score descending
    detections.sort_unstable_by(|a, b| {
        b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
    });
    
    let mut write_idx = 0;
    
    for i in 0..len {
        let detection_i = unsafe { *detections.get_unchecked(i) };
        let mut should_keep = true;
        
        // Check against already kept detections
        for j in 0..write_idx {
            let kept = unsafe { detections.get_unchecked(j) };
            
            // Skip different classes
            if kept.class != detection_i.class {
                continue;
            }
            
            // Compute IoU inline
            let x1_max = detection_i.bbox[0].max(kept.bbox[0]);
            let y1_max = detection_i.bbox[1].max(kept.bbox[1]);
            let x2_min = detection_i.bbox[2].min(kept.bbox[2]);
            let y2_min = detection_i.bbox[3].min(kept.bbox[3]);
            
            // Check for intersection
            if x1_max < x2_min && y1_max < y2_min {
                let intersection = (x2_min - x1_max) * (y2_min - y1_max);
                let area_i = (detection_i.bbox[2] - detection_i.bbox[0]) * (detection_i.bbox[3] - detection_i.bbox[1]);
                let area_j = (kept.bbox[2] - kept.bbox[0]) * (kept.bbox[3] - kept.bbox[1]);
                let union = area_i + area_j - intersection;
                
                if intersection > nms_threshold * union {
                    should_keep = false;
                    break;
                }
            }
        }
        
        if should_keep {
            unsafe {
                *detections.get_unchecked_mut(write_idx) = detection_i;
            }
            write_idx += 1;
        }
    }
    
    detections.truncate(write_idx);
}

/// Performs post-processing on inference results for YOLO models
/// 
/// Including the following steps of processing:
/// 1. Convert BBOX coordinates from (x, y, w, h) to (x1, y1, x2, y2) together
/// with restoring the letterbox padding applied during pre-processing
/// 2. Finds out the class id with the max probability - making it the 
/// class for the bbox along with its probabiliy
/// 3. Filter BBOXes on a given confidence threshold, before applying NMS(boosts performance significantly)
/// 4. Perform NMS on left over BBOXes
pub fn postprocess(
    results: &[u8],
    original_frame: &RawFrame,
    output_shape: &[i64],
    precision: InferencePrecision,
    pred_conf_threshold: f32,
    nms_iou_threshold: f32,
) -> Result<Vec<ResultBBOX>> {
    // Validate model output shape
    if output_shape.len() != 2 {
        anyhow::bail!(
            format!(
                "Got unexpected size of model output shape. Got {}, expected 2",
                output_shape.len()
            )
        );
    }

    let target_features = output_shape[0] as usize;
    let target_anchors = output_shape[1] as usize;
    let target_classes = target_features - 4;
    
    // Validate size of output data
    let expected_size = match precision {
        InferencePrecision::FP16 => target_anchors * target_features * 2,
        InferencePrecision::FP32 => target_anchors * target_features * 4,
    };
    
    if results.len() != expected_size {
        anyhow::bail!(
            format!(
                "Got unexpected size of model output data ({}). Got {}, expected {}",
                precision.to_string(),
                results.len(),
                expected_size
            )
        );
    }
    
    // Precompute letterbox parameters
    const TARGET_SIZE: usize = 640;
    let lb = calculate_letterbox(original_frame.height, original_frame.width, TARGET_SIZE);
    
    // Pre-allocate with exact capacity estimate (typically ~100-200 detections)
    let mut detections = Vec::with_capacity(256);
    
    match precision {
        InferencePrecision::FP16 => {
            let u16_data = unsafe {
                std::slice::from_raw_parts(results.as_ptr() as *const u16, results.len() / 2)
            };
            
            // Precompute strides
            let stride1 = target_anchors;
            let stride2 = target_anchors * 2;
            let stride3 = target_anchors * 3;
            let stride4 = target_anchors * 4;
            
            // Process anchors with optimized memory access pattern
            for anchor_idx in 0..target_anchors {
                unsafe {
                    // Load all bbox values at once for better cache usage
                    let x = processing::get_f16_to_f32_lut(*u16_data.get_unchecked(anchor_idx));
                    let y = processing::get_f16_to_f32_lut(*u16_data.get_unchecked(stride1 + anchor_idx));
                    let w = processing::get_f16_to_f32_lut(*u16_data.get_unchecked(stride2 + anchor_idx));
                    let h = processing::get_f16_to_f32_lut(*u16_data.get_unchecked(stride3 + anchor_idx));
                    
                    // Fused bbox transformation
                    let half_w = w * 0.5;
                    let half_h = h * 0.5;
                    let x1 = (x - half_w - lb.pad_x) * lb.inv_scale;
                    let y1 = (y - half_h - lb.pad_y) * lb.inv_scale;
                    let x2 = (x + half_w - lb.pad_x) * lb.inv_scale;
                    let y2 = (y + half_h - lb.pad_y) * lb.inv_scale;
                    
                    // Find max class with unrolled loop for common cases
                    let mut max_score = 0.0f32;
                    let mut max_class = 0usize;
                    
                    let class_base = stride4 + anchor_idx;
                    
                    for class_idx in 0..target_classes {
                        let prob_idx = class_base + class_idx * stride1;
                        let score = processing::get_f16_to_f32_lut(*u16_data.get_unchecked(prob_idx));
                        if score > max_score {
                            max_score = score;
                            max_class = class_idx;
                        }
                    }
                    
                    // Only store if above threshold
                    if max_score >= pred_conf_threshold {
                        detections.push(
                            ResultBBOX {
                                bbox: [x1, y1, x2, y2],
                                class: max_class,
                                score: max_score,
                            }
                        );
                    }
                }
            }
        }
        InferencePrecision::FP32 => {
            let f32_data = unsafe {
                std::slice::from_raw_parts(results.as_ptr() as *const f32, results.len() / 4)
            };
            
            // Precompute strides
            let stride1 = target_anchors;
            let stride2 = target_anchors * 2;
            let stride3 = target_anchors * 3;
            let stride4 = target_anchors * 4;
            
            for anchor_idx in 0..target_anchors {
                unsafe {
                    // Load bbox values
                    let x = *f32_data.get_unchecked(anchor_idx);
                    let y = *f32_data.get_unchecked(stride1 + anchor_idx);
                    let w = *f32_data.get_unchecked(stride2 + anchor_idx);
                    let h = *f32_data.get_unchecked(stride3 + anchor_idx);
                    
                    // Fused bbox transformation
                    let half_w = w * 0.5;
                    let half_h = h * 0.5;
                    let x1 = (x - half_w - lb.pad_x) * lb.inv_scale;
                    let y1 = (y - half_h - lb.pad_y) * lb.inv_scale;
                    let x2 = (x + half_w - lb.pad_x) * lb.inv_scale;
                    let y2 = (y + half_h - lb.pad_y) * lb.inv_scale;
                    
                    // Find max class with unrolling
                    let mut max_score = 0.0f32;
                    let mut max_class = 0usize;
                    
                    let class_base = stride4 + anchor_idx;
                    
                    for class_idx in 0..target_classes {
                        let prob_idx = class_base + class_idx * stride1;
                        let score = *f32_data.get_unchecked(prob_idx);
                        if score > max_score {
                            max_score = score;
                            max_class = class_idx;
                        }
                    }
                    
                    if max_score >= pred_conf_threshold {
                        detections.push(
                            ResultBBOX {
                                bbox: [x1, y1, x2, y2],
                                class: max_class,
                                score: max_score,
                            }
                        );
                    }
                }
            }
        }
    }
    
    // Fast NMS only if needed
    if detections.len() > 1 {
        bbox_nms(&mut detections, nms_iou_threshold);
    }
    
    Ok(detections)
}

/// Performs pre-processing on raw RGB frame for YOLO models
/// 
/// Performs the following steps of processing:
/// 1. Resizes the given image to 640x640 while preserving aspect ratio.
/// Applying letterbox padding to complete the missing pixels for certain aspect ratios.
/// 2. Normalizes pixels from 0-255 to 0-1
/// 3. Converting raw pixel values to required precision datatype
/// 4. Outputs raw bytes ordered by color channels: \[RRRBBBGGG\]
pub fn preprocess(
    frame: &RawFrame,
    precision: InferencePrecision,
) -> Result<Vec<u8>> {
    // Check if input size matches
    let frame_target_size = frame.height * frame.width * 3;
    if frame.data.len() != frame_target_size {
        anyhow::bail!(
            format!(
                "Got unexpected size of frame input. Got {}, expected {}",
                frame.data.len(),
                frame_target_size
            )
        );
    }

    // Calculate target size
    const TARGET_SIZE: usize = 640;

    // letterbox calculation
    let lb = calculate_letterbox(frame.height, frame.width, TARGET_SIZE);
    
    // Stack-allocated coordinate buffers - each thread gets its own!
    let mut y_src_offsets: [usize; TARGET_SIZE] = [0; TARGET_SIZE];
    let mut y_dst_offsets: [usize; TARGET_SIZE] = [0; TARGET_SIZE];
    let mut x_offsets: [usize; TARGET_SIZE] = [0; TARGET_SIZE];
    
    // Pre-compute Y coordinates
    for y in 0..lb.new_height.min(TARGET_SIZE) {
        y_src_offsets[y] = ((y as f32 * lb.inv_scale) as usize).min(frame.height - 1) * frame.width * 3;
        y_dst_offsets[y] = (y + lb.pad_y as usize) * TARGET_SIZE + lb.pad_x as usize;
    }
    
    // Pre-compute X coordinates  
    for x in 0..lb.new_width.min(TARGET_SIZE) {
        x_offsets[x] = ((x as f32 * lb.inv_scale) as usize).min(frame.width - 1) * 3;
    }

    // Use AVX2 if available, otherwise fall back to the original scalar implementation
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            // Safety: We've checked that the CPU supports the required features.
            // The function itself ensures memory safety by respecting buffer boundaries.
            let output = unsafe {
                preprocess_avx2(frame, precision, &lb, &y_src_offsets, &y_dst_offsets, &x_offsets)
            };
            return Ok(output);
        }
    }
    
    // Fallback for non-x86_64 or CPUs without AVX2/F16C
    preprocess_scalar(frame, precision, &lb, &y_src_offsets, &y_dst_offsets, &x_offsets)
}

/// Original scalar (non-SIMD) implementation for preprocessing.
fn preprocess_scalar(
    frame: &RawFrame,
    precision: InferencePrecision,
    lb: &LetterboxParams,
    y_src_offsets: &[usize; 640],
    y_dst_offsets: &[usize; 640],
    x_offsets: &[usize; 640],
) -> Result<Vec<u8>> {
    const TARGET_SIZE: usize = 640;
    const TARGET_PIXELS: usize = TARGET_SIZE * TARGET_SIZE;
    
    match precision {
        InferencePrecision::FP16 => {
            let gray_val = processing::get_f16_lut()[114];
            let mut output: Vec<u16> = vec![gray_val; TARGET_PIXELS * 3];
            
            let img_ptr = frame.data.as_ptr();
            let out_ptr = output.as_mut_ptr();
            let lut = processing::get_f16_lut();
            
            for y in 0..lb.new_height {
                let src_row = y_src_offsets[y];
                let dst_base = y_dst_offsets[y];
                
                for x in 0..lb.new_width {
                    let src = src_row + x_offsets[x];
                    let dst = dst_base + x;
                    
                    // Safety: Loop bounds `y` and `x` are constrained by `lb.new_height` and
                    // `lb.new_width`, ensuring `src` and `dst` offsets are in-bounds.
                    unsafe {
                        let r = *img_ptr.add(src) as usize;
                        let g = *img_ptr.add(src + 1) as usize;
                        let b = *img_ptr.add(src + 2) as usize;
                        
                        *out_ptr.add(dst) = lut[r];
                        *out_ptr.add(dst + TARGET_PIXELS) = lut[g];
                        *out_ptr.add(dst + TARGET_PIXELS * 2) = lut[b];
                    }
                }
            }
            // Safety: Transmuting a Vec<u16> to Vec<u8> is safe as u16 has alignment >= 1.
            // The new length and capacity are scaled by the size ratio.
            Ok(unsafe {
                let mut u8_vec = std::mem::ManuallyDrop::new(output);
                Vec::from_raw_parts(u8_vec.as_mut_ptr() as *mut u8, u8_vec.len() * 2, u8_vec.capacity() * 2)
            })
        }
        InferencePrecision::FP32 => {
            let gray_val = processing::get_f32_lut()[114];
            let mut output: Vec<f32> = vec![gray_val; TARGET_PIXELS * 3];

            let img_ptr = frame.data.as_ptr();
            let out_ptr = output.as_mut_ptr();
            let lut = processing::get_f32_lut();
            
            for y in 0..lb.new_height {
                let src_row = y_src_offsets[y];
                let dst_base = y_dst_offsets[y];
                
                for x in 0..lb.new_width {
                    let src = src_row + x_offsets[x];
                    let dst = dst_base + x;
                    
                    // Safety: Bounds are checked by the loop logic.
                    unsafe {
                        let r = *img_ptr.add(src) as usize;
                        let g = *img_ptr.add(src + 1) as usize;
                        let b = *img_ptr.add(src + 2) as usize;
                            
                        *out_ptr.add(dst) = lut[r];
                        *out_ptr.add(dst + TARGET_PIXELS) = lut[g];
                        *out_ptr.add(dst + TARGET_PIXELS * 2) = lut[b];
                    }
                }
            }

            // Safety: Transmuting a Vec<f32> to Vec<u8>.
            Ok(unsafe {
                let mut u8_vec = std::mem::ManuallyDrop::new(output);
                Vec::from_raw_parts(u8_vec.as_mut_ptr() as *mut u8, u8_vec.len() * 4, u8_vec.capacity() * 4)
            })
        }
    }
}


/// SIMD-accelerated (AVX2) implementation for preprocessing.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "f16c")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn preprocess_avx2(
    frame: &RawFrame,
    precision: InferencePrecision,
    lb: &LetterboxParams,
    y_src_offsets: &[usize; 640],
    y_dst_offsets: &[usize; 640],
    x_offsets: &[usize; 640],
) -> Vec<u8> {
    const TARGET_SIZE: usize = 640;
    const TARGET_PIXELS: usize = TARGET_SIZE * TARGET_SIZE;

    match precision {
        InferencePrecision::FP16 => {
            let total_bytes = TARGET_PIXELS * 3 * 2;
            let mut output: Vec<u8> = vec![0; total_bytes];

            // Fast gray fill using SIMD - use unaligned stores
            let gray_val_u16 = processing::get_f16_lut()[114];
            let gray_vec = _mm256_set1_epi16(gray_val_u16 as i16);
            let out_ptr_simd = output.as_mut_ptr() as *mut __m256i;
            let chunks = total_bytes / 32;
            for i in 0..chunks {
                _mm256_storeu_si256(out_ptr_simd.add(i), gray_vec);
            }
            // Remainder fill
            let remainder_start = chunks * 32;
            let gray_bytes = gray_val_u16.to_ne_bytes();
            for i in remainder_start..total_bytes {
                output[i] = gray_bytes[i % 2];
            }

            let img_ptr = frame.data.as_ptr();
            let out_ptr_r = output.as_mut_ptr() as *mut u16;
            let out_ptr_g = out_ptr_r.add(TARGET_PIXELS);
            let out_ptr_b = out_ptr_g.add(TARGET_PIXELS);
            
            let norm_vec = _mm256_set1_ps(1.0 / 255.0);

            for y in 0..lb.new_height {
                let src_row = y_src_offsets[y];
                let dst_base = y_dst_offsets[y];
                
                // Add bounds check to prevent buffer overflow
                if dst_base >= TARGET_PIXELS {
                    break;
                }
                
                let mut x = 0;
                while x + 8 <= lb.new_width && dst_base + x + 8 <= TARGET_PIXELS {
                    // Manually de-interleave 8 pixels into local buffers.
                    // This is faster than random-access `_mm_insert_epi8` and avoids complex shuffles.
                    let mut r_buf: [u8; 8] = [0; 8];
                    let mut g_buf: [u8; 8] = [0; 8];
                    let mut b_buf: [u8; 8] = [0; 8];
                    for i in 0..8 {
                        let src = src_row + x_offsets[x + i];
                        r_buf[i] = *img_ptr.add(src);
                        g_buf[i] = *img_ptr.add(src + 1);
                        b_buf[i] = *img_ptr.add(src + 2);
                    }

                    // Load u8 buffers into 128-bit SIMD registers
                    let r_u8x8 = _mm_loadl_epi64(r_buf.as_ptr() as *const __m128i);
                    let g_u8x8 = _mm_loadl_epi64(g_buf.as_ptr() as *const __m128i);
                    let b_u8x8 = _mm_loadl_epi64(b_buf.as_ptr() as *const __m128i);
                    
                    // Widen u8 -> i32
                    let r_i32 = _mm256_cvtepu8_epi32(r_u8x8);
                    let g_i32 = _mm256_cvtepu8_epi32(g_u8x8);
                    let b_i32 = _mm256_cvtepu8_epi32(b_u8x8);

                    // Convert i32 -> f32
                    let r_f32 = _mm256_cvtepi32_ps(r_i32);
                    let g_f32 = _mm256_cvtepi32_ps(g_i32);
                    let b_f32 = _mm256_cvtepi32_ps(b_i32);

                    // Normalize
                    let r_norm = _mm256_mul_ps(r_f32, norm_vec);
                    let g_norm = _mm256_mul_ps(g_f32, norm_vec);
                    let b_norm = _mm256_mul_ps(b_f32, norm_vec);

                    // Convert f32 -> f16. _MM_FROUND_TO_NEAREST_INT is the default rounding mode.
                    let r_f16 = _mm256_cvtps_ph(r_norm, 0);
                    let g_f16 = _mm256_cvtps_ph(g_norm, 0);
                    let b_f16 = _mm256_cvtps_ph(b_norm, 0);

                    // Store results using unaligned stores
                    let dst = dst_base + x;
                    _mm_storeu_si128(out_ptr_r.add(dst) as *mut __m128i, r_f16);
                    _mm_storeu_si128(out_ptr_g.add(dst) as *mut __m128i, g_f16);
                    _mm_storeu_si128(out_ptr_b.add(dst) as *mut __m128i, b_f16);

                    x += 8;
                }
                
                // Scalar remainder for pixels at the end of the row
                let lut = processing::get_f16_lut();
                while x < lb.new_width && dst_base + x < TARGET_PIXELS {
                    let src = src_row + x_offsets[x];
                    let dst = dst_base + x;
                    *out_ptr_r.add(dst) = lut[*img_ptr.add(src) as usize];
                    *out_ptr_g.add(dst) = lut[*img_ptr.add(src + 1) as usize];
                    *out_ptr_b.add(dst) = lut[*img_ptr.add(src + 2) as usize];
                    x += 1;
                }
            }
            output
        }
        InferencePrecision::FP32 => {
            let total_bytes = TARGET_PIXELS * 3 * 4;
            let mut output: Vec<u8> = vec![0; total_bytes];
            
            // Fast gray fill using SIMD - fixed pointer arithmetic
            let gray_val_f32 = processing::get_f32_lut()[114];
            let gray_vec = _mm256_set1_ps(gray_val_f32);
            let out_ptr_simd = output.as_mut_ptr() as *mut f32;
            let f32_chunks = total_bytes / 32; // 32 bytes = 8 f32s
            for i in 0..f32_chunks {
                _mm256_storeu_ps(out_ptr_simd.add(i * 8), gray_vec);
            }
            // Remainder fill
            let remainder_start = f32_chunks * 32;
            let gray_bytes = gray_val_f32.to_ne_bytes();
            for i in remainder_start..total_bytes {
                output[i] = gray_bytes[i % 4];
            }

            let img_ptr = frame.data.as_ptr();
            let out_ptr_r = output.as_mut_ptr() as *mut f32;
            let out_ptr_g = out_ptr_r.add(TARGET_PIXELS);
            let out_ptr_b = out_ptr_g.add(TARGET_PIXELS);

            let norm_vec = _mm256_set1_ps(1.0 / 255.0);

            for y in 0..lb.new_height {
                let src_row = y_src_offsets[y];
                let dst_base = y_dst_offsets[y];
                
                // Add bounds check to prevent buffer overflow
                if dst_base >= TARGET_PIXELS {
                    break;
                }
                
                let mut x = 0;
                while x + 8 <= lb.new_width && dst_base + x + 8 <= TARGET_PIXELS {
                    let mut r_buf: [u8; 8] = [0; 8];
                    let mut g_buf: [u8; 8] = [0; 8];
                    let mut b_buf: [u8; 8] = [0; 8];
                    for i in 0..8 {
                        let src = src_row + x_offsets[x + i];
                        r_buf[i] = *img_ptr.add(src);
                        g_buf[i] = *img_ptr.add(src + 1);
                        b_buf[i] = *img_ptr.add(src + 2);
                    }

                    let r_u8x8 = _mm_loadl_epi64(r_buf.as_ptr() as *const __m128i);
                    let g_u8x8 = _mm_loadl_epi64(g_buf.as_ptr() as *const __m128i);
                    let b_u8x8 = _mm_loadl_epi64(b_buf.as_ptr() as *const __m128i);
                    
                    let r_i32 = _mm256_cvtepu8_epi32(r_u8x8);
                    let g_i32 = _mm256_cvtepu8_epi32(g_u8x8);
                    let b_i32 = _mm256_cvtepu8_epi32(b_u8x8);

                    let r_f32 = _mm256_cvtepi32_ps(r_i32);
                    let g_f32 = _mm256_cvtepi32_ps(g_i32);
                    let b_f32 = _mm256_cvtepi32_ps(b_i32);

                    let r_norm = _mm256_mul_ps(r_f32, norm_vec);
                    let g_norm = _mm256_mul_ps(g_f32, norm_vec);
                    let b_norm = _mm256_mul_ps(b_f32, norm_vec);

                    let dst = dst_base + x;
                    _mm256_storeu_ps(out_ptr_r.add(dst), r_norm);
                    _mm256_storeu_ps(out_ptr_g.add(dst), g_norm);
                    _mm256_storeu_ps(out_ptr_b.add(dst), b_norm);

                    x += 8;
                }

                let lut = processing::get_f32_lut();
                while x < lb.new_width && dst_base + x < TARGET_PIXELS {
                    let src = src_row + x_offsets[x];
                    let dst = dst_base + x;
                    *out_ptr_r.add(dst) = lut[*img_ptr.add(src) as usize];
                    *out_ptr_g.add(dst) = lut[*img_ptr.add(src + 1) as usize];
                    *out_ptr_b.add(dst) = lut[*img_ptr.add(src + 2) as usize];
                    x += 1;
                }
            }
            output
        }
    }
}


/// Performs operations on a given frame, including pre/post processing, inference on the given frame
/// and posting the results to third party services
pub async fn process_frame(
    inference_model: &InferenceModel, 
    source_id: &str,
    source_config: &SourceConfig,
    frame: &RawFrame
) -> Result<FrameProcessStats> {
    let inference_start = Instant::now();

    // Pre process image
    let measure_start = Instant::now();
    let pre_frame = preprocess(&frame, inference_model.model_config().precision)
        .context("Error preprocessing image for YOLO")?;
    let pre_proc_time = measure_start.elapsed();

    // Get raw bboxes from inference
    let measure_start = Instant::now();
    let raw_results = inference_model.infer(pre_frame).await
        .context("Error performing inference for YOLO")?;
    let inference_time = measure_start.elapsed();

    // Process given bboxes
    let measure_start = Instant::now();
    let output_shape: [i64; 2] = inference_model.model_config().output_shape
        .clone()
        .try_into()
        .map_err(|_| anyhow::anyhow!("Output shape is invalid"))?;
    let bboxes = postprocess(
        &raw_results, 
        &frame,
        &output_shape,
        inference_model.model_config().precision,
        source_config.conf_threshold,
        source_config.nms_iou_threshold
    )
        .context("Error postprocessing BBOXes for YOLO")?;
    let post_proc_time = measure_start.elapsed();

    // Populate results
    let measure_start = Instant::now();
    //SourceProcessor::populate_bboxes(source_id, bboxes);
    let results_time = measure_start.elapsed();

    // Create statistics object
    let processing_time = frame.added.elapsed();
    let queue_time = processing_time - inference_start.elapsed();
    let stats = FrameProcessStats {
        queue: queue_time.as_micros() as u64,
        pre_processing: pre_proc_time.as_micros() as u64, 
        inference: inference_time.as_micros() as u64, 
        post_processing: post_proc_time.as_micros() as u64, 
        results: results_time.as_micros() as u64,
        processing: processing_time.as_micros() as u64
    };

    Ok(stats)
}