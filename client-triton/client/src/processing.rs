use std::sync::OnceLock;
use anyhow::{Result};

// Custom modules
use crate::inference::{InferencePrecision, InferenceResult, InferenceFrame};

// Thread-safe immutable lookup table for f16 to f32 conversion
static F16_TO_F32_LUT: OnceLock<Box<[f32; 65536]>> = OnceLock::new();

fn get_f16_lut() -> &'static [f32; 65536] {
    F16_TO_F32_LUT.get_or_init(|| {
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
    })
}

// Inline f16 lookup - single memory access, no computation
#[inline(always)]
fn f16_to_f32_fast(val: u16) -> f32 {
    unsafe {
        *get_f16_lut().get_unchecked(val as usize)
    }
}

// Precompute letterbox params struct for better cache locality
#[derive(Copy, Clone)]
struct LetterboxParams {
    pad_x: f32,
    pad_y: f32,
    inv_scale: f32,
}

#[inline(always)]
fn calculate_letterbox_fast(height: usize, width: usize) -> LetterboxParams {
    const TARGET_SIZE: f32 = 640.0;
    const TARGET_SIZE_HALF: f32 = 320.0;
    
    let w_inv = 1.0 / width as f32;
    let h_inv = 1.0 / height as f32;
    let scale = TARGET_SIZE * w_inv.min(h_inv);
    
    LetterboxParams {
        pad_x: TARGET_SIZE_HALF - width as f32 * scale * 0.5,
        pad_y: TARGET_SIZE_HALF - height as f32 * scale * 0.5,
        inv_scale: 1.0 / scale,
    }
}

// Optimized NMS with early exit and better memory access patterns
#[inline(never)] // Don't inline to keep instruction cache hot for main loop
fn bbox_nms_fast(detections: &mut Vec<InferenceResult>, nms_threshold: f32) {
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
                let area_i = (detection_i.bbox[2] - detection_i.bbox[0]) * 
                            (detection_i.bbox[3] - detection_i.bbox[1]);
                let area_j = (kept.bbox[2] - kept.bbox[0]) * 
                            (kept.bbox[3] - kept.bbox[1]);
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

pub fn postprocess_yolo(
    results: &[u8],
    original_frame: &InferenceFrame,
    output_shape: &[i64; 2],
    precision: InferencePrecision,
    pred_conf_threshold: f32,
    nms_iou_threshold: f32,
) -> Result<Vec<InferenceResult>> {
    // Initialize LUT once (thread-safe)
    get_f16_lut();
    
    let target_features = output_shape[0] as usize;
    let target_anchors = output_shape[1] as usize;
    let target_classes = target_features - 4;
    
    // Validate size
    let expected_size = match precision {
        InferencePrecision::FP16 => target_anchors * target_features * 2,
        InferencePrecision::FP32 => target_anchors * target_features * 4,
    };
    
    if results.len() != expected_size {
        anyhow::bail!(
            format!(
                "Got unexpected size of model output ({}). Got {}, expected {}",
                precision.to_string(),
                results.len(),
                expected_size
            )
        );
    }
    
    // Precompute letterbox parameters
    let lb = calculate_letterbox_fast(original_frame.height, original_frame.width);
    
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
                    let x = f16_to_f32_fast(*u16_data.get_unchecked(anchor_idx));
                    let y = f16_to_f32_fast(*u16_data.get_unchecked(stride1 + anchor_idx));
                    let w = f16_to_f32_fast(*u16_data.get_unchecked(stride2 + anchor_idx));
                    let h = f16_to_f32_fast(*u16_data.get_unchecked(stride3 + anchor_idx));
                    
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
                    
                    // Unroll for common class counts (80 for COCO)
                    if target_classes == 80 {
                        // Process 4 classes at a time for better pipelining
                        for class_idx in (0..80).step_by(4) {
                            let idx0 = class_base + class_idx * stride1;
                            let idx1 = class_base + (class_idx + 1) * stride1;
                            let idx2 = class_base + (class_idx + 2) * stride1;
                            let idx3 = class_base + (class_idx + 3) * stride1;
                            
                            let s0 = f16_to_f32_fast(*u16_data.get_unchecked(idx0));
                            let s1 = f16_to_f32_fast(*u16_data.get_unchecked(idx1));
                            let s2 = f16_to_f32_fast(*u16_data.get_unchecked(idx2));
                            let s3 = f16_to_f32_fast(*u16_data.get_unchecked(idx3));
                            
                            if s0 > max_score { max_score = s0; max_class = class_idx; }
                            if s1 > max_score { max_score = s1; max_class = class_idx + 1; }
                            if s2 > max_score { max_score = s2; max_class = class_idx + 2; }
                            if s3 > max_score { max_score = s3; max_class = class_idx + 3; }
                        }
                    } else {
                        // Generic path for other class counts
                        for class_idx in 0..target_classes {
                            let prob_idx = class_base + class_idx * stride1;
                            let score = f16_to_f32_fast(*u16_data.get_unchecked(prob_idx));
                            if score > max_score {
                                max_score = score;
                                max_class = class_idx;
                            }
                        }
                    }
                    
                    // Only store if above threshold
                    if max_score >= pred_conf_threshold {
                        detections.push(InferenceResult {
                            bbox: [x1, y1, x2, y2],
                            class: max_class,
                            score: max_score,
                        });
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
                    
                    if target_classes == 80 {
                        // Unrolled for COCO
                        for class_idx in (0..80).step_by(4) {
                            let idx0 = class_base + class_idx * stride1;
                            let idx1 = class_base + (class_idx + 1) * stride1;
                            let idx2 = class_base + (class_idx + 2) * stride1;
                            let idx3 = class_base + (class_idx + 3) * stride1;
                            
                            let s0 = *f32_data.get_unchecked(idx0);
                            let s1 = *f32_data.get_unchecked(idx1);
                            let s2 = *f32_data.get_unchecked(idx2);
                            let s3 = *f32_data.get_unchecked(idx3);
                            
                            if s0 > max_score { max_score = s0; max_class = class_idx; }
                            if s1 > max_score { max_score = s1; max_class = class_idx + 1; }
                            if s2 > max_score { max_score = s2; max_class = class_idx + 2; }
                            if s3 > max_score { max_score = s3; max_class = class_idx + 3; }
                        }
                    } else {
                        for class_idx in 0..target_classes {
                            let prob_idx = class_base + class_idx * stride1;
                            let score = *f32_data.get_unchecked(prob_idx);
                            if score > max_score {
                                max_score = score;
                                max_class = class_idx;
                            }
                        }
                    }
                    
                    if max_score >= pred_conf_threshold {
                        detections.push(InferenceResult {
                            bbox: [x1, y1, x2, y2],
                            class: max_class,
                            score: max_score,
                        });
                    }
                }
            }
        }
    }
    
    // Fast NMS only if needed
    if detections.len() > 1 {
        bbox_nms_fast(&mut detections, nms_iou_threshold);
    }
    
    Ok(detections)
}

// Cache-aligned LUTs - these are IMMUTABLE and thread-safe
#[repr(align(64))]
struct AlignedF16Lut([u16; 256]);

#[repr(align(64))]
struct AlignedF32Lut([f32; 256]);

static F16_LUT: AlignedF16Lut = AlignedF16Lut({
    let mut lut = [0u16; 256];
    let mut i = 0;
    while i < 256 {
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
        i += 1;
    }
    lut
});

static F32_LUT: AlignedF32Lut = AlignedF32Lut({
    let mut lut = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        lut[i] = i as f32 / 255.0;
        i += 1;
    }
    lut
});

// THREAD-SAFE version - no global mutable state!
pub fn preprocess_yolo(
    frame: &InferenceFrame,
    input_shape: &[i64; 3],
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
    let target_size = input_shape[1] as usize;
    let target_pixels = target_size * target_size;

    // Fast letterbox calculation
    let scale = (target_size as f32) / (frame.height.max(frame.width) as f32);
    let new_width = ((frame.width as f32 * scale) as usize).min(target_size);
    let new_height = ((frame.height as f32 * scale) as usize).min(target_size);
    let pad_x = (target_size - new_width) >> 1;
    let pad_y = (target_size - new_height) >> 1;
    let inv_scale = 1.0 / scale;
    
    // Stack-allocated coordinate buffers - each thread gets its own!
    const MAX_SIZE: usize = 640;
    let mut y_src_offsets: [usize; MAX_SIZE] = [0; MAX_SIZE];
    let mut y_dst_offsets: [usize; MAX_SIZE] = [0; MAX_SIZE];
    let mut x_offsets: [usize; MAX_SIZE] = [0; MAX_SIZE];
    
    // Pre-compute Y coordinates
    for y in 0..new_height.min(MAX_SIZE) {
        y_src_offsets[y] = ((y as f32 * inv_scale) as usize).min(frame.height - 1) * frame.width * 3;
        y_dst_offsets[y] = (y + pad_y) * target_size + pad_x;
    }
    
    // Pre-compute X coordinates  
    for x in 0..new_width.min(MAX_SIZE) {
        x_offsets[x] = ((x as f32 * inv_scale) as usize).min(frame.width - 1) * 3;
    }
    
    match precision {
        InferencePrecision::FP16 => {
            // Direct byte allocation - no transmute needed
            let gray_val = F16_LUT.0[114];
            let gray_bytes: [u8; 2] = unsafe { std::mem::transmute(gray_val) };
            
            let mut output: Vec<u8> = Vec::with_capacity(target_pixels * 6);
            unsafe { output.set_len(target_pixels * 6) };
            
            // Fast gray fill using 64-bit writes
            unsafe {
                let ptr = output.as_mut_ptr() as *mut u64;
                let gray_pattern = u64::from_ne_bytes([
                    gray_bytes[0], gray_bytes[1], gray_bytes[0], gray_bytes[1],
                    gray_bytes[0], gray_bytes[1], gray_bytes[0], gray_bytes[1],
                ]);
                
                let chunks = (target_pixels * 6) >> 3; // div by 8
                for i in 0..chunks {
                    *ptr.add(i) = gray_pattern;
                }
                
                // Handle remaining bytes
                let remainder = (target_pixels * 6) & 7;
                if remainder > 0 {
                    let start = chunks << 3;
                    for i in 0..remainder {
                        *output.as_mut_ptr().add(start + i) = gray_bytes[i & 1];
                    }
                }
            }
            
            unsafe {
                let img_ptr = frame.data.as_ptr();
                let out_ptr = output.as_mut_ptr() as *mut u16;
                let lut = F16_LUT.0.as_ptr();
                
                for y in 0..new_height {
                    let src_row = y_src_offsets[y];
                    let dst_base = y_dst_offsets[y];
                    
                    let mut x = 0;
                    
                    // 32x unroll for maximum ILP
                    while x + 32 <= new_width {
                        for i in 0..32 {
                            let src = src_row + x_offsets[x + i];
                            let dst = dst_base + x + i;
                            
                            let r = *img_ptr.add(src) as usize;
                            let g = *img_ptr.add(src + 1) as usize;
                            let b = *img_ptr.add(src + 2) as usize;
                            
                            *out_ptr.add(dst) = *lut.add(r);
                            *out_ptr.add(dst + target_pixels) = *lut.add(g);
                            *out_ptr.add(dst + (target_pixels << 1)) = *lut.add(b);
                        }
                        x += 32;
                    }
                    
                    // 8x unroll for remainder
                    while x + 8 <= new_width {
                        for i in 0..8 {
                            let src = src_row + x_offsets[x + i];
                            let dst = dst_base + x + i;
                            
                            let r = *img_ptr.add(src) as usize;
                            let g = *img_ptr.add(src + 1) as usize;
                            let b = *img_ptr.add(src + 2) as usize;
                            
                            *out_ptr.add(dst) = *lut.add(r);
                            *out_ptr.add(dst + target_pixels) = *lut.add(g);
                            *out_ptr.add(dst + (target_pixels << 1)) = *lut.add(b);
                        }
                        x += 8;
                    }
                    
                    // Final pixels
                    while x < new_width {
                        let src = src_row + x_offsets[x];
                        let dst = dst_base + x;
                        
                        let r = *img_ptr.add(src) as usize;
                        let g = *img_ptr.add(src + 1) as usize;
                        let b = *img_ptr.add(src + 2) as usize;
                        
                        *out_ptr.add(dst) = *lut.add(r);
                        *out_ptr.add(dst + target_pixels) = *lut.add(g);
                        *out_ptr.add(dst + (target_pixels << 1)) = *lut.add(b);
                        
                        x += 1;
                    }
                }
            }
            
            Ok(output)
        }
        InferencePrecision::FP32 => {
            // Direct byte allocation
            let gray_val = F32_LUT.0[114];
            let gray_bytes: [u8; 4] = unsafe { std::mem::transmute(gray_val) };
            
            let mut output: Vec<u8> = Vec::with_capacity(target_pixels * 12);
            unsafe { output.set_len(target_pixels * 12) };
            
            // Fast gray fill using 128-bit writes where possible
            unsafe {
                let ptr = output.as_mut_ptr();
                
                // Try 128-bit writes if available
                let ptr_128 = ptr as *mut u128;
                let gray_pattern = u128::from_ne_bytes([
                    gray_bytes[0], gray_bytes[1], gray_bytes[2], gray_bytes[3],
                    gray_bytes[0], gray_bytes[1], gray_bytes[2], gray_bytes[3],
                    gray_bytes[0], gray_bytes[1], gray_bytes[2], gray_bytes[3],
                    gray_bytes[0], gray_bytes[1], gray_bytes[2], gray_bytes[3],
                ]);
                
                let chunks = (target_pixels * 12) >> 4; // div by 16
                for i in 0..chunks {
                    *ptr_128.add(i) = gray_pattern;
                }
                
                // Handle remainder with 32-bit writes
                let remainder_start = chunks << 4;
                let remainder = (target_pixels * 12) - remainder_start;
                if remainder > 0 {
                    let ptr_32 = ptr.add(remainder_start) as *mut u32;
                    let gray_u32 = u32::from_ne_bytes(gray_bytes);
                    for i in 0..(remainder >> 2) {
                        *ptr_32.add(i) = gray_u32;
                    }
                }
            }
            
            unsafe {
                let img_ptr = frame.data.as_ptr();
                let out_ptr = output.as_mut_ptr() as *mut f32;
                let lut = F32_LUT.0.as_ptr();
                
                for y in 0..new_height {
                    let src_row = y_src_offsets[y];
                    let dst_base = y_dst_offsets[y];
                    
                    let mut x = 0;
                    
                    // 32x unroll
                    while x + 32 <= new_width {
                        for i in 0..32 {
                            let src = src_row + x_offsets[x + i];
                            let dst = dst_base + x + i;
                            
                            let r = *img_ptr.add(src) as usize;
                            let g = *img_ptr.add(src + 1) as usize;
                            let b = *img_ptr.add(src + 2) as usize;
                            
                            *out_ptr.add(dst) = *lut.add(r);
                            *out_ptr.add(dst + target_pixels) = *lut.add(g);
                            *out_ptr.add(dst + (target_pixels << 1)) = *lut.add(b);
                        }
                        x += 32;
                    }
                    
                    // 8x unroll
                    while x + 8 <= new_width {
                        for i in 0..8 {
                            let src = src_row + x_offsets[x + i];
                            let dst = dst_base + x + i;
                            
                            let r = *img_ptr.add(src) as usize;
                            let g = *img_ptr.add(src + 1) as usize;
                            let b = *img_ptr.add(src + 2) as usize;
                            
                            *out_ptr.add(dst) = *lut.add(r);
                            *out_ptr.add(dst + target_pixels) = *lut.add(g);
                            *out_ptr.add(dst + (target_pixels << 1)) = *lut.add(b);
                        }
                        x += 8;
                    }
                    
                    while x < new_width {
                        let src = src_row + x_offsets[x];
                        let dst = dst_base + x;
                        
                        let r = *img_ptr.add(src) as usize;
                        let g = *img_ptr.add(src + 1) as usize;
                        let b = *img_ptr.add(src + 2) as usize;
                        
                        *out_ptr.add(dst) = *lut.add(r);
                        *out_ptr.add(dst + target_pixels) = *lut.add(g);
                        *out_ptr.add(dst + (target_pixels << 1)) = *lut.add(b);
                        
                        x += 1;
                    }
                }
            }
            
            Ok(output)
        }
    }
}