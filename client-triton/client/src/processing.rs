use std::io::{Error, ErrorKind};

// Custom modules
use crate::inference::{InferencePrecision, InferenceResult};

// Fast f16 to f32 conversion using bit manipulation
fn f16_to_f32(f16_val: u16) -> f32 {
    let sign = (f16_val >> 15) & 0x1;
    let exp = (f16_val >> 10) & 0x1f;
    let frac = f16_val & 0x3ff;
    
    if exp == 0 {
        if frac == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        // Denormal
        let mut val = frac as f32 / 1024.0 / 16384.0;
        if sign == 1 { val = -val; }
        return val;
    } else if exp == 31 {
        // Infinity or NaN
        return if frac == 0 {
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else {
            f32::NAN
        };
    }
    
    // Normal numbers
    let exp_f32 = (exp as i32 - 15 + 127) as u32;
    let frac_f32 = (frac as u32) << 13;
    let bits = (sign as u32) << 31 | exp_f32 << 23 | frac_f32;
    f32::from_bits(bits)
}

fn calculate_letterbox_params(height: usize, width: usize) -> (usize, usize, f32, f32, f32) {
    const TARGET_SIZE: f32 = 640.0;
    let scale = (TARGET_SIZE / width as f32).min(TARGET_SIZE / height as f32);
    let new_width = (width as f32 * scale) as usize;
    let new_height = (height as f32 * scale) as usize;
    let pad_x = (TARGET_SIZE as usize - new_width) as f32 * 0.5;
    let pad_y = (TARGET_SIZE as usize - new_height) as f32 * 0.5;
    (new_width, new_height, pad_x, pad_y, 1.0 / scale)
}

fn bbox_nms(mut detections: Vec<InferenceResult>, nms_iou_threshold: f32) -> Vec<InferenceResult> {
    let len = detections.len();
    if len <= 1 {
        return detections;
    }
    
    // Sort by score descending - unstable for speed
    detections.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    
    let mut keep = Vec::with_capacity(len);
    let mut suppressed = vec![false; len];
    
    for i in 0..len {
        if unsafe { *suppressed.get_unchecked(i) } {
            continue;
        }
        
        let detection_i = unsafe { *detections.get_unchecked(i) };
        keep.push(detection_i);
        
        let bbox1 = &detection_i.bbox;
        let class1 = detection_i.class;
        
        // Pre-calculate bbox1 area
        let bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
        
        // Check remaining detections of same class only
        for j in (i + 1)..len {
            if unsafe { *suppressed.get_unchecked(j) } {
                continue;
            }
            
            let detection_j = unsafe { detections.get_unchecked(j) };
            
            if class1 != detection_j.class {
                continue;
            }
            
            let bbox2 = &detection_j.bbox;
            
            // Inline IoU calculation for speed
            let x1_max = bbox1[0].max(bbox2[0]);
            let y1_max = bbox1[1].max(bbox2[1]);
            let x2_min = bbox1[2].min(bbox2[2]);
            let y2_min = bbox1[3].min(bbox2[3]);
            
            let intersection_width = x2_min - x1_max;
            let intersection_height = y2_min - y1_max;
            
            if intersection_width > 0.0 && intersection_height > 0.0 {
                let intersection_area = intersection_width * intersection_height;
                let bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
                let union_area = bbox1_area + bbox2_area - intersection_area;
                let iou = intersection_area / union_area;
                
                if iou > nms_iou_threshold {
                    unsafe { *suppressed.get_unchecked_mut(j) = true; }
                }
            }
        }
    }
    
    keep
}

pub fn postprocess_yolo(
    results: &[u8],
    image_height: usize,
    image_width: usize,
    output_shape: &[i64; 2],
    precision: InferencePrecision,
    pred_conf_threshold: f32,
    nms_iou_threshold: f32,
) -> Result<Vec<InferenceResult>, Error> {
    let target_features = output_shape[0] as usize;
    let target_anchors = output_shape[1] as usize;
    let target_classes = target_features - 4;
    
    // Validate input size
    let expected_size = match precision {
        InferencePrecision::FP16 => target_anchors * target_features * 2,
        InferencePrecision::FP32 => target_anchors * target_features * 4,
    };
    
    if results.len() != expected_size {
        return Err(Error::new(
            ErrorKind::Other,
            format!(
                "Got unexpected size of model output ({}). Got {}, expected {}",
                precision.to_string(),
                results.len(),
                expected_size
            ),
        ));
    }
    
    // Calculate letterbox parameters once
    let (_, _, pad_x, pad_y, inv_scale) = calculate_letterbox_params(image_height, image_width);
    
    let mut detections = Vec::with_capacity(target_anchors.min(1000)); // Cap initial capacity
    
    match precision {
        InferencePrecision::FP16 => {
            // Cast to u16 slice for f16 data
            let u16_data = unsafe {
                std::slice::from_raw_parts(results.as_ptr() as *const u16, results.len() / 2)
            };
            
            // Process all anchors in single loop
            for anchor_idx in 0..target_anchors {
                // Calculate indices for transposed access pattern
                let x_idx = anchor_idx;
                let y_idx = target_anchors + anchor_idx;
                let w_idx = 2 * target_anchors + anchor_idx;
                let h_idx = 3 * target_anchors + anchor_idx;
                
                if h_idx >= u16_data.len() {
                    break;
                }
                
                unsafe {
                    // Convert f16 to f32 using fast bit manipulation
                    let x = f16_to_f32(*u16_data.get_unchecked(x_idx));
                    let y = f16_to_f32(*u16_data.get_unchecked(y_idx));
                    let w = f16_to_f32(*u16_data.get_unchecked(w_idx));
                    let h = f16_to_f32(*u16_data.get_unchecked(h_idx));
                    
                    // Convert center coordinates to corners and scale back
                    let half_w = w * 0.5;
                    let half_h = h * 0.5;
                    let final_x1 = (x - half_w - pad_x) * inv_scale;
                    let final_y1 = (y - half_h - pad_y) * inv_scale;
                    let final_x2 = (x + half_w - pad_x) * inv_scale;
                    let final_y2 = (y + half_h - pad_y) * inv_scale;
                    
                    // Find max class score - unrolled for small class counts
                    let mut max_score = 0.0f32;
                    let mut max_class = 0usize;
                    
                    let class_start_idx = 4 * target_anchors + anchor_idx;
                    for class_idx in 0..target_classes {
                        let prob_idx = class_start_idx + class_idx * target_anchors;
                        if prob_idx < u16_data.len() {
                            let score = f16_to_f32(*u16_data.get_unchecked(prob_idx));
                            if score > max_score {
                                max_score = score;
                                max_class = class_idx;
                            }
                        }
                    }
                    
                    // Early confidence filtering
                    if max_score >= pred_conf_threshold {
                        detections.push(InferenceResult {
                            bbox: [final_x1, final_y1, final_x2, final_y2],
                            class: max_class,
                            score: max_score,
                        });
                    }
                }
            }
        }
        InferencePrecision::FP32 => {
            // Cast to f32 slice
            let f32_data = unsafe {
                std::slice::from_raw_parts(results.as_ptr() as *const f32, results.len() / 4)
            };
            
            // Process all anchors in single loop
            for anchor_idx in 0..target_anchors {
                // Calculate indices for transposed access pattern
                let x_idx = anchor_idx;
                let y_idx = target_anchors + anchor_idx;
                let w_idx = 2 * target_anchors + anchor_idx;
                let h_idx = 3 * target_anchors + anchor_idx;
                
                if h_idx >= f32_data.len() {
                    break;
                }
                
                unsafe {
                    let x = *f32_data.get_unchecked(x_idx);
                    let y = *f32_data.get_unchecked(y_idx);
                    let w = *f32_data.get_unchecked(w_idx);
                    let h = *f32_data.get_unchecked(h_idx);
                    
                    // Convert center coordinates to corners and scale back
                    let half_w = w * 0.5;
                    let half_h = h * 0.5;
                    let final_x1 = (x - half_w - pad_x) * inv_scale;
                    let final_y1 = (y - half_h - pad_y) * inv_scale;
                    let final_x2 = (x + half_w - pad_x) * inv_scale;
                    let final_y2 = (y + half_h - pad_y) * inv_scale;
                    
                    // Find max class score
                    let mut max_score = 0.0f32;
                    let mut max_class = 0usize;
                    
                    let class_start_idx = 4 * target_anchors + anchor_idx;
                    for class_idx in 0..target_classes {
                        let prob_idx = class_start_idx + class_idx * target_anchors;
                        if prob_idx < f32_data.len() {
                            let score = *f32_data.get_unchecked(prob_idx);
                            if score > max_score {
                                max_score = score;
                                max_class = class_idx;
                            }
                        }
                    }
                    
                    // Early confidence filtering
                    if max_score >= pred_conf_threshold {
                        detections.push(InferenceResult {
                            bbox: [final_x1, final_y1, final_x2, final_y2],
                            class: max_class,
                            score: max_score,
                        });
                    }
                }
            }
        }
    }
    
    // Perform NMS only if we have detections
    if !detections.is_empty() {
        detections = bbox_nms(detections, nms_iou_threshold);
    }
    
    Ok(detections)
}

// Pre-computed lookup tables
static F16_LUT: [u16; 256] = {
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
};

static F32_LUT: [f32; 256] = {
    let mut lut = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        lut[i] = i as f32 / 255.0;
        i += 1;
    }
    lut
};

pub fn preprocess_yolo(
    image: &[u8],
    image_height: usize,
    image_width: usize,
    input_shape: &[i64; 3],
    precision: InferencePrecision,
) -> Result<Vec<u8>, Error> {
    let target_size: usize = input_shape[1] as usize;
    let target_pixels = target_size * target_size;
    
    // Calculate letterbox params
    let (new_width, new_height, pad_x, pad_y, inv_scale) = calculate_letterbox_params(image_height, image_width);
    let pad_x = pad_x as usize;
    let pad_y = pad_y as usize;
    
    // Pre-compute mappings - ONCE
    let mut y_indices = Vec::with_capacity(new_height);
    let mut x_indices = Vec::with_capacity(new_width);
    let mut row_offsets = Vec::with_capacity(new_height);
    
    for y in 0..new_height {
        let orig_y = ((y as f32 * inv_scale) as usize).min(image_height - 1);
        y_indices.push(orig_y);
        row_offsets.push(orig_y * image_width);
    }
    
    for x in 0..new_width {
        x_indices.push(((x as f32 * inv_scale) as usize).min(image_width - 1));
    }
    
    match precision {
        InferencePrecision::FP16 => {
            // Single allocation with gray padding
            let mut output: Vec<u16> = vec![F16_LUT[114]; target_pixels * 3];
            let output_ptr = output.as_mut_ptr();
            
            unsafe {
                let img_ptr = image.as_ptr();
                let lut_ptr = F16_LUT.as_ptr();
                
                // SINGLE THREADED - no overhead, maximum speed
                for y in 0..new_height {
                    let target_y = y + pad_y;
                    let orig_row_base = *row_offsets.get_unchecked(y);
                    let target_row_base = target_y * target_size + pad_x;
                    
                    // 8-pixel unrolling for ILP
                    let mut x = 0;
                    while x + 7 < new_width {
                        for i in 0..8 {
                            let target_idx = target_row_base + x + i;
                            let orig_idx = (orig_row_base + *x_indices.get_unchecked(x + i)) * 3;
                            
                            let r = *img_ptr.add(orig_idx) as usize;
                            let g = *img_ptr.add(orig_idx + 1) as usize;
                            let b = *img_ptr.add(orig_idx + 2) as usize;
                            
                            // CHW format
                            *output_ptr.add(target_idx) = *lut_ptr.add(r);
                            *output_ptr.add(target_idx + target_pixels) = *lut_ptr.add(g);
                            *output_ptr.add(target_idx + target_pixels * 2) = *lut_ptr.add(b);
                        }
                        x += 8;
                    }
                    
                    // Remaining pixels
                    while x < new_width {
                        let target_idx = target_row_base + x;
                        let orig_idx = (orig_row_base + *x_indices.get_unchecked(x)) * 3;
                        
                        let r = *img_ptr.add(orig_idx) as usize;
                        let g = *img_ptr.add(orig_idx + 1) as usize;
                        let b = *img_ptr.add(orig_idx + 2) as usize;
                        
                        *output_ptr.add(target_idx) = *lut_ptr.add(r);
                        *output_ptr.add(target_idx + target_pixels) = *lut_ptr.add(g);
                        *output_ptr.add(target_idx + target_pixels * 2) = *lut_ptr.add(b);
                        
                        x += 1;
                    }
                }
            }
            
            // Zero-copy conversion
            Ok(unsafe {
                std::slice::from_raw_parts(
                    output.as_ptr() as *const u8,
                    output.len() * 2
                ).to_vec()
            })
        }
        InferencePrecision::FP32 => {
            // Single allocation with gray padding
            let mut output: Vec<f32> = vec![F32_LUT[114]; target_pixels * 3];
            let output_ptr = output.as_mut_ptr();
            
            unsafe {
                let img_ptr = image.as_ptr();
                let lut_ptr = F32_LUT.as_ptr();
                
                // SINGLE THREADED - maximum speed
                for y in 0..new_height {
                    let target_y = y + pad_y;
                    let orig_row_base = *row_offsets.get_unchecked(y);
                    let target_row_base = target_y * target_size + pad_x;
                    
                    // 8-pixel unrolling
                    let mut x = 0;
                    while x + 7 < new_width {
                        for i in 0..8 {
                            let target_idx = target_row_base + x + i;
                            let orig_idx = (orig_row_base + *x_indices.get_unchecked(x + i)) * 3;
                            
                            let r = *img_ptr.add(orig_idx) as usize;
                            let g = *img_ptr.add(orig_idx + 1) as usize;
                            let b = *img_ptr.add(orig_idx + 2) as usize;
                            
                            // CHW format
                            *output_ptr.add(target_idx) = *lut_ptr.add(r);
                            *output_ptr.add(target_idx + target_pixels) = *lut_ptr.add(g);
                            *output_ptr.add(target_idx + target_pixels * 2) = *lut_ptr.add(b);
                        }
                        x += 8;
                    }
                    
                    // Remaining pixels
                    while x < new_width {
                        let target_idx = target_row_base + x;
                        let orig_idx = (orig_row_base + *x_indices.get_unchecked(x)) * 3;
                        
                        let r = *img_ptr.add(orig_idx) as usize;
                        let g = *img_ptr.add(orig_idx + 1) as usize;
                        let b = *img_ptr.add(orig_idx + 2) as usize;
                        
                        *output_ptr.add(target_idx) = *lut_ptr.add(r);
                        *output_ptr.add(target_idx + target_pixels) = *lut_ptr.add(g);
                        *output_ptr.add(target_idx + target_pixels * 2) = *lut_ptr.add(b);
                        
                        x += 1;
                    }
                }
            }
            
            // Zero-copy conversion
            Ok(unsafe {
                std::slice::from_raw_parts(
                    output.as_ptr() as *const u8,
                    output.len() * 4
                ).to_vec()
            })
        }
    }
}
