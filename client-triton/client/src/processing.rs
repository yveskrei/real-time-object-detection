use std::io::{Error, ErrorKind};
use std::cell::RefCell;
use std::collections::HashMap;
use half::f16;

// Custom modules
use crate::inference::{InferencePrecision, InferenceResult};

thread_local! {
    static OUTPUT_BUFFER_U16: RefCell<Vec<u16>> = RefCell::new(Vec::with_capacity(640 * 640 * 3));
    static OUTPUT_BUFFER_F32: RefCell<Vec<f32>> = RefCell::new(Vec::with_capacity(640 * 640 * 3));
    static MAPPING_CACHE: RefCell<HashMap<(usize, usize), FastMapping>> = RefCell::new(HashMap::new());
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

// Enhanced mapping cache with bilinear interpolation support
#[derive(Clone)]
struct FastMapping {
    y_indices: Vec<usize>,
    x_indices: Vec<usize>,
    row_offsets: Vec<usize>,
}

fn calculate_letterbox_params(height: usize, width: usize) -> (usize, usize, usize, usize, f32) {
    const TARGET_SIZE: usize = 640;
    let scale = (TARGET_SIZE as f32 / width as f32).min(TARGET_SIZE as f32 / height as f32);
    let new_width = (width as f32 * scale) as usize;
    let new_height = (height as f32 * scale) as usize;
    let pad_x = (TARGET_SIZE - new_width) / 2;
    let pad_y = (TARGET_SIZE - new_height) / 2;
    (new_width, new_height, pad_x, pad_y, 1.0 / scale)
}

fn generate_fast_mapping(height: usize, width: usize, new_height: usize, new_width: usize, inv_scale: f32) -> FastMapping {
    let mut y_indices = Vec::with_capacity(new_height);
    let mut x_indices = Vec::with_capacity(new_width);
    let mut row_offsets = Vec::with_capacity(new_height);
    
    for y in 0..new_height {
        let orig_y = ((y as f32 * inv_scale) as usize).min(height - 1);
        y_indices.push(orig_y);
        row_offsets.push(orig_y * width);
    }
    
    for x in 0..new_width {
        x_indices.push(((x as f32 * inv_scale) as usize).min(width - 1));
    }
    
    FastMapping { y_indices, x_indices, row_offsets }
}

fn preprocess_yolo_fp16(    
    image: &[u8], 
    image_height: usize, 
    image_width: usize, 
    target_size: usize, 
    target_pixels: usize
) -> Result<Vec<u8>, Error> {  
    let (new_width, new_height, pad_x, pad_y, inv_scale) = calculate_letterbox_params(image_height, image_width);
    
    OUTPUT_BUFFER_U16.with(|buffer_cell| {
        MAPPING_CACHE.with(|cache_cell| {
            let mut buffer = buffer_cell.borrow_mut();
            let mut cache = cache_cell.borrow_mut();
            
            // Pre-allocate and clear buffer
            if buffer.capacity() < target_pixels * 3 {
                buffer.reserve(target_pixels * 3);
            }
            buffer.clear();
            buffer.resize(target_pixels * 3, F16_LUT[114]);
            
            // Get or create cached mapping
            let cache_key = (image_height, image_width);
            let mapping = cache.entry(cache_key).or_insert_with(|| {
                generate_fast_mapping(image_height, image_width, new_height, new_width, inv_scale)
            });
            
            // Split into planes for better cache locality
            let buffer_ptr = buffer.as_mut_ptr();
            
            unsafe {
                let img_ptr = image.as_ptr();
                let lut_ptr = F16_LUT.as_ptr();
                
                // Process in chunks for better instruction-level parallelism
                for y in 0..new_height {
                    let target_y = y + pad_y;
                    let orig_row_base = *mapping.row_offsets.get_unchecked(y);
                    let target_row_base = target_y * target_size + pad_x;
                    
                    // Process 8 pixels at once for maximum throughput
                    let mut x = 0;
                    while x + 7 < new_width {
                        // Unroll 8 pixels for maximum ILP
                        for i in 0..8 {
                            let target_idx = target_row_base + x + i;
                            let orig_idx = (orig_row_base + *mapping.x_indices.get_unchecked(x + i)) * 3;
                            
                            let r = *img_ptr.add(orig_idx) as usize;
                            let g = *img_ptr.add(orig_idx + 1) as usize;
                            let b = *img_ptr.add(orig_idx + 2) as usize;
                            
                            // Direct pointer arithmetic for maximum speed
                            *buffer_ptr.add(target_idx) = *lut_ptr.add(r);
                            *buffer_ptr.add(target_idx + target_pixels) = *lut_ptr.add(g);
                            *buffer_ptr.add(target_idx + target_pixels * 2) = *lut_ptr.add(b);
                        }
                        x += 8;
                    }
                    
                    // Handle remaining pixels
                    while x < new_width {
                        let target_idx = target_row_base + x;
                        let orig_idx = (orig_row_base + *mapping.x_indices.get_unchecked(x)) * 3;
                        
                        let r = *img_ptr.add(orig_idx) as usize;
                        let g = *img_ptr.add(orig_idx + 1) as usize;
                        let b = *img_ptr.add(orig_idx + 2) as usize;
                        
                        *buffer_ptr.add(target_idx) = *lut_ptr.add(r);
                        *buffer_ptr.add(target_idx + target_pixels) = *lut_ptr.add(g);
                        *buffer_ptr.add(target_idx + target_pixels * 2) = *lut_ptr.add(b);
                        
                        x += 1;
                    }
                }
            }
            
            // Safe conversion to bytes
            Ok(unsafe {
                std::slice::from_raw_parts(
                    buffer.as_ptr() as *const u8,
                    buffer.len() * 2
                ).to_vec()
            })
        })
    })
}

fn preprocess_yolo_fp32(
    image: &[u8], 
    image_height: usize, 
    image_width: usize, 
    target_size: usize, 
    target_pixels: usize
) -> Result<Vec<u8>, Error> {    
    let (new_width, new_height, pad_x, pad_y, inv_scale) = calculate_letterbox_params(image_height, image_width);
    
    OUTPUT_BUFFER_F32.with(|buffer_cell| {
        MAPPING_CACHE.with(|cache_cell| {
            let mut buffer = buffer_cell.borrow_mut();
            let mut cache = cache_cell.borrow_mut();
            
            // Pre-allocate and clear buffer
            if buffer.capacity() < target_pixels * 3 {
                buffer.reserve(target_pixels * 3);
            }
            buffer.clear();
            buffer.resize(target_pixels * 3, F32_LUT[114]);
            
            // Get or create cached mapping
            let cache_key = (image_height, image_width);
            let mapping = cache.entry(cache_key).or_insert_with(|| {
                generate_fast_mapping(image_height, image_width, new_height, new_width, inv_scale)
            });
            
            let buffer_ptr = buffer.as_mut_ptr();
            
            unsafe {
                let img_ptr = image.as_ptr();
                let lut_ptr = F32_LUT.as_ptr();
                
                // Process with maximum unrolling for FP32
                for y in 0..new_height {
                    let target_y = y + pad_y;
                    let orig_row_base = *mapping.row_offsets.get_unchecked(y);
                    let target_row_base = target_y * target_size + pad_x;
                    
                    // Process 4 pixels at once (optimal for f32 SIMD)
                    let mut x = 0;
                    while x + 3 < new_width {
                        for i in 0..8 {
                            let target_idx = target_row_base + x + i;
                            let orig_idx = (orig_row_base + *mapping.x_indices.get_unchecked(x + i)) * 3;
                            
                            let r = *img_ptr.add(orig_idx) as usize;
                            let g = *img_ptr.add(orig_idx + 1) as usize;
                            let b = *img_ptr.add(orig_idx + 2) as usize;
                            
                            *buffer_ptr.add(target_idx) = *lut_ptr.add(r);
                            *buffer_ptr.add(target_idx + target_pixels) = *lut_ptr.add(g);
                            *buffer_ptr.add(target_idx + target_pixels * 2) = *lut_ptr.add(b);
                        }
                        x += 8;
                    }
                    
                    // Handle remaining pixels
                    while x < new_width {
                        let target_idx = target_row_base + x;
                        let orig_idx = (orig_row_base + *mapping.x_indices.get_unchecked(x)) * 3;
                        
                        let r = *img_ptr.add(orig_idx) as usize;
                        let g = *img_ptr.add(orig_idx + 1) as usize;
                        let b = *img_ptr.add(orig_idx + 2) as usize;
                        
                        *buffer_ptr.add(target_idx) = *lut_ptr.add(r);
                        *buffer_ptr.add(target_idx + target_pixels) = *lut_ptr.add(g);
                        *buffer_ptr.add(target_idx + target_pixels * 2) = *lut_ptr.add(b);
                        
                        x += 1;
                    }
                }
            }
            
            // Safe conversion to bytes
            Ok(unsafe {
                std::slice::from_raw_parts(
                    buffer.as_ptr() as *const u8,
                    buffer.len() * 4
                ).to_vec()
            })
        })
    })
}

fn postprocess_yolo_fp16(
    data: &[u8],
    image_height: usize,
    image_width: usize,
    target_anchors: usize,
    target_classes: usize
) -> Result<Vec<InferenceResult>, Error> {
    // Calculate letterbox parameters (same as preprocessing)
    let (_, _, pad_x, pad_y, inv_scale) = calculate_letterbox_params(image_height, image_width);
    
    // Convert to f32 for calculations
    let pad_x_f32 = pad_x as f32;
    let pad_y_f32 = pad_y as f32;
    
    let mut detections = Vec::with_capacity(100);
    
    // Cast bytes directly to u16 slice
    let u16_data = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u16,
            data.len() / 2
        )
    };
    
    for anchor_idx in 0..target_anchors {
        // Calculate indices for transposed access pattern
        let x_idx = anchor_idx;
        let y_idx = target_anchors + anchor_idx;
        let w_idx = 2 * target_anchors + anchor_idx;
        let h_idx = 3 * target_anchors + anchor_idx;
        
        // Bounds check before unsafe access
        if h_idx >= u16_data.len() {
            continue;
        }
        
        unsafe {
            // Convert f16 to f32 on the fly
            let x = f16::from_bits(*u16_data.get_unchecked(x_idx)).to_f32();
            let y = f16::from_bits(*u16_data.get_unchecked(y_idx)).to_f32();
            let w = f16::from_bits(*u16_data.get_unchecked(w_idx)).to_f32();
            let h = f16::from_bits(*u16_data.get_unchecked(h_idx)).to_f32();
            
            // Convert to corner coordinates
            let x1 = x - w * 0.5;
            let y1 = y - h * 0.5;
            let x2 = x + w * 0.5;
            let y2 = y + h * 0.5;
            
            // Model outputs are already in pixel coordinates (0-640 range)
            // Remove letterbox padding directly
            let x1_unpadded = x1 - pad_x_f32;
            let y1_unpadded = y1 - pad_y_f32;
            let x2_unpadded = x2 - pad_x_f32;
            let y2_unpadded = y2 - pad_y_f32;
            
            // Scale back to original image dimensions
            let final_x1 = x1_unpadded * inv_scale;
            let final_y1 = y1_unpadded * inv_scale;
            let final_x2 = x2_unpadded * inv_scale;
            let final_y2 = y2_unpadded * inv_scale;
            
            let mut max_score: f32 = 0.0;
            let mut max_class: usize = 0;
            
            for class_idx in 0..target_classes {
                let prob_idx = (4 + class_idx) * target_anchors + anchor_idx;
                if prob_idx < u16_data.len() {
                    let score = f16::from_bits(*u16_data.get_unchecked(prob_idx)).to_f32();
                    if score > max_score {
                        max_score = score;
                        max_class = class_idx;
                    }
                }
            }
            
            detections.push(InferenceResult {
                bbox: [final_x1, final_y1, final_x2, final_y2],
                class: max_class,
                score: max_score,
            });
        }
    }
    
    Ok(detections)
}

fn postprocess_yolo_fp32(
    data: &[u8],
    image_height: usize,
    image_width: usize,
    target_anchors: usize,
    target_classes: usize
) -> Result<Vec<InferenceResult>, Error> {
    // Calculate letterbox parameters (same as preprocessing)
    let (_, _, pad_x, pad_y, inv_scale) = calculate_letterbox_params(image_height, image_width);
    
    // Convert to f32 for calculations
    let pad_x_f32 = pad_x as f32;
    let pad_y_f32 = pad_y as f32;
    
    // Pre-allocate with estimated capacity
    let mut detections = Vec::with_capacity(100);
    
    // Cast bytes directly to f32 slice (unsafe but fast)
    let f32_data = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const f32,
            data.len() / 4
        )
    };
    
    // Process each anchor directly without transpose
    for anchor_idx in 0..target_anchors {
        // Calculate indices for transposed access pattern
        let x_idx = anchor_idx; // row 0
        let y_idx = target_anchors + anchor_idx;
        let w_idx = 2 * target_anchors + anchor_idx;
        let h_idx = 3 * target_anchors + anchor_idx;
        
        // Bounds check before unsafe access
        if h_idx >= f32_data.len() {
            continue;
        }
        
        unsafe {
            let x = *f32_data.get_unchecked(x_idx);
            let y = *f32_data.get_unchecked(y_idx);
            let w = *f32_data.get_unchecked(w_idx);
            let h = *f32_data.get_unchecked(h_idx);
            
            // Convert to corner coordinates
            let x1 = x - w * 0.5;
            let y1 = y - h * 0.5;
            let x2 = x + w * 0.5;
            let y2 = y + h * 0.5;
            
            // Model outputs are already in pixel coordinates (0-640 range)
            // Remove letterbox padding directly
            let x1_unpadded = x1 - pad_x_f32;
            let y1_unpadded = y1 - pad_y_f32;
            let x2_unpadded = x2 - pad_x_f32;
            let y2_unpadded = y2 - pad_y_f32;
            
            // Scale back to original image dimensions
            let final_x1 = x1_unpadded * inv_scale;
            let final_y1 = y1_unpadded * inv_scale;
            let final_x2 = x2_unpadded * inv_scale;
            let final_y2 = y2_unpadded * inv_scale;
            
            // Find max probability and class using SIMD-friendly loop
            let mut max_score: f32 = 0.0;
            let mut max_class: usize = 0;
            
            for class_idx in 0..target_classes {
                let prob_idx = (4 + class_idx) * target_anchors + anchor_idx;
                if prob_idx < f32_data.len() {
                    let score = *f32_data.get_unchecked(prob_idx);
                    if score > max_score {
                        max_score = score;
                        max_class = class_idx;
                    }
                }
            }
            
            detections.push(InferenceResult {
                bbox: [final_x1, final_y1, final_x2, final_y2],
                class: max_class,
                score: max_score,
            });
        }
    }
    
    Ok(detections)
}

fn bbox_nms(mut detections: Vec<InferenceResult>, nms_iou_threshold: f32) -> Vec<InferenceResult> {
    if detections.len() <= 1 {
        return detections;
    }
    
    // Sort by score (descending) using unstable sort for speed
    detections.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    
    let mut keep = Vec::with_capacity(detections.len());
    let mut suppressed = vec![false; detections.len()];
    
    for i in 0..detections.len() {
        if unsafe { *suppressed.get_unchecked(i) } {
            continue;
        }
        
        let detection_i = unsafe { detections.get_unchecked(i) };
        keep.push(*detection_i);
        
        // Only check remaining detections of same class
        for j in (i + 1)..detections.len() {
            if unsafe { *suppressed.get_unchecked(j) } {
                continue;
            }
            
            let detection_j = unsafe { detections.get_unchecked(j) };
            
            if detection_i.class != detection_j.class {
                continue;
            }
            
            if get_boxes_iou(&detection_i.bbox, &detection_j.bbox) > nms_iou_threshold {
                unsafe { *suppressed.get_unchecked_mut(j) = true; }
            }
        }
    }
    
    keep
}

fn get_boxes_iou(bbox1: &[f32; 4], bbox2: &[f32; 4]) -> f32 {
    let x1_max = bbox1[0].max(bbox2[0]);
    let y1_max = bbox1[1].max(bbox2[1]);
    let x2_min = bbox1[2].min(bbox2[2]);
    let y2_min = bbox1[3].min(bbox2[3]);
    
    let intersection_width = x2_min - x1_max;
    let intersection_height = y2_min - y1_max;
    
    if intersection_width <= 0.0 || intersection_height <= 0.0 {
        return 0.0;
    }
    
    let intersection_area = intersection_width * intersection_height;
    let bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
    let bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
    let union_area = bbox1_area + bbox2_area - intersection_area;
    
    intersection_area / union_area
}

pub fn preprocess_yolo(
    image: &[u8], 
    image_height: usize, 
    image_width: usize, 
    input_shape: &[i64; 3], 
    precision: InferencePrecision
) -> Result<Vec<u8>, Error> {
    // Pre-Process depending on precision
    if image.len() != image_height * image_width * 3 {
        return Err(Error::new(ErrorKind::Other, "Input image buffer size does not match dimensions. Must be RGB image."));
    }

    // Calculate target sizes
    let target_size = input_shape[1].clone() as usize;
    let target_pixels = target_size * target_size;

    // Get raw input bytes
    let bytes = match precision {
        InferencePrecision::FP16 => preprocess_yolo_fp16(&image, image_height, image_width, target_size, target_pixels),
        InferencePrecision::FP32 => preprocess_yolo_fp32(&image, image_height, image_width, target_size, target_pixels)
    }
    .map_err(|e| Error::new(ErrorKind::Other, e))?;

    Ok(bytes)
}

pub fn postprocess_yolo(    
    results: &[u8], 
    image_height: usize,
    image_width: usize,
    output_shape: &[i64; 2], 
    precision: InferencePrecision,
    pred_conf_threshold: f32,
    nms_iou_threshold: f32
) -> Result<Vec<InferenceResult>, Error> {
    // Calculate target sizes: FP16 = 2 bytes, FP32 = 4 bytes
    let target_features = output_shape[0].clone() as usize;
    let target_anchors = output_shape[1].clone() as usize;
    let target_classes = target_features - 4;
    let target_size = match precision {
        InferencePrecision::FP16 => target_anchors * target_features * 2,
        InferencePrecision::FP32 => target_anchors * target_features * 4
    } as usize;
    
    if results.len() != target_size {
        return Err(Error::new(
            ErrorKind::Other, 
            format!("Got unexpected size of model output ({}). Got {}, expected {}", precision.to_string(), results.len(), target_size)
        ))
    }

    // Get processed detections
    let mut detections = match precision {
        InferencePrecision::FP16 => postprocess_yolo_fp16(results, image_height, image_width, target_anchors, target_classes)?,
        InferencePrecision::FP32 => postprocess_yolo_fp32(results, image_height, image_width, target_anchors, target_classes)?,
    };

    // Filter by confidence before nms
    detections.retain(|r| r.score >= pred_conf_threshold);

    // Perform nms on bboxes
    detections = bbox_nms(detections, nms_iou_threshold);

    Ok(detections)
}
