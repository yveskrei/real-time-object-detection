use anyhow::{Result, Context};
use std::time::Instant;
use std::sync::Arc;
use uuid::Uuid;

// Custom modules
use crate::client_video::{Detection, RawFrame, ResultBBOX};
use crate::inference::InferenceModel;
use crate::statistics::FrameProcessStats;
use crate::processing;
use crate::utils::config::SourceConfig;
use crate::utils::config::{InferencePrecision, ModelType};

/// Performs pre-processing on raw RGB frame for YOLO models
/// 
/// Performs the following steps of processing:
/// 1. Resizes the given image to 640x640 while preserving aspect ratio.
/// Applying letterbox padding to complete the missing pixels for certain aspect ratios.
/// 2. Normalizes pixels from 0-255 to 0-1
/// 3. Converting raw pixel values to required precision datatype
/// 4. Outputs raw bytes ordered by color channels(Planar): \[RRRBBBGGG\]
pub fn preprocess_frame(
    frame: &RawFrame,
    precision: InferencePrecision,
    target_size: u32
) -> Result<Vec<u8>> {
    // Validate input
    let frame_target_size = (frame.height * frame.width * 3) as usize;
    if frame.data.len() != frame_target_size {
        anyhow::bail!(
            "Got unexpected size of frame input. Got {}, expected {}",
            frame.data.len(),
            frame_target_size
        );
    }

    // Preprocess with letterbox resize + YOLO normalization
    processing::resize_letterbox_and_normalize(
        &frame.data,
        frame.height,
        frame.width,
        target_size,
        target_size,
        precision
    )
}

/// Perform NMS reduction of bboxes
#[inline(never)] // Don't inline to keep instruction cache hot for main loop
fn bbox_nms(detections: &mut Vec<Detection>, nms_threshold: f32) {
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

/// Bytes one element occupies at the given precision
#[inline]
fn precision_size(precision: InferencePrecision) -> usize {
    match precision {
        InferencePrecision::FP16 => 2,
        InferencePrecision::FP32 => 4,
    }
}

/// Validates the raw output buffer against the shape the configuration declares
///
/// Shared by both decoders. This is the only guard between a mis-declared
/// `output_shape` and a decoder walking off the end of the tensor.
fn validate_output_size(
    results: &[u8],
    dims: [u32; 2],
    precision: InferencePrecision,
) -> Result<()> {
    let expected_size = (dims[0] as usize) * (dims[1] as usize) * precision_size(precision);

    if results.len() != expected_size {
        anyhow::bail!(
            "Got unexpected size of model output data ({}). Got {}, expected {} for shape [{}, {}]",
            precision.to_string(),
            results.len(),
            expected_size,
            dims[0],
            dims[1]
        );
    }

    Ok(())
}

/// Performs post-processing on inference results for raw YOLOv9 heads
///
/// The tensor is `[4 + classes, anchors]`, feature-major - so a single anchor's
/// values are strided across the whole buffer.
///
/// Including the following steps of processing:
/// 1. Convert BBOX coordinates from (x, y, w, h) to (x1, y1, x2, y2) together
/// with restoring the letterbox padding applied during pre-processing
/// 2. Finds out the class id with the max probability - making it the
/// class for the bbox along with its probabiliy
/// 3. Filter BBOXes on a given confidence threshold, before applying NMS(boosts performance significantly)
/// 4. Perform NMS on left over BBOXes
pub fn postprocess_yolov9(
    results: &[u8],
    original_frame: &RawFrame,
    target_size: u32,
    output_dims: [u32; 2],
    precision: InferencePrecision,
    pred_conf_threshold: f32,
    nms_iou_threshold: f32,
) -> Result<Vec<ResultBBOX>> {
    // Both dimensions come from the configured output_shape: features first
    // (4 bbox values + one score per class), anchors second.
    let target_features = output_dims[0];
    let target_anchors = output_dims[1];
    let target_classes = target_features - 4;

    // Validate size of output data
    validate_output_size(results, output_dims, precision)?;

    // Precompute letterbox parameters
    let letterbox = processing::calculate_letterbox(
        original_frame.height, 
        original_frame.width, 
        target_size
    );
    
    // Pre-allocate with exact capacity estimate (typically ~100-200 detections)
    let mut detections: Vec<Detection> = Vec::with_capacity(256);
    
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
                    let x = processing::get_f16_to_f32_lut(*u16_data.get_unchecked(anchor_idx as usize));
                    let y = processing::get_f16_to_f32_lut(*u16_data.get_unchecked((stride1 + anchor_idx) as usize));
                    let w = processing::get_f16_to_f32_lut(*u16_data.get_unchecked((stride2 + anchor_idx) as usize));
                    let h = processing::get_f16_to_f32_lut(*u16_data.get_unchecked((stride3 + anchor_idx) as usize));
                    
                    // Fused bbox transformation
                    let half_w = w * 0.5;
                    let half_h = h * 0.5;
                    let x1 = (x - half_w - letterbox.pad_x as f32) * letterbox.inv_scale;
                    let y1 = (y - half_h - letterbox.pad_y as f32) * letterbox.inv_scale;
                    let x2 = (x + half_w - letterbox.pad_x as f32) * letterbox.inv_scale;
                    let y2 = (y + half_h - letterbox.pad_y as f32) * letterbox.inv_scale;
                    
                    // Find max class with unrolled loop for common cases
                    let mut max_score: f32 = 0.0;
                    let mut max_class: u32 = 0;
                    
                    let class_base = stride4 + anchor_idx;
                    
                    for class_idx in 0..target_classes {
                        let prob_idx = (class_base + class_idx * stride1) as usize;
                        let score = processing::get_f16_to_f32_lut(*u16_data.get_unchecked(prob_idx));
                        if score > max_score {
                            max_score = score;
                            max_class = class_idx;
                        }
                    }
                    
                    // Only store if above threshold
                    if max_score >= pred_conf_threshold {
                        detections.push(
                            Detection {
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
                    let x = *f32_data.get_unchecked(anchor_idx as usize);
                    let y = *f32_data.get_unchecked((stride1 + anchor_idx) as usize);
                    let w = *f32_data.get_unchecked((stride2 + anchor_idx) as usize);
                    let h = *f32_data.get_unchecked((stride3 + anchor_idx) as usize);
                    
                    // Fused bbox transformation
                    let half_w = w * 0.5;
                    let half_h = h * 0.5;
                    let x1 = (x - half_w - letterbox.pad_x as f32) * letterbox.inv_scale;
                    let y1 = (y - half_h - letterbox.pad_y as f32) * letterbox.inv_scale;
                    let x2 = (x + half_w - letterbox.pad_x as f32) * letterbox.inv_scale;
                    let y2 = (y + half_h - letterbox.pad_y as f32) * letterbox.inv_scale;
                    
                    // Find max class with unrolling
                    let mut max_score: f32 = 0.0;
                    let mut max_class: u32 = 0;
                    
                    let class_base = stride4 + anchor_idx;
                    
                    for class_idx in 0..target_classes {
                        let prob_idx = (class_base + class_idx * stride1) as usize;
                        let score = *f32_data.get_unchecked(prob_idx);
                        if score > max_score {
                            max_score = score;
                            max_class = class_idx;
                        }
                    }
                    
                    if max_score >= pred_conf_threshold {
                        detections.push(
                            Detection {
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

    // Give the survivors their backend ids. Done after NMS so we don't mint uuids for
    // the thousands of candidates suppression is about to throw away.
    let bboxes = detections
        .into_iter()
        .map(|detection| detection.into_result(Uuid::new_v4().to_string()))
        .collect();

    Ok(bboxes)
}

/// Number of values per detection row in a YOLO26 end-to-end head
const YOLO26_ROW_FIELDS: u32 = 6;

/// Performs post-processing on inference results for YOLO26 end-to-end heads
///
/// The tensor is `[max_detections, 6]`, **row-major** - one detection is six
/// contiguous values `[x1, y1, x2, y2, score, class_id]`. Compared to the YOLOv9
/// head this is a far smaller and far friendlier read: 1,800 values in one linear
/// sweep instead of 705,600 walked with a stride.
///
/// The head has already done the expensive parts. Boxes arrive as xyxy in
/// letterboxed pixels, and duplicates are already suppressed - so this only:
/// 1. Drops rows under the confidence threshold
/// 2. Undoes the letterbox padding and scale
/// 3. Mints a backend id per surviving detection
///
/// **No NMS is applied**, and `nms_iou_threshold` is deliberately not a parameter.
/// Note the head can emit the same box more than once under different class ids;
/// every such row above the threshold is reported, which is the intended behaviour.
pub fn postprocess_yolo26(
    results: &[u8],
    original_frame: &RawFrame,
    target_size: u32,
    output_dims: [u32; 2],
    precision: InferencePrecision,
    pred_conf_threshold: f32,
) -> Result<Vec<ResultBBOX>> {
    let max_detections = output_dims[0];
    let row_fields = output_dims[1];

    // A [max_det, 7] shape would pass the byte check below and then silently
    // misparse every row, so the row width is checked on its own.
    if row_fields != YOLO26_ROW_FIELDS {
        anyhow::bail!(
            "YOLO26 expects {} values per detection ([x1, y1, x2, y2, score, class_id]), \
             but output_shape declares {}",
            YOLO26_ROW_FIELDS,
            row_fields
        );
    }

    // Validate size of output data
    validate_output_size(results, output_dims, precision)?;

    // Precompute letterbox parameters
    let letterbox = processing::calculate_letterbox(
        original_frame.height,
        original_frame.width,
        target_size
    );

    // Hoist the inverse transform out of the loop
    let pad_x = letterbox.pad_x as f32;
    let pad_y = letterbox.pad_y as f32;
    let inv_scale = letterbox.inv_scale;

    // The head caps detections at max_detections, so one allocation covers the
    // worst case and the vector never grows.
    let mut bboxes: Vec<ResultBBOX> = Vec::with_capacity(max_detections as usize);

    // Reads one row's six values, whatever the precision. Rows are contiguous, so
    // this is a straight linear walk of the buffer.
    macro_rules! decode_rows {
        ($read:expr) => {
            for row_idx in 0..max_detections as usize {
                let base = row_idx * YOLO26_ROW_FIELDS as usize;

                // Score first - most rows are below the threshold and cost nothing more
                let score = $read(base + 4);
                if score < pred_conf_threshold {
                    continue;
                }

                // Already xyxy, so only the letterbox inverse is needed
                let x1 = ($read(base) - pad_x) * inv_scale;
                let y1 = ($read(base + 1) - pad_y) * inv_scale;
                let x2 = ($read(base + 2) - pad_x) * inv_scale;
                let y2 = ($read(base + 3) - pad_y) * inv_scale;

                // class_id arrives as an integral float
                let class = $read(base + 5).round().max(0.0) as u32;

                bboxes.push(
                    Detection {
                        bbox: [x1, y1, x2, y2],
                        class,
                        score,
                    }
                    .into_result(Uuid::new_v4().to_string())
                );
            }
        };
    }

    match precision {
        InferencePrecision::FP16 => {
            let u16_data = unsafe {
                std::slice::from_raw_parts(results.as_ptr() as *const u16, results.len() / 2)
            };

            decode_rows!(|i: usize| processing::get_f16_to_f32_lut(unsafe {
                *u16_data.get_unchecked(i)
            }));
        }
        InferencePrecision::FP32 => {
            let f32_data = unsafe {
                std::slice::from_raw_parts(results.as_ptr() as *const f32, results.len() / 4)
            };

            decode_rows!(|i: usize| unsafe { *f32_data.get_unchecked(i) });
        }
    }

    Ok(bboxes)
}

/// Performs operations on a given frame, including pre/post processing, inference on the given frame
pub async fn process_frame(
    inference_model: &InferenceModel, 
    source_config: &SourceConfig,
    frame: Arc<RawFrame>
) -> Result<(FrameProcessStats, Vec<ResultBBOX>)> {
    let processing_start = Instant::now();
    let target_size = inference_model.model_config()
        .input_shape.last()
        .ok_or(anyhow::anyhow!("Invalid input shape"))?
        .clone() as u32;

    // Pre process
    let measure_start = Instant::now();
    let frame_clone = Arc::clone(&frame);
    let precision = inference_model.model_config().precision();
    let pre_frame = tokio::task::spawn_blocking(move || {
        preprocess_frame(&frame_clone, precision, target_size)
    })
        .await
        .context("Preprocess task failed")?
        .context("Error preprocessing image for YOLO")?;
    let pre_proc_time = measure_start.elapsed();

    // Inference
    let measure_start = Instant::now();
    let raw_results = inference_model.infer(vec![pre_frame])
        .await
        .context("Error performing inference for YOLO")?;
    let inference_time = measure_start.elapsed();

    let raw_results = match raw_results.into_iter().next() {
        Some(res) => res,
        None => anyhow::bail!("No inference results returned for YOLO"),
    };

    // Post process
    let measure_start = Instant::now();
    let post_conf_threshold = source_config.conf_threshold;
    let post_nms_iou_threshold = source_config.nms_iou_threshold;
    let model_type = inference_model.model_config().model_type;

    // Both decoders take the declared output shape, so nothing about the tensor
    // layout is hardcoded. All Copy, so the closure below borrows nothing.
    let output_shape = &inference_model.model_config().output_shape;
    if output_shape.len() != 2 {
        anyhow::bail!(
            "Expected a 2 dimensional output shape, got {} dimensions",
            output_shape.len()
        );
    }
    let output_dims: [u32; 2] = [output_shape[0] as u32, output_shape[1] as u32];

    let bboxes = tokio::task::spawn_blocking(move || match model_type {
        ModelType::YOLOV9 => postprocess_yolov9(
            &raw_results,
            &frame,
            target_size,
            output_dims,
            precision,
            post_conf_threshold,
            post_nms_iou_threshold
        ),
        ModelType::YOLO26 => postprocess_yolo26(
            &raw_results,
            &frame,
            target_size,
            output_dims,
            precision,
            post_conf_threshold
        ),
    })
        .await
        .context("Postprocess task failed")?
        .context("Error postprocessing BBOXes for YOLO")?;
    let post_proc_time = measure_start.elapsed();

    // Statistics
    let mut stats = FrameProcessStats::default();
    stats.pre_processing = pre_proc_time.as_micros() as u64;
    stats.inference = inference_time.as_micros() as u64;
    stats.post_processing = post_proc_time.as_micros() as u64;
    stats.processing = processing_start.elapsed().as_micros() as u64;

    Ok((stats, bboxes))
}