/// Module for DINOv3 model pre/post processing

use anyhow::{Result, Context};
use std::sync::Arc;
use std::time::Instant;

// Custom modules
use crate::inference::InferenceModel;
use crate::source::FrameProcessStats;
use crate::processing::{self, RawFrame, ResultEmbedding, ResultBBOX};
use crate::utils::config::InferencePrecision;

/// Performs pre-processing on raw RGB frame for DINOv3 model
/// 
/// This function performs pre-processing steps including resizing, center cropping,
/// and normalization(pixel & ImageNet) to prepare the frame for inference with DINOv3 models.
pub fn preprocess(
    frame: &RawFrame,
    precision: InferencePrecision,
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

    // Preprocess with letterbox resize + ImageNet normalization
    const TARGET_SIZE: u32 = 224;
    processing::resize_letterbox_and_normalize_imagenet(
        &frame.data,
        frame.height,
        frame.width,
        TARGET_SIZE,
        TARGET_SIZE,
        precision
    )
}

/// Performs post-processing on multiple raw inference results from DINOv3 models
/// 
/// Takes a Vec of raw Vec<u8> outputs from batch model inference and converts them to 
/// a Vec of ResultEmbedding containing the feature vectors.
pub fn postprocess(
    raw_results: Vec<Vec<u8>>,
    precision: InferencePrecision,
) -> Result<Vec<ResultEmbedding>> {
    let mut embeddings = Vec::with_capacity(raw_results.len());
    
    for raw_result in raw_results {
        let num_elements = match precision {
            InferencePrecision::FP16 => raw_result.len() / 2,
            InferencePrecision::FP32 => raw_result.len() / 4,
        };
        
        let embedding = match precision {
            InferencePrecision::FP16 => {
                let raw_ptr = raw_result.as_ptr() as *const u16;
                let mut data = Vec::with_capacity(num_elements);
                unsafe {
                    for i in 0..num_elements {
                        data.push(processing::get_f16_to_f32_lut(*raw_ptr.add(i)));
                    }
                }
                ResultEmbedding { data }
            }
            InferencePrecision::FP32 => {
                let raw_ptr = raw_result.as_ptr() as *const f32;
                let data = unsafe {
                    Vec::from_raw_parts(
                        raw_ptr as *mut f32,
                        num_elements,
                        num_elements
                    )
                };
                std::mem::forget(raw_result);
                ResultEmbedding { data }
            }
        };
        
        embeddings.push(embedding);
    }
    
    Ok(embeddings)
}

/// Preprocesses bounding boxes from a frame for DINOv3 inference
/// 
/// Crops each bbox region from the frame, applies letterbox resizing with padding,
/// and performs ImageNet normalization to prepare for DINOv3 model input.
pub fn preprocess_bboxes(
    frame: &RawFrame,
    bboxes: &Vec<ResultBBOX>,
    precision: InferencePrecision,
) -> Result<Vec<Vec<u8>>> {
    const TARGET_SIZE: u32 = 224;
    
    let mut results = Vec::with_capacity(bboxes.len());
    
    for bbox in bboxes {
        // Extract bbox coordinates [x1, y1, x2, y2]
        let x1 = bbox.bbox[0].max(0.0) as u32;
        let y1 = bbox.bbox[1].max(0.0) as u32;
        let x2 = (bbox.bbox[2].min(frame.width as f32)) as u32;
        let y2 = (bbox.bbox[3].min(frame.height as f32)) as u32;
        
        // Calculate bbox dimensions
        let bbox_width = x2.saturating_sub(x1);
        let bbox_height = y2.saturating_sub(y1);
        
        // Skip invalid bboxes
        if bbox_width == 0 || bbox_height == 0 {
            anyhow::bail!("Invalid bbox dimensions: {}x{}", bbox_width, bbox_height);
        }
        
        // Extract the bbox region from the frame
        let expected_size = (bbox_width * bbox_height * 3) as usize;
        let mut cropped_data = Vec::with_capacity(expected_size);
        
        let frame_stride = (frame.width * 3) as usize;
        
        for y in y1..y2 {
            let row_offset = (y as usize) * frame_stride;
            let start_x = (x1 as usize) * 3;
            let end_x = (x2 as usize) * 3;
            
            let row_start = row_offset + start_x;
            let row_end = row_offset + end_x;
            
            cropped_data.extend_from_slice(&frame.data[row_start..row_end]);
        }
        
        // Verify cropped data size
        if cropped_data.len() != expected_size {
            anyhow::bail!(
                "Cropped data size mismatch: got {} bytes, expected {} ({}x{}x3)",
                cropped_data.len(),
                expected_size,
                bbox_width,
                bbox_height
            );
        }
        
        // Apply letterbox resize + padding + ImageNet normalization
        let preprocessed = processing::resize_letterbox_and_normalize_imagenet(
            &cropped_data,
            bbox_height,
            bbox_width,
            TARGET_SIZE,
            TARGET_SIZE,
            precision
        )
            .context("Error preprocessing bbox for DINOv3")?;
        
        results.push(preprocessed);
    }
    
    Ok(results)
}

pub async fn process_frame(
    inference_model: &InferenceModel,
    frame: Arc<RawFrame>,
    bboxes: Arc<Vec<ResultBBOX>>
) -> Result<(FrameProcessStats, Vec<ResultEmbedding>)> {
    let processing_start = Instant::now();

    // Pre process
    let measure_start = Instant::now();
    let precision = inference_model.model_config().precision;
    let frame_clone = Arc::clone(&frame);
    let bboxes_clone = Arc::clone(&bboxes);
    
    let pre_inputs = tokio::task::spawn_blocking(move || {
        let mut pre_inputs = Vec::with_capacity(bboxes_clone.len() + 1);

        let pre_frame = preprocess(&frame_clone, precision)
            .context("Error preprocessing image for DinoV3")?;
        pre_inputs.push(pre_frame);

        let pre_bboxes = preprocess_bboxes(&frame_clone, &bboxes_clone, precision)
            .context("Error preprocessing bboxes for DinoV3")?;
        pre_inputs.extend(pre_bboxes);
        
        Ok::<_, anyhow::Error>(pre_inputs)
    })
        .await
        .context("Preprocess task failed")??;
    let pre_proc_time = measure_start.elapsed();

    // Inference
    let measure_start = Instant::now();
    let raw_results = inference_model.infer(pre_inputs)
        .await
        .context("Error performing inference for DinoV3")?;
    let inference_time = measure_start.elapsed();

    // Post process
    let measure_start = Instant::now();
    let embeddings = tokio::task::spawn_blocking(move || {
        postprocess(raw_results, precision)
    })
        .await
        .context("Postprocess task failed")?
        .context("Error postprocessing embedding vectors for DinoV3")?;
    let post_proc_time = measure_start.elapsed();

    // Statistics
    let mut stats = FrameProcessStats::default();
    stats.pre_processing = pre_proc_time.as_micros() as u64;
    stats.inference = inference_time.as_micros() as u64;
    stats.post_processing = post_proc_time.as_micros() as u64;
    stats.processing = processing_start.elapsed().as_micros() as u64;

    Ok((stats, embeddings))
}