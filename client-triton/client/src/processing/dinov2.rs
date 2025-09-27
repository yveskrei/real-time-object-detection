// dinov2.rs - Fixed version with improved SIMD preprocessing

use anyhow::{Result, Context};
use std::time::Instant;

// Custom modules
use crate::inference::{
    source::FrameProcessStats, 
    InferenceModel, 
    InferencePrecision
};
use crate::processing::{self, RawFrame, ResultEmbedding};

// SIMD intrinsics for x86-64 architecture
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// ImageNet normalization constants
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Calculate resize parameters for shortest edge resizing
#[derive(Copy, Clone)]
struct ResizeParams {
    new_width: usize,
    new_height: usize,
    scale_x: f32,
    scale_y: f32,
}

fn calculate_resize_shortest_edge(height: usize, width: usize, target_shortest: usize) -> ResizeParams {
    let min_dim = height.min(width) as f32;
    let scale = target_shortest as f32 / min_dim;
    
    let new_width = (width as f32 * scale).round() as usize;
    let new_height = (height as f32 * scale).round() as usize;
    
    ResizeParams {
        new_width,
        new_height,
        scale_x: new_width as f32 / width as f32,
        scale_y: new_height as f32 / height as f32,
    }
}

/// Performs pre-processing on raw RGB frame for DINOv2 models
/// 
/// This function acts as a dispatcher, checking for CPU support for AVX2 
/// and calling the appropriate implementation.
pub fn preprocess(
    frame: &RawFrame,
    precision: InferencePrecision,
) -> Result<Vec<u8>> {
    // Validate input
    let frame_target_size = frame.height * frame.width * 3;
    if frame.data.len() != frame_target_size {
        anyhow::bail!(
            "Got unexpected size of frame input. Got {}, expected {}",
            frame.data.len(),
            frame_target_size
        );
    }
    
    // Runtime check for AVX2 support
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // Unsafe is required for calling functions with target_feature
        return unsafe { preprocess_avx2(frame, precision) };
    }

    // Fallback to scalar implementation if AVX2 is not supported
    preprocess_scalar(frame, precision)
}

/// Scalar (non-SIMD) implementation of the preprocessing logic.
/// This is the fallback for systems without AVX2 support.
fn preprocess_scalar(
    frame: &RawFrame,
    precision: InferencePrecision,
) -> Result<Vec<u8>> {
    const SHORTEST_EDGE: usize = 256;
    const CROP_SIZE: usize = 224;
    const CROP_PIXELS: usize = CROP_SIZE * CROP_SIZE;

    // Step 1: Calculate resize parameters
    let resize_params = calculate_resize_shortest_edge(frame.height, frame.width, SHORTEST_EDGE);
    
    // Step 2: Calculate center crop parameters
    let crop_x = (resize_params.new_width.saturating_sub(CROP_SIZE)) / 2;
    let crop_y = (resize_params.new_height.saturating_sub(CROP_SIZE)) / 2;
    
    let actual_crop_width = CROP_SIZE.min(resize_params.new_width);
    let actual_crop_height = CROP_SIZE.min(resize_params.new_height);

    match precision {
        InferencePrecision::FP16 => {
            let mut output: Vec<u8> = vec![0; CROP_PIXELS * 3 * 2];
            let out_ptr = output.as_mut_ptr() as *mut u16;
            
            for y in 0..actual_crop_height {
                for x in 0..actual_crop_width {
                    let src_x = (((x + crop_x) as f32 / resize_params.scale_x).round() as usize).min(frame.width - 1);
                    let src_y = (((y + crop_y) as f32 / resize_params.scale_y).round() as usize).min(frame.height - 1);
                    
                    let src_idx = (src_y * frame.width + src_x) * 3;
                    let dst_idx = y * CROP_SIZE + x;
                    
                    let r = frame.data[src_idx] as f32;
                    let g = frame.data[src_idx + 1] as f32;
                    let b = frame.data[src_idx + 2] as f32;
                    
                    let r_norm = (r / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
                    let g_norm = (g / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
                    let b_norm = (b / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
                    
                    unsafe {
                        *out_ptr.add(dst_idx) = processing::get_f32_to_f16_lut(r_norm);
                        *out_ptr.add(dst_idx + CROP_PIXELS) = processing::get_f32_to_f16_lut(g_norm);
                        *out_ptr.add(dst_idx + CROP_PIXELS * 2) = processing::get_f32_to_f16_lut(b_norm);
                    }
                }
            }
            Ok(output)
        }
        InferencePrecision::FP32 => {
            let mut output: Vec<u8> = vec![0; CROP_PIXELS * 3 * 4];
            let out_ptr = output.as_mut_ptr() as *mut f32;

            for y in 0..actual_crop_height {
                for x in 0..actual_crop_width {
                    let src_x = (((x + crop_x) as f32 / resize_params.scale_x).round() as usize).min(frame.width - 1);
                    let src_y = (((y + crop_y) as f32 / resize_params.scale_y).round() as usize).min(frame.height - 1);
                    
                    let src_idx = (src_y * frame.width + src_x) * 3;
                    let dst_idx = y * CROP_SIZE + x;
                    
                    let r = frame.data[src_idx] as f32;
                    let g = frame.data[src_idx + 1] as f32;
                    let b = frame.data[src_idx + 2] as f32;
                    
                    unsafe {
                        *out_ptr.add(dst_idx) = (r / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
                        *out_ptr.add(dst_idx + CROP_PIXELS) = (g / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
                        *out_ptr.add(dst_idx + CROP_PIXELS * 2) = (b / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
                    }
                }
            }
            Ok(output)
        }
    }
}

/// AVX2-accelerated implementation of the preprocessing logic.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn preprocess_avx2(
    frame: &RawFrame,
    precision: InferencePrecision,
) -> Result<Vec<u8>> {
    const SHORTEST_EDGE: usize = 256;
    const CROP_SIZE: usize = 224;
    const CROP_PIXELS: usize = CROP_SIZE * CROP_SIZE;

    // Step 1: Calculate resize parameters
    let resize_params = calculate_resize_shortest_edge(frame.height, frame.width, SHORTEST_EDGE);
    
    // Step 2: Calculate center crop parameters
    let crop_x_start = (resize_params.new_width.saturating_sub(CROP_SIZE)) / 2;
    let crop_y_start = (resize_params.new_height.saturating_sub(CROP_SIZE)) / 2;
    
    let actual_crop_width = CROP_SIZE.min(resize_params.new_width);
    let actual_crop_height = CROP_SIZE.min(resize_params.new_height);

    // Precompute constants for SIMD operations
    let scale_255 = _mm256_set1_ps(1.0 / 255.0);
    let mean_r = _mm256_set1_ps(IMAGENET_MEAN[0]);
    let mean_g = _mm256_set1_ps(IMAGENET_MEAN[1]);
    let mean_b = _mm256_set1_ps(IMAGENET_MEAN[2]);
    let std_r = _mm256_set1_ps(IMAGENET_STD[0]);
    let std_g = _mm256_set1_ps(IMAGENET_STD[1]);
    let std_b = _mm256_set1_ps(IMAGENET_STD[2]);

    match precision {
        InferencePrecision::FP16 => {
            let mut output: Vec<u8> = vec![0; CROP_PIXELS * 3 * 2];
            let r_ptr = output.as_mut_ptr() as *mut u16;
            let g_ptr = unsafe { r_ptr.add(CROP_PIXELS) };
            let b_ptr = unsafe { g_ptr.add(CROP_PIXELS) };
            
            for y in 0..actual_crop_height {
                let src_y = (((y + crop_y_start) as f32 / resize_params.scale_y).round() as usize).min(frame.height - 1);
                let src_y_offset = src_y * frame.width * 3;
                
                let mut x = 0;
                while x + 8 <= actual_crop_width {
                    // Gather 8 RGB pixels (24 bytes total)
                    let mut r_vals = [0u8; 8];
                    let mut g_vals = [0u8; 8];
                    let mut b_vals = [0u8; 8];
                    
                    for i in 0..8 {
                        let src_x = (((x + i + crop_x_start) as f32 / resize_params.scale_x).round() as usize).min(frame.width - 1);
                        let src_idx = src_y_offset + src_x * 3;
                        r_vals[i] = frame.data[src_idx];
                        g_vals[i] = frame.data[src_idx + 1];
                        b_vals[i] = frame.data[src_idx + 2];
                    }

                    // Load and convert u8 to f32
                    let r_i32 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(r_vals.as_ptr() as *const __m128i));
                    let g_i32 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(g_vals.as_ptr() as *const __m128i));
                    let b_i32 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(b_vals.as_ptr() as *const __m128i));
                    
                    let r_f32 = _mm256_cvtepi32_ps(r_i32);
                    let g_f32 = _mm256_cvtepi32_ps(g_i32);
                    let b_f32 = _mm256_cvtepi32_ps(b_i32);

                    // Normalize: (pixel / 255.0 - mean) / std
                    let r_scaled = _mm256_mul_ps(r_f32, scale_255);
                    let g_scaled = _mm256_mul_ps(g_f32, scale_255);
                    let b_scaled = _mm256_mul_ps(b_f32, scale_255);
                    
                    let r_subbed = _mm256_sub_ps(r_scaled, mean_r);
                    let g_subbed = _mm256_sub_ps(g_scaled, mean_g);
                    let b_subbed = _mm256_sub_ps(b_scaled, mean_b);
                    
                    let r_normed = _mm256_div_ps(r_subbed, std_r);
                    let g_normed = _mm256_div_ps(g_subbed, std_g);
                    let b_normed = _mm256_div_ps(b_subbed, std_b);

                    // Convert f32 to f16
                    let r_f16 = unsafe { _mm256_cvtps_ph(r_normed, _MM_FROUND_TO_NEAREST_INT) };
                    let g_f16 = unsafe { _mm256_cvtps_ph(g_normed, _MM_FROUND_TO_NEAREST_INT) };
                    let b_f16 = unsafe { _mm256_cvtps_ph(b_normed, _MM_FROUND_TO_NEAREST_INT) };

                    // Store results in planar format
                    let dst_idx = y * CROP_SIZE + x;
                    unsafe {
                        _mm_storeu_si128(r_ptr.add(dst_idx) as *mut __m128i, r_f16);
                        _mm_storeu_si128(g_ptr.add(dst_idx) as *mut __m128i, g_f16);
                        _mm_storeu_si128(b_ptr.add(dst_idx) as *mut __m128i, b_f16);
                    }

                    x += 8;
                }
                
                // Handle remaining pixels with scalar code
                for remaining_x in x..actual_crop_width {
                    let src_x = (((remaining_x + crop_x_start) as f32 / resize_params.scale_x).round() as usize).min(frame.width - 1);
                    let src_idx = src_y_offset + src_x * 3;
                    let dst_idx = y * CROP_SIZE + remaining_x;
                    
                    let r = frame.data[src_idx] as f32;
                    let g = frame.data[src_idx + 1] as f32;
                    let b = frame.data[src_idx + 2] as f32;
                    
                    let r_norm = (r / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
                    let g_norm = (g / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
                    let b_norm = (b / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
                    
                    unsafe {
                        *r_ptr.add(dst_idx) = processing::get_f32_to_f16_lut(r_norm);
                        *g_ptr.add(dst_idx) = processing::get_f32_to_f16_lut(g_norm);
                        *b_ptr.add(dst_idx) = processing::get_f32_to_f16_lut(b_norm);
                    }
                }
            }
            Ok(output)
        }
        InferencePrecision::FP32 => {
            let mut output: Vec<u8> = vec![0; CROP_PIXELS * 3 * 4];
            let r_ptr = output.as_mut_ptr() as *mut f32;
            let g_ptr = unsafe { r_ptr.add(CROP_PIXELS) };
            let b_ptr = unsafe { g_ptr.add(CROP_PIXELS) };

            for y in 0..actual_crop_height {
                let src_y = (((y + crop_y_start) as f32 / resize_params.scale_y).round() as usize).min(frame.height - 1);
                let src_y_offset = src_y * frame.width * 3;
                
                let mut x = 0;
                while x + 8 <= actual_crop_width {
                    // Gather 8 RGB pixels
                    let mut r_vals = [0u8; 8];
                    let mut g_vals = [0u8; 8];
                    let mut b_vals = [0u8; 8];
                    
                    for i in 0..8 {
                        let src_x = (((x + i + crop_x_start) as f32 / resize_params.scale_x).round() as usize).min(frame.width - 1);
                        let src_idx = src_y_offset + src_x * 3;
                        r_vals[i] = frame.data[src_idx];
                        g_vals[i] = frame.data[src_idx + 1];
                        b_vals[i] = frame.data[src_idx + 2];
                    }

                    // Load and convert u8 to f32
                    let r_i32 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(r_vals.as_ptr() as *const __m128i));
                    let g_i32 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(g_vals.as_ptr() as *const __m128i));
                    let b_i32 = _mm256_cvtepu8_epi32(_mm_loadl_epi64(b_vals.as_ptr() as *const __m128i));
                    
                    let r_f32 = _mm256_cvtepi32_ps(r_i32);
                    let g_f32 = _mm256_cvtepi32_ps(g_i32);
                    let b_f32 = _mm256_cvtepi32_ps(b_i32);

                    // Normalize: (pixel / 255.0 - mean) / std
                    let r_scaled = _mm256_mul_ps(r_f32, scale_255);
                    let g_scaled = _mm256_mul_ps(g_f32, scale_255);
                    let b_scaled = _mm256_mul_ps(b_f32, scale_255);
                    
                    let r_subbed = _mm256_sub_ps(r_scaled, mean_r);
                    let g_subbed = _mm256_sub_ps(g_scaled, mean_g);
                    let b_subbed = _mm256_sub_ps(b_scaled, mean_b);
                    
                    let r_normed = _mm256_div_ps(r_subbed, std_r);
                    let g_normed = _mm256_div_ps(g_subbed, std_g);
                    let b_normed = _mm256_div_ps(b_subbed, std_b);

                    // Store results in planar format
                    let dst_idx = y * CROP_SIZE + x;
                    unsafe {
                        _mm256_storeu_ps(r_ptr.add(dst_idx), r_normed);
                        _mm256_storeu_ps(g_ptr.add(dst_idx), g_normed);
                        _mm256_storeu_ps(b_ptr.add(dst_idx), b_normed);
                    }

                    x += 8;
                }
                
                // Handle remaining pixels with scalar code
                for remaining_x in x..actual_crop_width {
                    let src_x = (((remaining_x + crop_x_start) as f32 / resize_params.scale_x).round() as usize).min(frame.width - 1);
                    let src_idx = src_y_offset + src_x * 3;
                    let dst_idx = y * CROP_SIZE + remaining_x;
                    
                    let r = frame.data[src_idx] as f32;
                    let g = frame.data[src_idx + 1] as f32;
                    let b = frame.data[src_idx + 2] as f32;
                    
                    unsafe {
                        *r_ptr.add(dst_idx) = (r / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
                        *g_ptr.add(dst_idx) = (g / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
                        *b_ptr.add(dst_idx) = (b / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
                    }
                }
            }
            Ok(output)
        }
    }
}

/// Performs post-processing on raw inference results from DINOv2 models
/// 
/// Takes the raw Vec<u8> output from model inference and converts it to 
/// a ResultEmbedding containing the feature vector. This function also dispatches
/// to a SIMD-optimized version if supported.
pub fn postprocess(
    raw_results: &[u8],
    precision: InferencePrecision,
) -> Result<ResultEmbedding> {
    match precision {
        InferencePrecision::FP16 => {
            if raw_results.len() % 2 != 0 {
                anyhow::bail!("FP16 raw results length must be even. Got {}", raw_results.len());
            }
            let num_elements = raw_results.len() / 2;
            
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") {
                return unsafe { postprocess_fp16_avx2(raw_results, num_elements) };
            }

            // Scalar fallback
            let mut embedding_data = Vec::with_capacity(num_elements);
            let raw_ptr = raw_results.as_ptr() as *const u16;
            for i in 0..num_elements {
                let fp16_val = unsafe { *raw_ptr.add(i) };
                embedding_data.push(processing::get_f16_to_f32_lut(fp16_val));
            }
            Ok(ResultEmbedding { data: embedding_data })
        }
        InferencePrecision::FP32 => {
            if raw_results.len() % 4 != 0 {
                anyhow::bail!("FP32 raw results length must be divisible by 4. Got {}", raw_results.len());
            }
            let num_elements = raw_results.len() / 4;
            let mut embedding_data = Vec::with_capacity(num_elements);
            let raw_ptr = raw_results.as_ptr() as *const f32;
            unsafe {
                embedding_data.set_len(num_elements);
                std::ptr::copy_nonoverlapping(raw_ptr, embedding_data.as_mut_ptr(), num_elements);
            }
            Ok(ResultEmbedding { data: embedding_data })
        }
    }
}

/// AVX2-accelerated post-processing for FP16 results.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn postprocess_fp16_avx2(
    raw_results: &[u8],
    num_elements: usize
) -> Result<ResultEmbedding> {
    let mut embedding_data = Vec::with_capacity(num_elements);
    unsafe { embedding_data.set_len(num_elements) };

    let in_ptr = raw_results.as_ptr() as *const u16;
    let out_ptr = embedding_data.as_mut_ptr() as *mut f32;

    let chunks = num_elements / 8;
    for i in 0..chunks {
        let in_offset = i * 8;
        let data_f16 = unsafe { _mm_loadu_si128(in_ptr.add(in_offset) as *const __m128i) };
        let data_f32 = unsafe { _mm256_cvtph_ps(data_f16) };
        unsafe { _mm256_storeu_ps(out_ptr.add(in_offset), data_f32) };
    }
    
    // Handle remaining elements
    let remainder_start = chunks * 8;
    for i in remainder_start..num_elements {
        let fp16_val = unsafe { *in_ptr.add(i) };
        unsafe { *out_ptr.add(i) = processing::get_f16_to_f32_lut(fp16_val) };
    }

    Ok(ResultEmbedding { data: embedding_data })
}

pub async fn process_frame(
    inference_model: &InferenceModel, 
    source_id: &str,
    frame: &RawFrame
) -> Result<FrameProcessStats> {
    let inference_start = Instant::now();

    // Pre process image
    let measure_start = Instant::now();
    let pre_frame = preprocess(&frame, inference_model.model_config().precision)
        .context("Error preprocessing image for DinoV2")?;
    let pre_proc_time = measure_start.elapsed();

    // Get embedding from inference
    let measure_start = Instant::now();
    let raw_results = inference_model.infer(pre_frame)
        .await
        .context("Error performing inference for DinoV2")?;
    let inference_time = measure_start.elapsed();

    // Parse embedding vector
    let measure_start = Instant::now();
    let embedding = postprocess(&raw_results, inference_model.model_config().precision)
        .context("Error postprocessing embedding vector for DinoV2")?;
    let post_proc_time = measure_start.elapsed();

    // Populate results
    let measure_start = Instant::now();
    //SourceProcessor::populate_embedding(source_id, embedding);
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