use std::sync::{Arc, RwLock};
use std::time::Instant;
use std::sync::atomic::{Ordering, AtomicU64};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use anyhow::{Result, Context};

// Custom modules
use crate::inference::{self, InferenceFrame, InferenceResult};
use crate::inference::processing;
use crate::utils::config::AppConfig;

/// Holds source processors at global scope
pub static PROCESSORS: Lazy<RwLock<HashMap<String, Arc<SourceProcessor>>>> = 
    Lazy::new(|| RwLock::new(HashMap::new()));
pub fn get_source_processor(stream_id: &str) -> Result<Arc<SourceProcessor>> {
    Ok(
        PROCESSORS
        .read()
        .map_err(|_| anyhow::anyhow!("Source processor is not initiated!"))?
        .get(&stream_id.to_string())
        .cloned()
        .context("Error getting source processor")?
    )
}
pub async fn init_source_processors(app_config: &AppConfig) -> Result<()> {
    for source_id in app_config.source_ids().iter() {
        let confidence_threshold = app_config.source_confs()
            .get(source_id)
            .context("Source does not have confidence threshold setting")?;
        let inference_frame = app_config.source_inf_frames()
            .get(source_id)
            .context("Source does not have inference frame setting")?;
        
        // Start processor
        let processor = Arc::new(
            SourceProcessor::new(
                source_id.clone(), 
                *confidence_threshold, 
                *inference_frame
            ).await
        );

        // Set in global variable
        PROCESSORS
            .write()
            .map_err(|_| anyhow::anyhow!(format!("Cannot set processor for source {}", source_id)))?
            .insert(source_id.clone(), processor);
    }

    Ok(())
}

/// Responsible for managing inference/processing for each source
/// 
/// Performs inference for each source seperately. Allows us to control 
/// each source seperately, with various settings, such as:
/// 1. confidence_threshold: What confidence threshold we apply to results for this specific source.
/// Especially relevant in case this source is known as more problematic and requires higher confidence
/// 2. inference_frame: How many frames we want to skip before performing inference. In other words, 
/// "Inference on every N frame". This allows us to skip inference on frames when source has higher frame
/// rate, having minimal effect on the end user's experience.
pub struct SourceProcessor {
    // Settings for multi-threading
    frame_sender: tokio::sync::mpsc::Sender<InferenceFrame>,
    _handle: tokio::task::JoinHandle<()>,

    // Source specific settings
    source_id: String,
    confidence_threshold: f32,
    inference_frame: usize,

    // Source statistics
    frames_total: Arc<AtomicU64>,
    frames_success: Arc<AtomicU64>,
    frames_failed: Arc<AtomicU64>
}

impl SourceProcessor {
    pub async fn new(
        source_id: String,
        confidence_threshold: f32,
        inference_frame: usize
    ) -> Self {
        // Set inference queue to 10 frames. it will fail adding when sending more
        let (tx, mut rx) = tokio::sync::mpsc::channel::<InferenceFrame>(10);

        // Create global counters
        let frames_total = Arc::new(AtomicU64::new(0));
        let frames_success = Arc::new(AtomicU64::new(0));
        let frames_failed = Arc::new(AtomicU64::new(0));
        
        // Spawn dedicated thread for this source
        let process_source_id = source_id.clone();
        let process_frames_success = frames_success.clone();
        let process_frames_failed = frames_failed.clone();

        let handle = tokio::spawn(async move {
            while let Some(frame) = rx.recv().await {
                let process_result = Self::process_frame_internal(
                    &process_source_id,
                    &frame,
                    confidence_threshold
                ).await;

                match process_result {
                    Ok(_) => {
                        process_frames_success.fetch_add(1, Ordering::Relaxed);
                    },
                    Err(e) => {
                        process_frames_failed.fetch_add(1, Ordering::Relaxed);

                        tracing::error!(
                            source_id=process_source_id,
                            error=e.to_string(),
                            "error processing source frame"
                        )
                    }
                }
            }
        });

        // Small yield to let task start
        tokio::task::yield_now().await;

        tracing::info!(
            source_id=source_id,
            "initiated client processing"
        );
        
        Self {
            frame_sender: tx,
            _handle: handle,

            source_id,
            confidence_threshold,
            inference_frame,

            frames_total,
            frames_success,
            frames_failed,
        }
    }

    pub fn process_frame(&self, raw_frame: &[u8], height: usize, width: usize) {
        // Send processing request to seperate thread
        let frames_total = self.frames_total.fetch_add(1, Ordering::Relaxed);

        // Send inference results on every N frame
        if frames_total % (self.inference_frame as u64) == 0 {
            let frame = InferenceFrame {
                data: raw_frame.to_vec(),
                height,
                width
            };

            match self.frame_sender.try_send(frame) {
                Err(e) => {
                    self.frames_failed.fetch_add(1, Ordering::Relaxed);

                    tracing::error!(
                        source_id=&self.source_id,
                        error=e.to_string(),
                        "frame queue is full"
                    )
                },
                Ok(_) => ()
            }
        }
    }

    async fn process_frame_internal(
        source_id: &str,
        frame: &InferenceFrame, 
        confidence_threshold: f32
    ) -> Result<()> {
        // Perform inference on raw frame and populate results
        let inference_model = inference::get_inference_model()?;
        let inference_start = Instant::now();

        // Pre-process raw frame
        let pre_proc_frame = processing::preprocess_yolo(
            frame,
            inference_model.input_shape(),
            inference_model.precision()
        )?;
        let pre_proc_time = inference_start.elapsed();


        // Perform inference on frame
        let inference_results = inference_model.infer(&pre_proc_frame).await?;
        let inference_time = inference_start.elapsed() - pre_proc_time;

        // Post-process inference results
        let bboxes = processing::postprocess_yolo(
            &inference_results, 
            &frame,
            inference_model.output_shape(),
            inference_model.precision(),
            confidence_threshold,
            inference_model.nms_iou_threshold()
        )?;
        let post_proc_time = inference_start.elapsed() - inference_time - pre_proc_time;

        // Populate inference results
        //SourceProcessor::populate_results(&source_id, &bboxes);

        tracing::info!(
            source_id=source_id,
            pre_processing=pre_proc_time.as_micros(),
            inference=inference_time.as_micros(),
            post_processing=post_proc_time.as_micros(),
            total=inference_start.elapsed().as_micros(),
            "successful inference"
        );

        Ok(())
    }

    pub fn populate_results(source_id: &str, bboxes: &[InferenceResult]) {
        // Populate results to third party stuff, e.g. Kafka
        tracing::info!(
            source_id=source_id,
            bboxes=bboxes.len(),
            "Got bboxes to populate!"
        )
    }
}