use std::io::{Error, ErrorKind};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Instant;
use std::sync::atomic::{Ordering, AtomicU64};
use once_cell::sync::Lazy;
use std::collections::HashMap;

// Custom modules
use crate::inference::{InferenceModel, InferenceFrame};
use crate::{processing, stream_client};

// Static source processors
pub static PROCESSORS: Lazy<RwLock<HashMap<String, Arc<SourceProcessor>>>> = 
    Lazy::new(|| RwLock::new(HashMap::new()));
fn get_source_processor(stream_id: &str) -> Arc<SourceProcessor> {
    PROCESSORS
        .read()
        .expect("Source processor is not initiated!")
        .get(&stream_id.to_string())
        .cloned()
        .expect("Error getting source processor")
}

// Static inference model
pub static INFERENCE_MODEL: OnceLock<Arc<InferenceModel>> = OnceLock::new();
fn get_inference_model() -> &'static Arc<InferenceModel> {
    INFERENCE_MODEL
        .get()
        .expect("Infernece model is not initiated!")
}

pub struct SourceProcessor {
    // Settings for multi-threading
    frame_sender: tokio::sync::mpsc::UnboundedSender<InferenceFrame>,
    _handle: tokio::task::JoinHandle<()>,

    // Source specific settings
    source_id: String,
    confidence_threshold: f32,
    inference_frame: usize,

    // Source statistics
    frames_total: AtomicU64,
    frames_success: AtomicU64,
    frames_failed: AtomicU64
}

impl SourceProcessor {
    pub fn new(
        source_id: String,
        confidence_threshold: f32,
        inference_frame: usize
    ) -> Self {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<InferenceFrame>();
        
        // Spawn dedicated thread for this source
        let process_source_id = source_id.clone();
        let handle = tokio::spawn(async move {
            while let Some(frame) = rx.recv().await {
                let process_result = Self::process_frame_internal(
                    &process_source_id,
                    &frame,
                    confidence_threshold
                ).await;

                if let Err(e) = process_result {
                    tracing::error!(
                        source_id=process_source_id,
                        error=e.to_string(),
                        "error processing source frame"
                    )
                }
            }
        });

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

            frames_total: AtomicU64::new(0),
            frames_success: AtomicU64::new(0),
            frames_failed: AtomicU64::new(0),
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

            let _ = self.frame_sender.send(frame);
        }
    }

    async fn process_frame_internal(
        source_id: &str,
        frame: &InferenceFrame, 
        confidence_threshold: f32
    ) -> Result<(), Error> {
        // Perform inference on raw frame and populate results
        let inference_model = get_inference_model();
        let inference_start = Instant::now();

        // Pre-process raw frame
        let pre_proc_frame = processing::preprocess_yolo(
            frame,
            inference_model.input_shape(),
            inference_model.precision()
        )
            .unwrap();
        let pre_proc_time = inference_start.elapsed();


        // Perform inference on frame
        let inference_results = inference_model.infer(&pre_proc_frame).await
            .map_err(|e| Error::new(ErrorKind::Other, e))?;
        let inference_time = inference_start.elapsed() - pre_proc_time;

        // Post-process inference results
        let bboxes = processing::postprocess_yolo(
            &inference_results, 
            &frame,
            inference_model.output_shape(),
            inference_model.precision(),
            confidence_threshold,
            inference_model.nms_iou_threshold()
        )
            .unwrap();
        let post_proc_time = inference_start.elapsed() - inference_time - pre_proc_time;

        // Populate results to third party
        stream_client::populate_results(&source_id, &bboxes);

        tracing::info!(
            source_id=source_id,
            pre_processing=pre_proc_time.as_micros(),
            inference=inference_time.as_micros(),
            post_processing=post_proc_time.as_micros(),
            "successful inference"
        );

        Ok(())
    }
}