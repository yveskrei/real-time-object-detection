//! Responsible for handling video stream frames, sending them to inference
//! and populating results to third party systems

use std::sync::Arc;
use tokio::sync::OnceCell;
use std::sync::atomic::{Ordering, AtomicU64};
use std::collections::HashMap;
use anyhow::{Result, Context};
use tokio::time::{Duration, interval, Instant};

// Custom modules
use crate::inference::{self, InferenceFrame, InferenceResult};
use crate::inference::processing;
use crate::utils::config::AppConfig;

/// Static instances of source processors
pub static PROCESSORS: OnceCell<HashMap<String, Arc<SourceProcessor>>> = OnceCell::const_new();

/// Returns a source processor instance by given stream ID
pub async fn get_source_processor(stream_id: &str) -> Result<Arc<SourceProcessor>> {
    Ok(
        PROCESSORS
        .get()
        .context("Cannot get static source processors variable")?
        .get(&stream_id.to_string())
        .cloned()
        .context("Error getting stream source processor")?
    )
}

/// Initiates source processors for given list of sources
pub async fn init_source_processors(app_config: &AppConfig) -> Result<()> {
    let mut processors: HashMap<String, Arc<SourceProcessor>> = HashMap::new();

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

        processors.insert(
            source_id.to_string(),
            processor
        );
    }

    // Set to global variable
    PROCESSORS.set(processors)
        .map_err(|_| anyhow::anyhow!("Error setting source processors"))?;

    Ok(())
}

/// Responsible for giving information about times at specific parts of inference
pub struct SourceProcessStats {
    pub pre_processing: u64,
    pub inference: u64,
    pub post_processing: u64,
    pub results: u64
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
    _process_handle: tokio::task::JoinHandle<()>,
    _stats_handle: tokio::task::JoinHandle<()>,

    // Source specific settings
    source_id: String,
    confidence_threshold: f32,
    inference_frame: usize,

    // Source statistics
    frames_total: Arc<AtomicU64>,
    frames_expected: Arc<AtomicU64>,
    frames_success: Arc<AtomicU64>,
    frames_failed: Arc<AtomicU64>,
    total_pre_proc_time: Arc<AtomicU64>,
    total_inference_time: Arc<AtomicU64>,
    total_post_proc_time: Arc<AtomicU64>,
    total_results_time: Arc<AtomicU64>,
    total_processing_time: Arc<AtomicU64>,
}

impl SourceProcessor {
    /// Creates a new instance of source processor
    /// 
    /// 1. Creates a seperate channel of communication between the main thread and a seperate
    /// thread pool, so we can send frames for inference and not block the execution of other parts
    /// of our code.
    /// 2. Reports statistics about the given source processor in terms performance, including times of 
    /// processing, how many successful/failed frames we have and what is our general success rate 
    pub async fn new(
        source_id: String,
        confidence_threshold: f32,
        inference_frame: usize
    ) -> Self {
        // Create global counters
        let frames_total = Arc::new(AtomicU64::new(0));
        let frames_expected = Arc::new(AtomicU64::new(0));
        let frames_success = Arc::new(AtomicU64::new(0));
        let frames_failed = Arc::new(AtomicU64::new(0));
        let total_pre_proc_time = Arc::new(AtomicU64::new(0));
        let total_inference_time = Arc::new(AtomicU64::new(0));
        let total_post_proc_time = Arc::new(AtomicU64::new(0));
        let total_results_time = Arc::new(AtomicU64::new(0));
        let total_processing_time = Arc::new(AtomicU64::new(0));
        
        // Spawn seperate threadpool channel for inference
        // Set inference queue to 10 frames. it will fail adding when sending more
        let (tx, mut rx) = tokio::sync::mpsc::channel::<InferenceFrame>(10);
        let process_source_id = source_id.clone();
        let process_frames_success = frames_success.clone();
        let process_frames_failed = frames_failed.clone();
        let process_total_pre_proc_time = total_pre_proc_time.clone();
        let process_total_inference_time = total_inference_time.clone();
        let process_total_post_proc_time = total_post_proc_time.clone();
        let process_total_results_time = total_results_time.clone();
        let process_total_processing_time = total_processing_time.clone();

        let process_handle = tokio::spawn(async move {
            while let Some(frame) = rx.recv().await {
                let process_result = Self::process_frame_internal(
                    &process_source_id,
                    &frame,
                    confidence_threshold
                ).await;

                match process_result {
                    Ok(stats) => {
                        process_frames_success.fetch_add(1, Ordering::Relaxed);

                        // Calculate time since frame added to queue
                        let processing = frame.added.elapsed().as_micros() as u64;

                        // Add inference statistics to counters
                        process_total_pre_proc_time.fetch_add(stats.pre_processing, Ordering::Relaxed);
                        process_total_inference_time.fetch_add(stats.inference, Ordering::Relaxed);
                        process_total_post_proc_time.fetch_add(stats.post_processing, Ordering::Relaxed);
                        process_total_results_time.fetch_add(stats.results, Ordering::Relaxed);
                        process_total_processing_time.fetch_add(processing, Ordering::Relaxed);
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

        // Spawn seperate task for source statistics
        let stats_source_id = source_id.clone();
        let stats_inference_frame = inference_frame.clone();
        let stats_frames_total = frames_total.clone();
        let stats_frames_expected = frames_expected.clone();
        let stats_frames_success = frames_success.clone();
        let stats_frames_failed = frames_failed.clone();
        let stats_total_pre_proc_time = total_pre_proc_time.clone();
        let stats_total_inference_time = total_inference_time.clone();
        let stats_total_post_proc_time = total_post_proc_time.clone();
        let stats_total_results_time = total_results_time.clone();
        let stats_total_processing_time = total_processing_time.clone();
        let stats_interval = Duration::from_secs(5);

        let stats_handle = tokio::spawn(async move {
            let mut interval = interval(stats_interval);
            
            loop {
                interval.tick().await;

                Self::process_stats_internal(
                    &stats_source_id, 
                    stats_inference_frame, 
                    stats_frames_total.load(Ordering::Relaxed), 
                    stats_frames_expected.load(Ordering::Relaxed), 
                    stats_frames_success.load(Ordering::Relaxed), 
                    stats_frames_failed.load(Ordering::Relaxed), 
                    stats_total_pre_proc_time.load(Ordering::Relaxed), 
                    stats_total_inference_time.load(Ordering::Relaxed), 
                    stats_total_post_proc_time.load(Ordering::Relaxed), 
                    stats_total_results_time.load(Ordering::Relaxed),
                    stats_total_processing_time.load(Ordering::Relaxed)
                );

                // Reset statistics
                stats_frames_total.store(0, Ordering::Relaxed);
                stats_frames_expected.store(0, Ordering::Relaxed);
                stats_frames_success.store(0, Ordering::Relaxed);
                stats_frames_failed.store(0, Ordering::Relaxed);
                stats_total_pre_proc_time.store(0, Ordering::Relaxed);
                stats_total_inference_time.store(0, Ordering::Relaxed);
                stats_total_post_proc_time.store(0, Ordering::Relaxed);
                stats_total_results_time.store(0, Ordering::Relaxed);
                stats_total_processing_time.store(0, Ordering::Relaxed);

            }
        });

        // Start separate tasks
        tokio::task::yield_now().await;

        tracing::info!(
            source_id=source_id,
            "initiated client processing"
        );
        
        Self {
            frame_sender: tx,
            _process_handle: process_handle,
            _stats_handle: stats_handle,
            source_id,
            confidence_threshold,
            inference_frame,
            frames_total,
            frames_expected,
            frames_success,
            frames_failed,
            total_pre_proc_time,
            total_inference_time,
            total_post_proc_time,
            total_results_time,
            total_processing_time
        }
    }

    /// Sends inference requests to a seperate thread pool
    pub fn process_frame(&self, raw_frame: &[u8], height: usize, width: usize) {
        // Send processing request to seperate thread
        let frames_total = self.frames_total.fetch_add(1, Ordering::Relaxed);

        // Send inference results on every N frame
        if frames_total % (self.inference_frame as u64) == 0 {
            // Count frames we actually expected getting
            self.frames_expected.fetch_add(1, Ordering::Relaxed);

            // Send frame to processing
            let frame = InferenceFrame {
                data: raw_frame.to_vec(),
                height,
                width,
                added: Instant::now()
            };

            if let Err(e) = self.frame_sender.try_send(frame) {
                self.frames_failed.fetch_add(1, Ordering::Relaxed);

                tracing::error!(
                    source_id=&self.source_id,
                    error=e.to_string(),
                    "frame queue is full"
                )
            }
        }
    }

    /// Populates inference results to third party services
    pub fn populate_results(source_id: &str, bboxes: &[InferenceResult]) {
        tracing::info!(
            source_id=source_id,
            bboxes=bboxes.len(),
            "Got bboxes to populate!"
        )
    }


    /// Used to perform inference on a raw frame and return stats about timing
    async fn process_frame_internal(
        source_id: &str,
        frame: &InferenceFrame, 
        confidence_threshold: f32
    ) -> Result<SourceProcessStats> {
        // Perform inference on raw frame and populate results
        let inference_model = inference::get_inference_model()?;
        let inference_start = Instant::now();

        // Pre-process raw frame
        let pre_proc_frame = processing::preprocess_yolo(
            frame,
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

        let results_time = inference_start.elapsed() - pre_proc_time - inference_time - post_proc_time;

        Ok(
            SourceProcessStats {
                pre_processing: pre_proc_time.as_micros() as u64, 
                inference: inference_time.as_micros() as u64, 
                post_processing: post_proc_time.as_micros() as u64, 
                results: results_time.as_micros() as u64
            }
        )
    }

    /// Reports inference statistics for the given source processor
    fn process_stats_internal(
        source_id: &str,
        inference_frame: usize,
        total_frames: u64,
        total_expected: u64,
        total_success: u64,
        total_failed: u64,
        total_pre_proc_time: u64,
        total_inference_time: u64,
        total_post_proc_time: u64,
        total_results_time: u64,
        total_processing_time: u64
    ) {
        let mut avg_pre_proc: f64 = 0.00;
        let mut avg_inference: f64 = 0.00;
        let mut avg_post_proc: f64 = 0.00;
        let mut avg_results: f64 = 0.00;
        let mut avg_processing: f64 = 0.00;

        if total_success > 0 {
            avg_pre_proc = (total_pre_proc_time as f64) / (total_success as f64);
            avg_inference = (total_inference_time as f64) / (total_success as f64);
            avg_post_proc = (total_post_proc_time as f64) / (total_success as f64);
            avg_results = (total_results_time as f64) / (total_success as f64);
            avg_processing = (total_processing_time as f64) / (total_success as f64);
        }

        tracing::info!(
            source_id=source_id,
            inference_every_n=inference_frame,
            total_frames=total_frames,
            total_expected=total_expected,
            total_success=total_success,
            total_failed=total_failed,
            avg_pre_proc=avg_pre_proc,
            avg_inference=avg_inference,
            avg_post_proc=avg_post_proc,
            avg_results=avg_results,
            avg_processing=avg_processing,
            "inference statistics"
        );
    }
}