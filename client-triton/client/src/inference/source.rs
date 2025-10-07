//! Responsible for handling video stream frames, sending them to inference
//! and populating results to third party systems

use std::sync::Arc;
use std::sync::atomic::{Ordering, AtomicU64};
use std::collections::HashMap;
use anyhow::{Result, Context};
use tokio::time::{Duration, interval, Instant};
use tokio::sync::{RwLock, Semaphore};
use std::sync::LazyLock;
use serde_json::json;

// Custom modules
use crate::inference::{
    self, 
    queue::FixedSizeQueue, 
    InferenceModelType
};
use crate::processing::{self, RawFrame, ResultBBOX, ResultEmbedding};
use crate::utils::config::{AppConfig, SourceConfig};
use crate::utils::kafka;

// Variables
pub static PROCESSORS: LazyLock<RwLock<HashMap<String, Arc<SourceProcessor>>>> = LazyLock::new(|| RwLock::new(HashMap::new()));
pub static MAX_QUEUE_FRAMES: usize = 5;
pub static MAX_PARALLEL_FRAME_PROCESSING: usize = 5;
pub static SOURCE_STATS_INTERVAL: Duration = Duration::from_secs(1);

/// Returns a source processor instance by given stream ID
pub async fn get_source_processor(stream_id: &str) -> Result<Arc<SourceProcessor>> {
    Ok(
        PROCESSORS
        .read()
        .await
        .get(&stream_id.to_string())
        .cloned()
        .context("Error getting stream source processor")?
    )
}

/// Initiates source processors for given list of sources
pub async fn init_source_processors(app_config: &AppConfig) -> Result<()> {
    let mut processors: HashMap<String, Arc<SourceProcessor>> = HashMap::new();

    for source in app_config.sources_config().sources.keys() {
        let source_config = app_config.sources_config().sources
            .get(source)
            .context("Source config is not set")?;
        
        // Start processor
        let processor = Arc::new(
            SourceProcessor::new(
                source.to_string(), 
                source_config.clone()
            ).await
        );

        processors.insert(
            source.to_string(),
            processor
        );
    }

    // Set to global variable
    *PROCESSORS.write().await = processors;

    Ok(())
}

/// Responsible for giving information about times at specific parts of inference
pub struct FrameProcessStats {
    pub queue: u64,
    pub pre_processing: u64,
    pub inference: u64,
    pub post_processing: u64,
    pub results: u64,
    pub processing: u64
}

pub struct SourceStats {
    pub frames_total: AtomicU64,
    pub frames_expected: AtomicU64,
    pub frames_success: AtomicU64,
    pub frames_failed: AtomicU64,
    pub total_queue_time: AtomicU64,
    pub total_pre_proc_time: AtomicU64,
    pub total_inference_time: AtomicU64,
    pub total_post_proc_time: AtomicU64,
    pub total_results_time: AtomicU64,
    pub total_processing_time: AtomicU64
}

impl SourceStats {
    pub fn new() -> Self {
        Self {
            frames_total: AtomicU64::new(0),
            frames_expected: AtomicU64::new(0),
            frames_success: AtomicU64::new(0),
            frames_failed: AtomicU64::new(0),
            total_queue_time: AtomicU64::new(0),
            total_pre_proc_time: AtomicU64::new(0),
            total_inference_time: AtomicU64::new(0),
            total_post_proc_time: AtomicU64::new(0),
            total_results_time: AtomicU64::new(0),
            total_processing_time: AtomicU64::new(0)
        }
    }

    pub fn reset(&self) {
        self.frames_total.store(0, Ordering::Relaxed);
        self.frames_expected.store(0, Ordering::Relaxed);
        self.frames_success.store(0, Ordering::Relaxed);
        self.frames_failed.store(0, Ordering::Relaxed);
        self.total_queue_time.store(0, Ordering::Relaxed);
        self.total_pre_proc_time.store(0, Ordering::Relaxed);
        self.total_inference_time.store(0, Ordering::Relaxed);
        self.total_post_proc_time.store(0, Ordering::Relaxed);
        self.total_results_time.store(0, Ordering::Relaxed);
        self.total_processing_time.store(0, Ordering::Relaxed);
    }

    pub fn add_stats(&self, stats: &FrameProcessStats) {
        self.total_queue_time.fetch_add(stats.queue, Ordering::Relaxed);
        self.total_pre_proc_time.fetch_add(stats.pre_processing, Ordering::Relaxed);
        self.total_inference_time.fetch_add(stats.inference, Ordering::Relaxed);
        self.total_post_proc_time.fetch_add(stats.post_processing, Ordering::Relaxed);
        self.total_results_time.fetch_add(stats.results, Ordering::Relaxed);
        self.total_processing_time.fetch_add(stats.processing, Ordering::Relaxed);
    }
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
    queue: Arc<FixedSizeQueue<RawFrame>>,
    process_handle: tokio::task::JoinHandle<()>,
    stats_handle: tokio::task::JoinHandle<()>,

    // Source specific settings
    source_id: String,
    source_config: SourceConfig,

    // Source statistics
    source_stats: Arc<SourceStats>
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
        source_config: SourceConfig
    ) -> Self {
        // Create global counters
        let source_stats = Arc::new(SourceStats::new());
        
        // Create a queue for frames. We set a maximum number of frames possible to be in queue at a given time
        // When the limit reaches, it drops the oldest frame in the queue, making it possible for new frames
        // to be added to the queue and be processed.
        let queue_stats = Arc::clone(&source_stats);
        let queue_drop_callback = move |_: &RawFrame| {
            queue_stats.frames_failed.fetch_add(1, Ordering::Relaxed);
        };
        let source_queue = Arc::new(FixedSizeQueue::<RawFrame>::new(MAX_QUEUE_FRAMES, Some(queue_drop_callback)));
        let queue_parallel_limit = Arc::new(Semaphore::new(MAX_PARALLEL_FRAME_PROCESSING));
        
        // Create a seperate task for handling frames - performing inference
        let process_queue_parallel_limit = Arc::clone(&queue_parallel_limit);
        let process_source_queue = Arc::clone(&source_queue);
        let process_source_id = source_id.clone();
        let process_source_config = source_config.clone();
        let process_source_stats = Arc::clone(&source_stats);

        let process_handle = tokio::spawn(async move {
            let frame_process: Result<()> = async {
                loop {
                    // Try to acquire permit without blocking
                    match Arc::clone(&process_queue_parallel_limit).try_acquire_owned() {
                        Ok(permit) => {
                            // Only pull from queue when we have a permit available
                            if let Some(frame) = process_source_queue.receiver.recv().await {
                                // Move values to the new thread
                                let process_source_id = process_source_id.clone();
                                let process_source_config = process_source_config.clone();
                                let process_source_stats = Arc::clone(&process_source_stats);

                                // Spawn processing in a new thread with permit
                                tokio::spawn(async move {
                                    // Keep permit alive until processing completes
                                    let _permit = permit;
                                    
                                    let process_result = Self::process_frame_internal(
                                        &process_source_id,
                                        &process_source_config,
                                        &frame
                                    ).await;

                                    // Count processing statistics
                                    process_source_stats.frames_total.fetch_add(1, Ordering::Relaxed);
                                    process_source_stats.frames_expected.fetch_add(1, Ordering::Relaxed);
                                    match &process_result {
                                        Ok(stats) => {
                                            process_source_stats.frames_success.fetch_add(1, Ordering::Relaxed);

                                            // Add inference statistics to counters
                                            process_source_stats.add_stats(&stats);
                                        },
                                        Err(_) => {
                                            process_source_stats.frames_failed.fetch_add(1, Ordering::Relaxed);
                                        }
                                    }
                                    
                                    // Handle processing error
                                    if let Err(e) = process_result {
                                        tracing::error!(
                                            source_id=process_source_id,
                                            error=e.to_string(),
                                            "error processing source frame"
                                        )
                                    };
                                });
                            }
                        }
                        Err(_) => {
                            // No permits available - yield control to other tasks
                            // This prevents busy-waiting while allowing quick retry
                            tokio::task::yield_now().await;
                        }
                    }
                }
            }.await;

            if let Err(e) = frame_process {
                tracing::error!(
                    source_id=process_source_id,
                    error=e.to_string(),
                    "Stopped processing frames - due to fatal error"
                )
            }
        });

        // Create a seperate task for printing source statistics
        let stats_source_id = source_id.clone();
        let stats_source_config = source_config.clone();
        let stats_source_stats = Arc::clone(&source_stats);
        let stats_interval = SOURCE_STATS_INTERVAL.clone();

        let stats_handle = tokio::spawn(async move {
            let mut interval = interval(stats_interval);
            
            loop {
                interval.tick().await;

                Self::process_stats_internal(
                    &stats_source_id, 
                    &stats_source_config,
                    &stats_source_stats
                );

                // Reset statistics
                stats_source_stats.reset();

            }
        });

        // Start separate tasks
        tokio::task::yield_now().await;

        tracing::info!(
            source_id=source_id,
            "initiated client processing"
        );
        
        Self {
            queue: source_queue,
            process_handle,
            stats_handle,
            source_id,
            source_config,
            source_stats
        }
    }

    /// Sends inference requests to a seperate thread pool
    pub fn process_frame(&self, raw_frame: Vec<u8>, height: usize, width: usize) {
        let frames_total = self.source_stats.frames_total.load(Ordering::Relaxed);

        // Send inference results on every N frame
        if (frames_total + 1) % (self.source_config.inf_frame as u64) == 0 {
            // Create new frame object
            let frame = RawFrame {
                data: raw_frame,
                height,
                width,
                added: Instant::now()
            };

            // Send new frame to queue
            if let Err(e) = self.queue.sender.send_sync(frame) {
                self.source_stats.frames_failed.fetch_add(1, Ordering::Relaxed);

                tracing::error!(
                    source_id=&self.source_id,
                    error=e.to_string(),
                    "source frame queue is full"
                )
            }
        } else {
            // Add to statistics
            self.source_stats.frames_total.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Used to perform inference on a raw frame and return stats about timing
    #[allow(unreachable_patterns)]
    async fn process_frame_internal(
        source_id: &str,
        source_config: &SourceConfig,
        frame: &RawFrame, 
    ) -> Result<FrameProcessStats> {
        // Perform inference on raw frame and populate results
        let inference_model = inference::get_inference_model()?;

        let stats = match inference_model.model_config().model_type {
            InferenceModelType::YOLO => {
                processing::yolo::process_frame(
                    &inference_model,
                    &source_id,
                    &source_config,
                    &frame
                ).await?
            },
            InferenceModelType::DINO => {
                processing::dinov2::process_frame(
                    &inference_model,
                    &source_id,
                    &frame
                ).await?
            },
            _ => anyhow::bail!("Model type is not supported for processing!")
        };

        Ok(stats)
    }

    /// Reports inference statistics for the given source processor
    fn process_stats_internal(
        source_id: &str,
        source_config: &SourceConfig,
        source_stats: &SourceStats
    ) {
        let mut avg_queue: f64 = 0.00;
        let mut avg_pre_proc: f64 = 0.00;
        let mut avg_inference: f64 = 0.00;
        let mut avg_post_proc: f64 = 0.00;
        let mut avg_results: f64 = 0.00;
        let mut avg_processing: f64 = 0.00;

        // Extract values of statistics
        let frames_total = source_stats.frames_total.load(Ordering::Relaxed) as u64;
        let frames_expected = source_stats.frames_expected.load(Ordering::Relaxed) as u64;
        let frames_success = source_stats.frames_success.load(Ordering::Relaxed) as u64;
        let frames_failed = source_stats.frames_failed.load(Ordering::Relaxed) as u64;
        let total_queue_time = source_stats.total_queue_time.load(Ordering::Relaxed) as u64;
        let total_pre_proc_time = source_stats.total_pre_proc_time.load(Ordering::Relaxed) as u64;
        let total_inference_time = source_stats.total_inference_time.load(Ordering::Relaxed) as u64;
        let total_post_proc_time = source_stats.total_post_proc_time.load(Ordering::Relaxed) as u64;
        let total_results_time = source_stats.total_results_time.load(Ordering::Relaxed) as u64;
        let total_processing_time = source_stats.total_processing_time.load(Ordering::Relaxed) as u64;
        
        if frames_success > 0 {
            avg_queue = (total_queue_time as f64) / (frames_success as f64);
            avg_pre_proc = (total_pre_proc_time as f64) / (frames_success as f64);
            avg_inference = (total_inference_time as f64) / (frames_success as f64);
            avg_post_proc = (total_post_proc_time as f64) / (frames_success as f64);
            avg_results = (total_results_time as f64) / (frames_success as f64);
            avg_processing = (total_processing_time as f64) / (frames_success as f64);
        }

        tracing::info!(
            source_id=source_id,
            inference_every_n=source_config.inf_frame,
            frames_total=frames_total,
            frames_expected=frames_expected,
            frames_success=frames_success,
            frames_failed=frames_failed,
            avg_queue=avg_queue,
            avg_pre_proc=avg_pre_proc,
            avg_inference=avg_inference,
            avg_post_proc=avg_post_proc,
            avg_results=avg_results,
            avg_processing=avg_processing,
            "inference statistics"
        );
    }

    /// Populates BBOXes to third party services
    pub async fn populate_bboxes(source_id: &str, bboxes: Vec<ResultBBOX>) {
        // Populate to Kafka
        if let Ok(kafka_producer) = kafka::get_kafka_producer() {
            if let Ok(data) = serde_json::to_string(&bboxes) {
                let message = json!({
                    "type": "BBOX",
                    "data": data
                }).to_string();
                
                if let Err(e) = kafka_producer.produce(Some(source_id), &message).await {
                    tracing::warn!(
                        source_id = source_id,
                        error = %e,
                        "Failed to populate bboxes to Kafka"
                    );
                }
            }
        };
    }

    /// Populates embedding to third party services
    pub async fn populate_embedding(source_id: &str, embedding: ResultEmbedding) {
        // Populate to Kafka
        if let Ok(kafka_producer) = kafka::get_kafka_producer() {
            if let Ok(data) = serde_json::to_string(&embedding) {
                let message = json!({
                    "type": "Embedding",
                    "data": data
                }).to_string();
                
                if let Err(e) = kafka_producer.produce(Some(source_id), &message).await {
                    tracing::warn!(
                        source_id = source_id,
                        error = %e,
                        "Failed to populate embedding to Kafka"
                    );
                }
            }
        };
    }
}

impl Drop for SourceProcessor {
    fn drop(&mut self) {
        // Abort tokio tasks
        self.process_handle.abort();
        self.stats_handle.abort();
    }
}