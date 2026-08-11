//! Responsible for handling video stream frames, sending them to inference
//! and populating results to third party systems

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use tokio::sync::Semaphore;

// Custom modules
use crate::client_video::{ClientVideo, RawFrame, ResultBBOX};
use crate::processing;
use crate::services;
use crate::statistics::{FrameProcessStats, SourceStats};
use crate::utils::config::{AppConfig, ModelPurpose, SourceConfig};
use crate::utils::queue::FixedSizeQueue;

// Variables
pub static MAX_QUEUE_FRAMES: usize = 15;

/// A frame waiting in a source's queue
///
/// `added` lives here rather than on `RawFrame` because arrival time is a property of
/// our queue, not of the frame the video library handed us.
pub struct QueuedFrame {
    pub frame: Arc<RawFrame>,
    pub added: tokio::time::Instant,
}

pub struct SourceProcessors {
    processors: HashMap<u32, Arc<SourceProcessor>>,
}

impl SourceProcessors {
    pub fn new(app_config: &AppConfig) -> Result<Self> {
        let mut processors = HashMap::new();

        for (source_id, source_config) in app_config.sources_config().sources.iter() {
            // Start processor
            let processor = Arc::new(SourceProcessor::new(*source_id, source_config.clone()));

            processors.insert(*source_id, processor);
        }

        Ok(Self { processors })
    }

    pub fn processor(&self, stream_id: u32) -> Result<Arc<SourceProcessor>> {
        self.processors
            .get(&stream_id)
            .cloned()
            .context("Error getting stream source processor")
    }

    pub fn processors(&self) -> &HashMap<u32, Arc<SourceProcessor>> {
        &self.processors
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
#[allow(dead_code)]
pub struct SourceProcessor {
    // Settings for multi-threading
    queue: Arc<FixedSizeQueue<QueuedFrame>>,
    queue_semaphore: Arc<Semaphore>,
    process_handle: tokio::task::JoinHandle<()>,

    // Source specific settings
    source_id: u32,
    source_config: Arc<SourceConfig>,
    source_stats: Arc<SourceStats>,
}

impl SourceProcessor {
    /// Creates a new instance of source processor
    ///
    /// 1. Creates a seperate channel of communication between the main thread and a seperate
    /// thread pool, so we can send frames for inference and not block the execution of other parts
    /// of our code.
    /// 2. Reports statistics about the given source processor in terms performance, including times of
    /// processing, how many successful/failed frames we have and what is our general success rate
    pub fn new(source_id: u32, source_config: SourceConfig) -> Self {
        // Create global counters
        let source_stats = Arc::new(SourceStats::new());
        let source_config = Arc::new(source_config);

        // Create a queue for frames. We set a maximum number of frames possible to be in queue at a given time
        // When the limit reaches, it drops the oldest frame in the queue, making it possible for new frames
        // to be added to the queue and be processed.
        let queue_stats = Arc::clone(&source_stats);
        let queue_drop_callback = move |_: QueuedFrame| {
            queue_stats.frames_failed.fetch_add(1, Ordering::Relaxed);
        };
        let source_queue = Arc::new(FixedSizeQueue::<QueuedFrame>::new(
            MAX_QUEUE_FRAMES,
            Some(queue_drop_callback),
        ));
        let queue_semaphore = Arc::new(Semaphore::new(MAX_QUEUE_FRAMES));

        // Create a seperate task for handling frames - performing inference
        let process_queue_semaphore = Arc::clone(&queue_semaphore);
        let process_source_queue = Arc::clone(&source_queue);
        let process_source_config = Arc::clone(&source_config);
        let process_source_stats = Arc::clone(&source_stats);

        let process_handle = tokio::spawn(async move {
            let frame_process: Result<()> = async {
                loop {
                    // Try to acquire permit without blocking
                    match Arc::clone(&process_queue_semaphore).acquire_owned().await {
                        Ok(permit) => {
                            // Only pull from queue when we have a permit available
                            if let Some(queued_frame) = process_source_queue.receiver.recv().await {
                                // Move values to the new thread
                                let process_source_config = Arc::clone(&process_source_config);
                                let process_source_stats = Arc::clone(&process_source_stats);

                                // Spawn processing in a new thread with permit
                                tokio::spawn(async move {
                                    // Keep permit alive until processing completes
                                    let _permit = permit;

                                    let process_result = SourceProcessor::process_frame(
                                        &process_source_config,
                                        queued_frame,
                                    )
                                    .await;

                                    // Count processing statistics
                                    process_source_stats
                                        .frames_total
                                        .fetch_add(1, Ordering::Relaxed);
                                    process_source_stats
                                        .frames_expected
                                        .fetch_add(1, Ordering::Relaxed);
                                    match &process_result {
                                        Ok(stats) => {
                                            process_source_stats
                                                .frames_success
                                                .fetch_add(1, Ordering::Relaxed);

                                            // Add inference statistics to counters
                                            process_source_stats.accumulate(stats);
                                        }
                                        Err(_) => {
                                            process_source_stats
                                                .frames_failed
                                                .fetch_add(1, Ordering::Relaxed);
                                        }
                                    }

                                    // Handle processing error
                                    if let Err(e) = process_result {
                                        tracing::error!(
                                            source_id = source_id,
                                            error = ?e,
                                            "error processing source frame"
                                        )
                                    };
                                });
                            }
                        }
                        Err(e) => {
                            tracing::info!(
                                source_id = source_id,
                                error = e.to_string(),
                                "Error acquiring permit for parallelism. Should not happen"
                            )
                        }
                    }
                }
            }
            .await;

            if let Err(e) = frame_process {
                tracing::error!(
                    source_id = source_id,
                    error = e.to_string(),
                    "Stopped processing frames - due to fatal error"
                )
            }
        });

        tracing::info!(source_id = source_id, "initiated client processing");

        Self {
            queue: source_queue,
            queue_semaphore,
            process_handle,
            source_id,
            source_config,
            source_stats,
        }
    }

    /// Sends inference requests to a seperate thread pool
    pub async fn add_to_queue(&self, raw_frame: RawFrame) {
        let frames_total = self.source_stats.frames_total.load(Ordering::Relaxed);

        // Send inference results on every N frame
        if (frames_total + 1).is_multiple_of(self.source_config.inf_frame as u64) {
            // Stamp arrival time as the frame enters the queue
            let queued_frame = QueuedFrame {
                frame: Arc::new(raw_frame),
                added: tokio::time::Instant::now(),
            };

            // Send new frame to queue
            self.queue.sender.send_async(queued_frame).await;
        } else {
            // Add to statistics
            self.source_stats
                .frames_total
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Used to perform inference on a raw frame and return stats about timing
    async fn process_frame(
        source_config: &SourceConfig,
        queued_frame: QueuedFrame,
    ) -> Result<FrameProcessStats> {
        let QueuedFrame { frame, added } = queued_frame;
        let frame_queue_time = added.elapsed();
        let services = services::get_services()?;

        // Get BBOXes for frame
        let frame_model = services
            .inference_models()
            .model(ModelPurpose::FrameDetection)?;

        let yolo_frame = Arc::clone(&frame);
        let (mut stats, bboxes) =
            processing::yolo::process_frame(&frame_model, source_config, yolo_frame).await?;

        // Populate BBOXes if we have any
        if !bboxes.is_empty() {
            let measure_start = tokio::time::Instant::now();

            // Populate BBOXes to third party services
            let results_frame = Arc::clone(&frame);
            let results_arc = Arc::new(bboxes);
            SourceProcessor::populate_bboxes(results_frame, results_arc).await;

            // Update results time
            let results_time = measure_start.elapsed();
            stats.results += results_time.as_micros() as u64;
        }

        // Return statistics
        stats.queue = frame_queue_time.as_micros() as u64;
        stats.processing += frame_queue_time.as_micros() as u64;
        Ok(stats)
    }

    /// Populates BBOXes to third party services
    pub async fn populate_bboxes(frame: Arc<RawFrame>, bboxes: Arc<Vec<ResultBBOX>>) {
        // Send to client video
        let client_frame = Arc::clone(&frame);
        let client_bboxes = Arc::clone(&bboxes);

        if let Err(e) = ClientVideo::populate_bboxes(&client_frame, &client_bboxes).await {
            tracing::warn!(
                source_id = client_frame.source_id,
                error = ?e,
                "Failed to populate bboxes to client video"
            );
        };
    }
}

impl SourceProcessor {
    pub fn source_stats(&self) -> &SourceStats {
        &self.source_stats
    }
}

impl Drop for SourceProcessor {
    fn drop(&mut self) {
        // Abort tokio tasks
        self.process_handle.abort();
    }
}
