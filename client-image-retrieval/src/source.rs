//! Responsible for handling video stream frames, sending them to inference
//! and populating results to third party systems

use anyhow::{Context, Result};
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

// Custom modules
use crate::client_video::RawFrame;
use crate::processing::{self, ResultEmbedding};
use crate::services;
use crate::statistics::{FrameProcessStats, SourceStats};
use crate::utils::config::{AppConfig, ModelPurpose, SourceConfig};
use crate::utils::queue::FixedSizeQueue;

// Variables
pub static MAX_QUEUE_FRAMES: usize = 1;

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

        // Create a seperate task for handling frames - performing inference
        let process_source_queue = Arc::clone(&source_queue);
        let process_source_config = Arc::clone(&source_config);
        let process_source_stats = Arc::clone(&source_stats);

        let process_handle = tokio::spawn(async move {
            let frame_process: Result<()> = async {
                loop {
                    // Only pull from queue when we have a permit available
                    if let Some(queued_frame) = process_source_queue.receiver.recv().await {
                        // Move values to the new thread
                        let process_source_config = Arc::clone(&process_source_config);
                        let process_source_stats = Arc::clone(&process_source_stats);

                        // Process frame
                        let process_result =
                            SourceProcessor::process_frame(&process_source_config, queued_frame)
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

        // Process frame in two parallel ways: embedding for whole frame
        // and extraction of bboxes
        let mut stats: FrameProcessStats = {
            let mut processing_stats = FrameProcessStats::default();

            let frame_bboxes_model = services
                .inference_models()
                .model(ModelPurpose::FrameDetection)?;
            let frame_embedding_model = services
                .inference_models()
                .model(ModelPurpose::FrameEmbedding)?;

            // Clones to pass between threads
            let detection_frame = Arc::clone(&frame);
            let embedding_frame = Arc::clone(&frame);

            let ((detection_stats, bboxes), (embedding_stats, frame_embedding)) =
                futures::try_join!(
                    processing::yolo::process_frame(
                        &frame_bboxes_model,
                        source_config,
                        detection_frame
                    ),
                    processing::dino::process_frame(&frame_embedding_model, embedding_frame),
                )
                .context("Error processing frame")?;

            // Gather statistics
            processing_stats.accumulate(&detection_stats);
            processing_stats.accumulate(&embedding_stats);

            // Assign embeddings - 1 frame + m bboxes
            let mut embeddings = Vec::with_capacity(1 + bboxes.len());
            embeddings.push(frame_embedding);

            // Process bboxes if exist, crop and get embeddings for them
            if !bboxes.is_empty() {
                let bbox_embedding_model = services
                    .inference_models()
                    .model(ModelPurpose::BBOXEmbedding)?;
                let embedding_frame = Arc::clone(&frame);

                let (bbox_embedding_stats, mut bbox_embeddings) = processing::dino::process_bboxes(
                    &bbox_embedding_model,
                    embedding_frame,
                    &bboxes,
                )
                .await?;

                // Gather results
                embeddings.append(&mut bbox_embeddings);
                processing_stats.accumulate(&bbox_embedding_stats);
            }

            // Populate embeddings if we have any
            if !embeddings.is_empty() {
                let measure_start = tokio::time::Instant::now();
                SourceProcessor::populate_embeddings(frame.source_id, &embeddings).await;

                // Update results time
                let results_time = measure_start.elapsed();
                processing_stats.results += results_time.as_micros() as u64;
            }

            processing_stats
        };

        // Return statistics
        stats.queue = frame_queue_time.as_micros() as u64;
        stats.processing += frame_queue_time.as_micros() as u64;

        Ok(stats)
    }

    /// Populate embeddings to third party services
    pub async fn populate_embeddings(source_id: u32, embeddings: &[ResultEmbedding]) {
        match services::get_services() {
            Ok(serv) => {
                if let Err(e) = serv
                    .elastic()
                    .populate_embeddings(source_id, Utc::now().timestamp_millis(), embeddings)
                    .await
                {
                    tracing::warn!(
                        source_id = source_id,
                        error = ?e,
                        "Failed to populate bboxes to client video"
                    );
                };
            }
            Err(_) => {
                tracing::warn!(
                    source_id = source_id,
                    "Error populating embeddings, could not get services"
                )
            }
        }
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
