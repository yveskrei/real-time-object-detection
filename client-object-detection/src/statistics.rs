use anyhow::{Context, Result};
use nvml_wrapper::Nvml;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// Custom modules
use crate::services;
use crate::utils::config::DeviceType;

// Variables
pub static SOURCE_STATS_INTERVAL: tokio::time::Duration = tokio::time::Duration::from_secs(1);
pub static GPU_STATS_INTERVAL: std::time::Duration = std::time::Duration::from_secs(1);

/// Represents GPU statistics that are reported by the application
pub struct GPUStats {
    pub hardware_name: String,
    pub uuid: String,
    pub serial: String,
    pub memory_total: u64,
    pub memory_used: u64,
    pub memory_free: u64,
    pub util_perc: u32,
    pub memory_perc: u32,
}

/// Responsible for giving information about times at specific parts of inference
pub struct FrameProcessStats {
    pub queue: u64,
    pub pre_processing: u64,
    pub inference: u64,
    pub post_processing: u64,
    pub results: u64,
    pub processing: u64,
}

impl Default for FrameProcessStats {
    fn default() -> Self {
        Self {
            queue: 0,
            pre_processing: 0,
            inference: 0,
            post_processing: 0,
            results: 0,
            processing: 0,
        }
    }
}

impl FrameProcessStats {
    pub fn accumulate(&mut self, other: &Self) {
        self.queue += other.queue;
        self.pre_processing += other.pre_processing;
        self.inference += other.inference;
        self.post_processing += other.post_processing;
        self.results += other.results;
        self.processing += other.processing;
    }
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
    pub total_processing_time: AtomicU64,
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
            total_processing_time: AtomicU64::new(0),
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

    pub fn accumulate(&self, stats: &FrameProcessStats) {
        self.total_queue_time
            .fetch_add(stats.queue, Ordering::Relaxed);
        self.total_pre_proc_time
            .fetch_add(stats.pre_processing, Ordering::Relaxed);
        self.total_inference_time
            .fetch_add(stats.inference, Ordering::Relaxed);
        self.total_post_proc_time
            .fetch_add(stats.post_processing, Ordering::Relaxed);
        self.total_results_time
            .fetch_add(stats.results, Ordering::Relaxed);
        self.total_processing_time
            .fetch_add(stats.processing, Ordering::Relaxed);
    }
}

#[allow(dead_code)]
pub struct Statistics {
    is_running: Arc<AtomicBool>,
    nvml: Option<Nvml>,
    source_stats_handle: Option<tokio::task::JoinHandle<()>>,
    gpu_stats_handle: Option<tokio::task::JoinHandle<()>>,
}

impl Statistics {
    /// Creates the statistics reporters
    ///
    /// NVML is initiated here rather than inside the reporting thread - it dlopens and
    /// initialises the driver library, which is expensive and should happen once at startup.
    /// On a GPU deployment, failing to initiate it is fatal: we would silently lose all GPU
    /// telemetry for the lifetime of the process.
    pub fn new(device_type: DeviceType) -> Result<Self> {
        let nvml = match device_type {
            DeviceType::CPU => {
                tracing::warn!("Device type is CPU, GPU statistics will not be reported");

                None
            }
            DeviceType::GPU => match Nvml::init() {
                Ok(nvml) => Some(nvml),
                Err(e) => anyhow::bail!("Error initiating NVML wrapper for GPU statistics: {}", e),
            },
        };

        Ok(Self {
            is_running: Arc::new(AtomicBool::new(true)),
            nvml,
            source_stats_handle: None,
            gpu_stats_handle: None,
        })
    }

    pub fn start(&mut self) -> Result<()> {
        // Spawn an independent task to print source statistics
        let source_stats_is_running = Arc::clone(&self.is_running);
        let source_stats_interval = SOURCE_STATS_INTERVAL;
        let source_stats_handle = tokio::task::spawn(async move {
            let mut interval = tokio::time::interval(source_stats_interval);
            if let Ok(services) = services::get_services() {
                while source_stats_is_running.load(Ordering::Relaxed) {
                    interval.tick().await;

                    // Iterate over each source
                    let processors = services.source_processors().read().await;
                    for (source_id, source_processor) in processors.processors().iter() {
                        let stats = source_processor.source_stats();
                        Self::print_source_stats(*source_id, stats);

                        stats.reset();
                    }
                }
            } else {
                tracing::warn!("Could not read source processors");
            }
        });

        // Spawn an independent task to print GPU statistics
        //
        // Only present on a GPU deployment - on CPU there is nothing to report and the
        // thread is never spawned. The NVML handle is moved into the thread for its lifetime.
        let gpu_stats_handle = self.nvml.take().map(|nvml| {
            let gpu_stats_is_running = Arc::clone(&self.is_running);
            let gpu_stats_interval = GPU_STATS_INTERVAL;

            tokio::task::spawn_blocking(move || {
                while gpu_stats_is_running.load(Ordering::Relaxed) {
                    let start_time = std::time::Instant::now();

                    if let Ok(gpu_stats) = Self::get_gpu_statistics(&nvml) {
                        Self::print_gpu_stats(&gpu_stats)
                    }

                    // Sleep for the remaining period of time
                    let elapsed = start_time.elapsed();
                    if elapsed < gpu_stats_interval {
                        std::thread::sleep(gpu_stats_interval - elapsed);
                    }
                }
            })
        });

        self.source_stats_handle = Some(source_stats_handle);
        self.gpu_stats_handle = gpu_stats_handle;

        Ok(())
    }

    /// Returns statistics about the NVIDIA GPU installed on the machine
    pub fn get_gpu_statistics(nvml: &Nvml) -> Result<GPUStats> {
        let device = nvml
            .device_by_index(0)
            .context("Error getting GPU ID 0 device")?;

        // GPU general information
        let hardware_name = device.name().context("Error getting GPU name")?;
        let gpu_uuid = device.uuid().unwrap_or("".to_string());
        let gpu_serial = device.serial().unwrap_or("".to_string());

        // GPU memory information
        let memory_info = device
            .memory_info()
            .context("Error getting GPU memory information")?;
        let gpu_memory_total = memory_info.total / 1024 / 1024;
        let gpu_memory_used = memory_info.used / 1024 / 1024;
        let gpu_memory_free = memory_info.free / 1024 / 1024;
        let mut gpu_memory: u32 = 0;

        if gpu_memory_total > 0 {
            gpu_memory = (gpu_memory_used as f32 * 100.0 / gpu_memory_total as f32) as u32;
        }

        // GPU performance information
        let utilization = device
            .utilization_rates()
            .context("Error getting GPU utilization information")?;
        let gpu_util = utilization.gpu;

        Ok(GPUStats {
            hardware_name,
            uuid: gpu_uuid,
            serial: gpu_serial,
            memory_total: gpu_memory_total,
            memory_used: gpu_memory_used,
            memory_free: gpu_memory_free,
            util_perc: gpu_util,
            memory_perc: gpu_memory,
        })
    }

    /// Reports GPU Statistics
    pub fn print_gpu_stats(stats: &GPUStats) {
        tracing::info!(
            hardware_name = stats.hardware_name,
            uuid = stats.uuid,
            serial = stats.serial,
            memory_total_mb = stats.memory_total,
            memory_used_mb = stats.memory_used,
            memory_free_mb = stats.memory_free,
            util_perc = stats.util_perc,
            memory_perc = stats.memory_perc,
            "GPU utilization information"
        );
    }

    /// Reports inference statistics for the given source processor
    fn print_source_stats(source_id: u32, source_stats: &SourceStats) {
        let mut avg_queue: f64 = 0.00;
        let mut avg_pre_proc: f64 = 0.00;
        let mut avg_inference: f64 = 0.00;
        let mut avg_post_proc: f64 = 0.00;
        let mut avg_results: f64 = 0.00;
        let mut avg_processing: f64 = 0.00;

        // Extract values of statistics
        let frames_total = source_stats.frames_total.load(Ordering::Relaxed);
        let frames_expected = source_stats.frames_expected.load(Ordering::Relaxed);
        let frames_success = source_stats.frames_success.load(Ordering::Relaxed);
        let frames_failed = source_stats.frames_failed.load(Ordering::Relaxed);
        let total_queue_time = source_stats.total_queue_time.load(Ordering::Relaxed);
        let total_pre_proc_time = source_stats.total_pre_proc_time.load(Ordering::Relaxed);
        let total_inference_time = source_stats.total_inference_time.load(Ordering::Relaxed);
        let total_post_proc_time = source_stats.total_post_proc_time.load(Ordering::Relaxed);
        let total_results_time = source_stats.total_results_time.load(Ordering::Relaxed);
        let total_processing_time = source_stats.total_processing_time.load(Ordering::Relaxed);

        if frames_success > 0 {
            avg_queue = (total_queue_time as f64) / (frames_success as f64);
            avg_pre_proc = (total_pre_proc_time as f64) / (frames_success as f64);
            avg_inference = (total_inference_time as f64) / (frames_success as f64);
            avg_post_proc = (total_post_proc_time as f64) / (frames_success as f64);
            avg_results = (total_results_time as f64) / (frames_success as f64);
            avg_processing = (total_processing_time as f64) / (frames_success as f64);
        }

        tracing::info!(
            source_id = source_id,
            frames_total = frames_total,
            frames_expected = frames_expected,
            frames_success = frames_success,
            frames_failed = frames_failed,
            avg_queue = avg_queue,
            avg_pre_proc = avg_pre_proc,
            avg_inference = avg_inference,
            avg_post_proc = avg_post_proc,
            avg_results = avg_results,
            avg_processing = avg_processing,
            "inference statistics"
        );
    }
}

impl Drop for Statistics {
    fn drop(&mut self) {
        self.is_running.store(false, Ordering::Relaxed);

        // Abort tokio tasks
        if let Some(handle) = &self.source_stats_handle {
            handle.abort();
        }
        if let Some(handle) = &self.gpu_stats_handle {
            handle.abort();
        }
    }
}
