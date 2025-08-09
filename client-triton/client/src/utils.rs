use image::{ImageReader, GenericImageView};
use anyhow::{Result, Context};
use nvml_wrapper::Nvml;

// Custom modules
pub mod config;
pub mod s3;

/// Represents GPU statistics that are reported by the application
pub struct GPUStats {
    pub name: String,
    pub uuid: String,
    pub serial: String,
    pub memory_total: u64,
    pub memory_used: u64,
    pub memory_free: u64,
    pub util_perc: u32,
    pub memory_perc: u32,
}

/// used to get image from path, returns as raw bytes
pub fn get_image_raw(path: &str) -> Result<(Vec<u8>, usize, usize)> {
    let image = ImageReader::open(path)
        .context("Error opening image from path")?
        .decode()
        .context("Error decoding image")?;

    // Get dimensions
    let (width, height) = image.dimensions();
    let width = width as usize;
    let height = height as usize;

    // Convert to RGB8 if needed
    let img_rgb8 = image.to_rgb8();

    // Get raw pixel data
    Ok((img_rgb8.into_raw(), height, width))
}

/// Returns the name of the NVIDIA GPU installed on the machine
pub fn get_gpu_name() -> Result<String> {
    let nvml = Nvml::init()
        .context("Error initiating NVML wrapper")?;
    let device = nvml.device_by_index(0)
        .context("Error getting GPU ID 0 device")?;
    Ok(
        device.name()
            .context("Error getting GPU ID 0 Name")?
    )
}


/// Returns statistics about the NVIDIA GPU installed on the machine
pub fn get_gpu_statistics() -> Result<GPUStats> {
    let nvml = Nvml::init()
        .context("Error initiating NVML wrapper")?;
    let device = nvml.device_by_index(0)
        .context("Error getting GPU ID 0 device")?;

    // GPU general information
    let gpu_name = device.name()
        .context("Error getting GPU name")?;
    let gpu_uuid = device.uuid()
        .unwrap_or("".to_string());
    let gpu_serial = device.serial()
        .unwrap_or("".to_string());


    // GPU memory information
    let memory_info = device.memory_info()
        .context("Error getting GPU memory information")?;
    let gpu_memory_total = memory_info.total / 1024 / 1024;
    let gpu_memory_used = memory_info.used / 1024 / 1024;
    let gpu_memory_free = memory_info.free / 1024 / 1024;
    let mut gpu_memory: u32 = 0;

    if gpu_memory_total > 0 {
        gpu_memory = (gpu_memory_used as f32 * 100.0 / gpu_memory_total as f32) as u32;
    }

    // GPU performance information
    let utilization = device.utilization_rates()
        .context("Error getting GPU utilization information")?;
    let gpu_util = utilization.gpu;


    Ok(
        GPUStats {
            name: gpu_name,
            uuid: gpu_uuid,
            serial: gpu_serial,
            memory_total: gpu_memory_total,
            memory_used: gpu_memory_used,
            memory_free: gpu_memory_free,
            util_perc: gpu_util,
            memory_perc: gpu_memory
        }
    )
}