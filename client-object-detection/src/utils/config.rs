//! Responsible for holding all application configuration under one place
//! for easy access and setting format for same variables

use anyhow::{self, Context, Result};
use nvml_wrapper::Nvml;
use serde::Deserialize;
use serde_yaml;
use std::collections::HashMap;
use std::path::Path;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Clone, Debug, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub model_type: ModelType,
    pub precision: InferencePrecision,
    pub input_name: String,
    pub input_shape: Vec<i64>,
    pub output_name: String,
    pub output_shape: Vec<i64>,
    pub batch_max_size: u32,
    pub batch_max_queue_delay: u32,
    pub batch_preferred_sizes: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SourcesConfig {
    #[serde(default)]
    pub sources: HashMap<u32, SourceConfig>,
    pub ids: Vec<u32>,
    pub default: SourceConfig,
    #[serde(default)]
    pub custom: HashMap<u32, SourceConfigOptional>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SourceConfig {
    pub inf_frame: u32,
    pub conf_threshold: f32,
    pub nms_iou_threshold: f32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SourceConfigOptional {
    pub inf_frame: Option<u32>,
    pub conf_threshold: Option<f32>,
    pub nms_iou_threshold: Option<f32>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TritonConfig {
    pub url: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct InferenceConfig {
    pub device_type: DeviceType,
    pub instances: InstancesConfig,
    pub models: HashMap<ModelPurpose, ModelConfig>,
}

/// Represents how many instances of each model we load onto Triton
///
/// `custom` is keyed first by the hardware name reported by `DeviceType::hardware_name`,
/// and then by the model purpose - so a deployment can tune the count per hardware
/// *and* per task.
#[derive(Clone, Debug, Deserialize)]
pub struct InstancesConfig {
    pub default: u32,
    #[serde(default)]
    pub custom: HashMap<String, HashMap<ModelPurpose, u32>>,
}

impl InstancesConfig {
    /// Resolves instance count for the given hardware and model purpose,
    /// falling back to `default` when either level is absent.
    pub fn resolve(&self, hardware_name: &str, model_purpose: ModelPurpose) -> u32 {
        self.custom
            .get(hardware_name)
            .and_then(|per_purpose| per_purpose.get(&model_purpose))
            .copied()
            .unwrap_or(self.default)
    }
}

/// Represents the model architecture, which decides how its output tensor is decoded
#[derive(PartialEq, Eq, Clone, Copy, Debug, Deserialize)]
pub enum ModelType {
    YOLOV9,
    YOLO26,
}

impl ModelType {
    pub fn to_string(&self) -> String {
        match self {
            ModelType::YOLOV9 => "YOLOV9",
            ModelType::YOLO26 => "YOLO26",
        }
        .to_string()
    }
}

/// Represents porpose for model - What task is it defined to perform
#[derive(PartialEq, Eq, Clone, Copy, Debug, Deserialize, Hash)]
pub enum ModelPurpose {
    FrameDetection,
}

impl ModelPurpose {
    pub fn to_string(&self) -> String {
        match self {
            ModelPurpose::FrameDetection => "Frame Detection",
        }
        .to_string()
    }
}

/// Represents the inference model precision type
#[derive(PartialEq, Eq, Clone, Copy, Debug, Deserialize)]
pub enum InferencePrecision {
    FP32,
    FP16,
}

impl InferencePrecision {
    pub fn to_string(&self) -> String {
        match self {
            InferencePrecision::FP32 => "FP32",
            InferencePrecision::FP16 => "FP16",
        }
        .to_string()
    }
}

/// Represents the type of device we perform inference on
///
/// Selects the Triton platform, the model file the server expects, and the
/// instance kind:
/// - `GPU` -> `tensorrt_plan`, `model.plan`, `KIND_GPU`
/// - `CPU` -> `onnxruntime_onnx`, `model.onnx`, `KIND_CPU`
#[derive(PartialEq, Eq, Clone, Copy, Debug, Deserialize)]
pub enum DeviceType {
    GPU,
    CPU,
}

impl DeviceType {
    pub fn to_string(&self) -> String {
        match self {
            DeviceType::GPU => "GPU",
            DeviceType::CPU => "CPU",
        }
        .to_string()
    }

    pub fn hardware_name(&self) -> Result<String> {
        match self {
            DeviceType::CPU => Ok("CPU".to_string()),
            DeviceType::GPU => {
                let nvml =
                    Nvml::init().context("Error initiating NVML wrapper to resolve GPU name")?;

                let device = nvml
                    .device_by_index(0)
                    .context("Error getting GPU ID 0 device")?;

                match device.name() {
                    Ok(name) => Ok(name),
                    Err(e) => anyhow::bail!("Error getting GPU name: {}", e),
                }
            }
        }
    }
}

/// Represents all the configuation variables used by the application
#[derive(Debug, Deserialize)]
pub struct AppConfig {
    local: bool,
    sources_config: SourcesConfig,
    triton_config: TritonConfig,
    inference_config: InferenceConfig,

    /// Resolved at startup, not read from the configuration file
    #[serde(skip)]
    hardware_name: String,
}

impl AppConfig {
    /// Creates a new instance of the configuration object
    pub fn new() -> Result<Self> {
        let mut config: AppConfig =
            AppConfig::load_config_file().context("Error loading configuation file")?;

        // Initiate app logging
        AppConfig::init_logging(config.local);

        // Parse sources
        let mut sources: HashMap<u32, SourceConfig> = HashMap::new();
        for source_id in config.sources_config().ids.iter() {
            // Get source preferred config
            let mut source_config = config.sources_config().default.clone();
            let custom_config = config.sources_config().custom.get(source_id);

            // Assign custom values - override defaults if exist
            source_config.inf_frame = custom_config
                .and_then(|o| o.inf_frame)
                .filter(|&x| x >= 1 && x <= 30)
                .unwrap_or(source_config.inf_frame);

            source_config.conf_threshold = custom_config
                .and_then(|o| o.conf_threshold)
                .filter(|&x| x >= 0.00 && x <= 1.00)
                .unwrap_or(source_config.conf_threshold);

            source_config.nms_iou_threshold = custom_config
                .and_then(|o| o.nms_iou_threshold)
                .filter(|&x| x >= 0.00 && x <= 1.00)
                .unwrap_or(source_config.nms_iou_threshold);

            sources.insert(*source_id, source_config);
        }
        config.sources_config.sources = sources;

        // Resolve the hardware we perform inference on. Bails if a GPU device type
        // is configured but the GPU name cannot be read.
        let device_type = config.inference_config().device_type;
        config.hardware_name = device_type
            .hardware_name()
            .context("Error resolving inference hardware name")?;

        Ok(config)
    }

    /// Loads environment variables from a local .env file
    fn load_config_file() -> Result<AppConfig> {
        // Path relative to cwd
        let config_file = "secrets/config.yaml".to_string();
        let config_path = Path::new(&config_file);

        // Load configuration file
        let contents =
            std::fs::read_to_string(config_path).context("Error locating configuration file")?;

        let config_file: AppConfig =
            serde_yaml::from_str(&contents).context("Error parsing configuration file")?;

        Ok(config_file)
    }

    /// Initiates structured logging
    fn init_logging(local: bool) {
        let file_appender = RollingFileAppender::new(Rotation::NEVER, "logs", "app.log");
        let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

        // Append logging to local file
        let file_layer = if local {
            Some(
                tracing_subscriber::fmt::layer()
                    .json()
                    .with_timer(fmt::time::UtcTime::rfc_3339())
                    .with_writer(non_blocking),
            )
        } else {
            None
        };

        tracing_subscriber::registry()
            .with(EnvFilter::from_default_env())
            .with(
                // Console layer - pretty format
                tracing_subscriber::fmt::layer()
                    .json()
                    .with_timer(fmt::time::UtcTime::rfc_3339())
                    .with_writer(std::io::stdout),
            )
            .with(file_layer)
            .init();

        std::mem::forget(_guard);
    }
}

impl AppConfig {
    pub fn is_local(&self) -> bool {
        self.local
    }

    pub fn sources_config(&self) -> &SourcesConfig {
        &self.sources_config
    }

    pub fn triton_config(&self) -> &TritonConfig {
        &self.triton_config
    }

    pub fn inference_config(&self) -> &InferenceConfig {
        &self.inference_config
    }

    pub fn hardware_name(&self) -> &str {
        &self.hardware_name
    }
}
