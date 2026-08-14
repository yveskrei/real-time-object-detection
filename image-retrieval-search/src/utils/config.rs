//! Responsible for holding all application configuration under one place
//! for easy access and setting format for same variables

use anyhow::{self, Context, Result};
use bincode::{Decode, Encode};
use nvml_wrapper::Nvml;
use serde::{Deserialize, Serialize};
use serde_yaml;
use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};
use utoipa::ToSchema;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, Deserialize, Serialize, ToSchema)]
pub enum SearchType {
    LIGHT,
    MEDIUM,
    HEAVY,
}

impl FromStr for SearchType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "LIGHT" => Ok(SearchType::LIGHT),
            "MEDIUM" => Ok(SearchType::MEDIUM),
            "HEAVY" => Ok(SearchType::HEAVY),
            _ => Err(anyhow::anyhow!("Invalid search type: {}", s)),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub model_type: ModelType,
    /// Omit to take the device default - FP32 on CPU, FP16 on GPU. An explicit value
    /// always wins, and must match the precision the artifact was actually built at:
    /// this drives the client's own pre/post-processing widths, so a mismatch produces
    /// wrong numbers rather than an error. Resolved by [`AppConfig::new`], so
    /// [`ModelConfig::precision`] is always populated by the time anything reads it.
    #[serde(default)]
    precision: Option<InferencePrecision>,
    pub input_name: String,
    pub input_shape: Vec<i64>,
    pub output_name: String,
    pub output_shape: Vec<i64>,
    pub batch_max_size: u32,
    pub batch_max_queue_delay: u32,
    pub batch_preferred_sizes: Vec<u32>,
}

impl ModelConfig {
    /// The precision this model runs at, resolved against the device
    pub fn precision(&self) -> InferencePrecision {
        self.precision
            .expect("precision is resolved for every model in AppConfig::new")
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct TritonConfig {
    pub url: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ElasticConfig {
    pub url: String,
    pub index_name: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct InferenceConfig {
    pub models: HashMap<ModelPurpose, ModelConfig>,
    pub instances: InstancesConfig,
}

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

/// Represents the model architecture, which decides how its output tensor is decoded
#[derive(PartialEq, Eq, Clone, Copy, Debug, Deserialize)]
pub enum ModelType {
    DINOV3,
}

impl ModelType {
    pub fn to_string(&self) -> String {
        match self {
            ModelType::DINOV3 => "DINOV3",
        }
        .to_string()
    }
}

/// Represents porpose for model - What task is it defined to perform
///
/// Unlike the ingest clients, this one accepts the purpose over HTTP and stores it
/// alongside a pending upload, so it carries the serialization derives the API layer
/// (`utoipa`, `serde`) and the Redis payload (`bincode`) need.
#[derive(
    PartialEq, Eq, Clone, Copy, Debug, Deserialize, Serialize, Hash, Encode, Decode, ToSchema,
)]
pub enum ModelPurpose {
    FrameEmbedding,
    BBOXEmbedding,
}

impl ModelPurpose {
    pub fn to_string(&self) -> String {
        match self {
            ModelPurpose::FrameEmbedding => "Frame Embedding",
            ModelPurpose::BBOXEmbedding => "BBOX Embedding",
        }
        .to_string()
    }
}

impl FromStr for ModelPurpose {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "FRAMEEMBEDDING" => Ok(ModelPurpose::FrameEmbedding),
            "BBOXEMBEDDING" => Ok(ModelPurpose::BBOXEmbedding),
            _ => Err(anyhow::anyhow!("Invalid model purpose: {}", s)),
        }
    }
}

/// Name of the environment variable naming the inference device
pub const DEVICE_TYPE_ENV: &str = "DEVICE_TYPE";

/// Represents the type of device we perform inference on
///
/// Read from the `DEVICE_TYPE` environment variable rather than the configuration
/// file, so one variable switches the whole deployment and the moon `:cpu` / `:gpu`
/// tasks can set it. Selects the Triton platform, the model file the server expects,
/// the instance kind, and the default precision:
/// - `GPU` -> `tensorrt_plan`, `model.plan`, `KIND_GPU`, FP16
/// - `CPU` -> `onnxruntime_onnx`, `model.onnx`, `KIND_CPU`, FP32
#[derive(PartialEq, Eq, Clone, Copy, Debug, Deserialize)]
pub enum DeviceType {
    GPU,
    CPU,
}

impl FromStr for DeviceType {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_uppercase().as_str() {
            "GPU" => Ok(DeviceType::GPU),
            "CPU" => Ok(DeviceType::CPU),
            other => anyhow::bail!("Invalid device type '{}', expected CPU or GPU", other),
        }
    }
}

impl DeviceType {
    pub fn to_string(&self) -> String {
        match self {
            DeviceType::GPU => "GPU",
            DeviceType::CPU => "CPU",
        }
        .to_string()
    }

    /// Reads the inference device from the environment
    ///
    /// Deliberately has no default. The device decides the Triton platform, the model
    /// file, the instance kind and the default precision, so guessing it wrong yields a
    /// model that loads and returns nonsense rather than one that fails loudly.
    pub fn from_env() -> Result<Self> {
        let raw = std::env::var(DEVICE_TYPE_ENV).map_err(|_| {
            anyhow::anyhow!(
                "{} is not set. Set it to CPU or GPU - for example \
                 `{}=CPU cargo run --release`. The moon tasks set it for you: \
                 `moon run <project>:cpu` or `moon run <project>:gpu`.",
                DEVICE_TYPE_ENV,
                DEVICE_TYPE_ENV
            )
        })?;

        raw.parse()
            .with_context(|| format!("Error reading {}", DEVICE_TYPE_ENV))
    }

    /// Precision assumed for a model that does not declare one
    ///
    /// CPU runs through ONNX Runtime, whose exports here are FP32; GPU runs prebuilt
    /// TensorRT engines, which are FP16. A model whose artifact differs declares
    /// `precision` explicitly.
    pub fn default_precision(&self) -> InferencePrecision {
        match self {
            DeviceType::GPU => InferencePrecision::FP16,
            DeviceType::CPU => InferencePrecision::FP32,
        }
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

#[derive(PartialEq, Clone, Debug, Deserialize)]
pub struct SearchConfigOption {
    pub output_vectors: u32,
    pub num_candidates: u32,
    pub centriod_visit_percentage: u32,
    pub vector_oversample_multiplier: f32,
}

#[derive(PartialEq, Clone, Debug, Deserialize)]
pub struct RedisConfig {
    pub url: String,
    pub username: String,
    pub password: String,
}

/// Represents all the configuation variables used by the application
#[derive(Debug, Deserialize)]
pub struct AppConfig {
    local: bool,
    port: u16,
    elastic_config: ElasticConfig,
    triton_config: TritonConfig,
    redis_config: RedisConfig,
    inference_config: InferenceConfig,
    search_config: HashMap<SearchType, SearchConfigOption>,

    /// Read from `DEVICE_TYPE` at startup, not from the configuration file.
    /// `Option` only so `serde(skip)` has a `Default`; [`AppConfig::new`] always fills it.
    #[serde(skip)]
    device_type: Option<DeviceType>,

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

        // Resolve the device from the environment before anything that depends on it.
        let device_type = DeviceType::from_env()?;
        config.device_type = Some(device_type);

        // Resolve the hardware we perform inference on. Bails if the device is GPU
        // but the GPU name cannot be read.
        config.hardware_name = device_type
            .hardware_name()
            .context("Error resolving inference hardware name")?;

        // Give every model that did not declare a precision the device default, so the
        // rest of the application only ever sees a resolved value.
        let default_precision = device_type.default_precision();
        for model_config in config.inference_config.models.values_mut() {
            model_config.precision.get_or_insert(default_precision);
        }

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

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn elastic_config(&self) -> &ElasticConfig {
        &self.elastic_config
    }

    pub fn triton_config(&self) -> &TritonConfig {
        &self.triton_config
    }

    pub fn redis_config(&self) -> &RedisConfig {
        &self.redis_config
    }

    pub fn inference_config(&self) -> &InferenceConfig {
        &self.inference_config
    }

    pub fn search_config(&self) -> &HashMap<SearchType, SearchConfigOption> {
        &self.search_config
    }

    /// The device we perform inference on, from `DEVICE_TYPE`
    pub fn device_type(&self) -> DeviceType {
        self.device_type
            .expect("device type is resolved in AppConfig::new")
    }

    pub fn hardware_name(&self) -> &str {
        &self.hardware_name
    }
}
