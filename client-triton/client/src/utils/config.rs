//! Responsible for holding all application configuration under one place
//! for easy access and setting format for same variables

use dotenvy::from_path;
use std::path::{Path};
use std::env;
use std::collections::HashMap;
use std::str::FromStr;
use anyhow::{self, Result, Context};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, fmt};
use tracing_appender::rolling::{RollingFileAppender, Rotation};

// Custom modules
use crate::inference::{InferencePrecision, InferenceModelType};
use crate::utils;

/// Represents the local environment the codebase is on
/// 
/// Used mostly to read an environment variables file and
/// use it in the application rather than external variables
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Environment {
    Production,
    NonProduction
}

impl FromStr for Environment {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_uppercase().as_str() {
            "NP" => Ok(Environment::NonProduction),
            "PROD" => Ok(Environment::Production),
            _ => anyhow::bail!("Invalid environment type")
        }
    }
}

#[derive(Clone)]
pub struct ModelConfig {
    pub model_type: InferenceModelType,
    pub input_name: String,
    pub input_shape: Vec<i64>,
    pub output_name: String,
    pub output_shape: Vec<i64>,
    pub max_batch_size: usize,
    pub max_queue_delay: usize,
    pub perf_batch_sizes: Vec<usize>,
    pub precision: InferencePrecision
}

#[derive(Clone)]
pub struct SourcesConfig {
    pub sources: HashMap<String, SourceConfig>,
    pub conf_default: f32,
    pub inf_frame_default: usize,
    pub nms_iou_default: f32
}

#[derive(Clone)]
pub struct SourceConfig {
    pub inf_frame: usize,
    pub conf_threshold: f32,
    pub nms_iou_threshold: f32
}

#[derive(Clone)]
pub struct TritonConfig {
    pub url: String,
    pub models_dir: String,
    pub model_name: String,
    pub model_version: String
}

#[derive(Debug, Clone)]
pub struct KafkaConfig {
    pub brokers: String,
    pub topic: String
}


/// Represents all the configuation variables used by the application
pub struct AppConfig {
    local: bool,
    environment: Environment,
    gpu_name: String,
    sources_config: SourcesConfig,
    kafka_config: KafkaConfig,
    triton_config: TritonConfig,
    model_config: ModelConfig,
}

impl AppConfig {
    /// Creates a new instance of the configuration object
    pub fn new() -> Result<Self> {
        let app_local: bool = env::var("APP_LOCAL")
            .unwrap_or_default()
            .parse()
            .unwrap_or(false);

        let app_env: Environment = env::var("APP_ENV")
            .unwrap_or_default()
            .parse()
            .unwrap_or(Environment::NonProduction);

        // Load variables from local env file
        if app_local {
            AppConfig::load_env_file(app_env)?;
        }

        // Initiate app logging
        AppConfig::init_logging(app_local);

        // GPU information
        let gpu_name = utils::get_gpu_name()
            .context("Error getting GPU name")?;

        // Streams
        let source_ids: Vec<String> = AppConfig::parse_list(
            &env::var("SOURCE_IDS")
            .context("SOURCES_IDS variable not found")?
        );
        let mut sources_config = SourcesConfig {
            sources: HashMap::new(),
            conf_default: env::var("SOURCE_CONF_DEFAULT")
                .unwrap_or("0.85".to_string())
                .parse()
                .context("SOURCE_CONF_DEFAULT must be a float")?,
            inf_frame_default: env::var("SOURCE_INF_FRAME_DEFAULT")
                .unwrap_or("1".to_string())
                .parse()
                .context("SOURCE_INF_FRAME_DEFAULT must be a positive integer")?,
            nms_iou_default: env::var("SOURCE_NMS_IOU_DEFAULT")
                .unwrap_or("0.50".to_string())
                .parse()
                .context("SOURCE_NMS_IOU_DEFAULT must be a float")?
        };

        let source_confs: HashMap<String, f32> = AppConfig::parse_key_values(
            &env::var("SOURCE_CONFS")
            .unwrap_or("".to_string())
        );
        let source_inf_frames: HashMap<String, usize> = AppConfig::parse_key_values(
            &env::var("SOURCE_INF_FRAMES")
            .unwrap_or("".to_string())
        );
        let source_nms_ious: HashMap<String, f32> = AppConfig::parse_key_values(
            &env::var("SOURCE_NMS_IOUS")
            .unwrap_or("".to_string())
        );

        let mut sources: HashMap<String, SourceConfig> = HashMap::new();
        for source in source_ids.iter() {
            let conf_threshold: f32 = source_confs
                .get(source)
                .cloned()
                .filter(|&x| x >= 0.00 && x <= 1.00)
                .unwrap_or(sources_config.conf_default);
            let inf_frame: usize = source_inf_frames
                .get(source)
                .cloned()
                .filter(|&x| x <= 30)
                .unwrap_or(sources_config.inf_frame_default);
            let nms_iou_threshold: f32 = source_nms_ious
                .get(source)
                .cloned()
                .filter(|&x| x >= 0.00 && x <= 1.00)
                .unwrap_or(sources_config.nms_iou_default);
            
            sources.insert(
                source.clone(), 
                SourceConfig {
                    conf_threshold,
                    inf_frame,
                    nms_iou_threshold
                }
            );
        }
        sources_config.sources = sources;

        // Kafka
        let kafka_config = KafkaConfig {
            brokers: env::var("KAFKA_BROKERS")
                .context("KAFKA_BROKERS variable not found")?,
            topic: env::var("KAFKA_TOPIC")
                .context("KAFKA_TOPIC variable not found")?,
        };

        // Triton
        let triton_config = TritonConfig {
            url: env::var("TRITON_URL")
                .context("TRITON_URL variable not found")?,
            models_dir: env::var("TRITON_MODELS_DIR")
                .context("TRITON_MODELS_DIR variable not found")?,
            model_name: env::var("TRITON_MODEL_NAME")
                .context("TRITON_MODEL_NAME variable not found")?,
            model_version: env::var("TRITON_MODEL_VERSION")
                .context("TRITON_MODEL_VERSION variable not found")?
        };

        // Model
        let model_config = ModelConfig { 
            model_type: env::var("MODEL_TYPE")
                .context("MODEL_TYPE variable not found")?
                .parse()
                .context("MODEL_TYPE must be a valid model type")?,
            input_name: env::var("MODEL_INPUT_NAME")
                .context("MODEL_INPUT_NAME variable not found")?,
            input_shape: AppConfig::parse_list(
                    &env::var("MODEL_INPUT_SHAPE")
                    .context("MODEL_INPUT_SHAPE variable not found")?
                ),
            output_name: env::var("MODEL_OUTPUT_NAME")
                .context("MODEL_OUTPUT_NAME variable not found")?,
            output_shape: AppConfig::parse_list(
                    &env::var("MODEL_OUTPUT_SHAPE")
                    .context("MODEL_OUTPUT_SHAPE variable not found")?
                ),
            max_batch_size: env::var("MODEL_MAX_BATCH_SIZE")
                .context("MODEL_MAX_BATCH_SIZE variable not found")?
                .parse()
                .context("MODEL_MAX_BATCH_SIZE must be a positive number")?,
            max_queue_delay: env::var("MODEL_MAX_QUEUE_DELAY")
                .context("MODEL_MAX_QUEUE_DELAY variable not found")?
                .parse()
                .context("MODEL_MAX_QUEUE_DELAY must be a positive number")?,
            perf_batch_sizes: AppConfig::parse_list(
                    &env::var("MODEL_PERF_BATCH_SIZES")
                    .context("MODEL_PERF_BATCH_SIZES variable not found")?
                ),
            precision: env::var("MODEL_PRECISION")
                .context("MODEL_PRECISION variable not found")?
                .parse()
                .context("Must be valid precision")?
        };

        Ok(Self {
            local: app_local,
            environment: app_env,
            gpu_name,
            sources_config,
            kafka_config,
            triton_config,
            model_config,
        })
    }

    /// Loads environment variables from a local .env file
    fn load_env_file(environment: Environment) -> Result<()> {
        let env_file = match environment {
            Environment::Production => ".env",
            Environment::NonProduction => ".env"
        };

        // Path relative to cwd
        let env_file = format!("secrets/{}", env_file);
        let env_path = Path::new(&env_file);
        
        // Load variables to environment
        from_path(env_path)
            .context("Error loading local env file")?;

        Ok(())
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
                    .with_writer(non_blocking)
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
                    .with_writer(std::io::stdout)
            )
            .with(file_layer)
            .init();

        std::mem::forget(_guard);
    }

    /// Parses environment variables as an hashmap
    fn parse_key_values<T>(input: &str) -> HashMap<String, T>
    where
        T: FromStr,
        T::Err: std::fmt::Debug
    {
        input
            .split(',')
            .filter_map(|pair| {
                let mut parts = pair.splitn(2, ':');
                let key = parts.next()?.trim();
                let value_str = parts.next()?.trim();
                match value_str.parse::<T>() {
                    Ok(value) => Some((key.to_string(), value)),
                    Err(_) => None
                }
            })
            .collect()
    }

    /// Parses environment variable as a list
    fn parse_list<T>(input: &str) -> Vec<T>
    where
        T: FromStr,
        T::Err: std::fmt::Debug,
    {
        input
            .split(',')
            .filter_map(|s| s.trim().parse::<T>().ok())
            .collect()
    }

    pub fn set_source_ids(&mut self, ids: &[String]) {
        let mut sources: HashMap<String, SourceConfig> = HashMap::new();

        for source in ids.iter() {
            sources.insert(
                source.clone(), 
                SourceConfig {
                    conf_threshold: self.sources_config().conf_default,
                    inf_frame: self.sources_config().inf_frame_default,
                    nms_iou_threshold: self.sources_config().nms_iou_default
                }
            );
        }

        // Set to config
        self.sources_config.sources = sources;
    }
}

impl AppConfig {
    pub fn is_local(&self) -> bool {
        self.local
    }

    pub fn environment(&self) -> Environment {
        self.environment
    }

    pub fn gpu_name(&self) -> &str {
        &self.gpu_name
    }

    pub fn sources_config(&self) -> &SourcesConfig {
        &self.sources_config
    }

    pub fn kafka_config(&self) -> &KafkaConfig {
        &self.kafka_config
    }

    pub fn triton_config(&self) -> &TritonConfig {
        &self.triton_config
    }

    pub fn model_config(&self) -> &ModelConfig {
        &self.model_config
    }
}