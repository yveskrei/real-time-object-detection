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
use crate::inference::InferencePrecision;
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

#[derive(Clone)]
pub struct ModelConfig {
    pub name: String,
    pub version: String,
    pub input_name: String,
    pub input_shape: [i64; 3],
    pub output_name: String,
    pub output_shape: [i64; 2],
    pub max_batch_size: usize,
    pub perf_batch_sizes: Vec<usize>,
    pub precision: InferencePrecision
}

pub struct SourcesConfig {
    pub ids: Vec<String>,
    pub confs: HashMap<String, f32>,
    pub conf_default: f32,
    pub inf_frames: HashMap<String, usize>,
    pub inf_frame_default: usize
}

/// Represents all the configuation variables used by the application
pub struct AppConfig {
    local: bool,
    environment: Environment,
    gpu_name: String,
    source_image_test: String,
    sources_config: SourcesConfig,
    triton_url: String,
    triton_models_dir: String,
    model_config: ModelConfig,
    nms_iou_threshold: f32
}

impl AppConfig {
    /// Creates a new instance of the configuration object
    pub fn new(local: bool, environment: Environment) -> Result<Self> {
        // Load variables from local env file
        if local {
            AppConfig::load_env_file(environment)?;
        }

        // Initiate app logging
        AppConfig::init_logging(local);

        // GPU information
        let gpu_name = utils::get_gpu_name()
            .context("Error getting GPU name")?;

        // Streams
        let source_image_test = env::var("SOURCE_IMAGE_TEST")
            .context("SOURCE_IMAGE_TEST variable not found")?;
        let source_ids: Vec<String> = AppConfig::parse_list(
            &env::var("SOURCE_IDS")
            .context("SOURCES_IDS variable not found")?
        );

        // Append confidence threshold for each source
        // Check if source has a prefrred setting and assign default value if not
        let mut source_confs: HashMap<String, f32> = AppConfig::parse_key_values(
            &env::var("SOURCE_CONFS")
            .unwrap_or("".to_string())
        );
        let source_conf_default: f32  = env::var("SOURCE_CONF_DEFAULT")
            .context("SOURCE_CONF_DEFAULT variable not found")?
            .parse()
            .context("SOURCE_CONF_DEFAULT must be a float")?;

        for source in source_ids.iter() {
            let valid = source_confs
                .get(source)
                .map(|v| (0.0..=1.0).contains(v))
                .unwrap_or(false);

            if !valid {
                source_confs.insert(
                    source.to_string(), 
                    source_conf_default
                );
            }
        }

        // Append setting for what frame we want to send inference on for each source
        // Check if source has a prefrred setting and assign default value if not
        let mut source_inf_frames: HashMap<String, usize> = AppConfig::parse_key_values(
            &env::var("SOURCE_INF_FRAMES")
            .unwrap_or("".to_string())
        );
        let source_inf_frame_default: usize  = env::var("SOURCE_INF_FRAME_DEFAULT")
            .context("SOURCE_INF_FRAME_DEFAULT variable not found")?
            .parse()
            .context("SOURCE_INF_FRAME_DEFAULT must be a positive integer")?;
        
        for source in source_ids.iter() {
            let valid = source_inf_frames
                .get(source)
                .map(|v| (0..=30).contains(v))
                .unwrap_or(false);

            if !valid {
                source_inf_frames.insert(
                    source.to_string(), 
                    source_inf_frame_default
                );
            }
        }

        let sources_config = SourcesConfig {
            ids: source_ids,
            confs: source_confs,
            conf_default: source_conf_default,
            inf_frames: source_inf_frames,
            inf_frame_default: source_inf_frame_default
        };

        // Triton
        let triton_url = env::var("TRITON_URL")
            .context("TRITON_URL variable not found")?;
        let triton_models_dir = env::var("TRITON_MODELS_DIR")
            .context("TRITON_MODELS_DIR variable not found")?;

        // Model
        let model_config = ModelConfig { 
            name: env::var("MODEL_NAME")
                .context("MODEL_NAME variable not found")?,

            version: env::var("MODEL_VERSION")
            .context("MODEL_VERSION variable not found")?,

            input_name: env::var("MODEL_INPUT_NAME")
            .context("MODEL_INPUT_NAME variable not found")?,

            input_shape: AppConfig::parse_list(
                &env::var("MODEL_INPUT_SHAPE")
                .context("MODEL_INPUT_SHAPE variable not found")?
            )
                .try_into()
                .map_err(|_| anyhow::anyhow!("Input must be exactly 3 in length (e.g. 3, 640, 640)"))?,

            output_name: env::var("MODEL_OUTPUT_NAME")
                .context("MODEL_OUTPUT_NAME variable not found")?,

            output_shape: AppConfig::parse_list(
                &env::var("MODEL_OUTPUT_SHAPE")
                .context("MODEL_OUTPUT_SHAPE variable not found")?
            )
                .try_into()
                .map_err(|_| anyhow::anyhow!("Output must be exactly 2 in length (e.g. 84, 8400)"))?,

            max_batch_size: env::var("MODEL_MAX_BATCH_SIZE")
                .context("MODEL_MAX_BATCH_SIZE variable not found")?
                .parse()
                .context("MODEL_MAX_BATCH_SIZE must be a positive number")?,

            perf_batch_sizes: AppConfig::parse_list(
                &env::var("MODEL_PERF_BATCH_SIZES")
                    .context("MODEL_PERF_BATCH_SIZES variable not found")?
            ),

            precision: env::var("MODEL_PRECISION")
                .context("MODEL_PRECISION variable not found")?
                .parse()
                .context("Must be valid precision")?
        };

        // Detection processing
        let nms_iou_threshold: f32  = env::var("NMS_IOU_THRESHOLD")
            .context("NMS_IOU_THRESHOLD variable not found")?
            .parse()
            .context("NMS_IOU_THRESHOLD must be a float")?;

        Ok(Self {
            local,
            environment,
            gpu_name,
            source_image_test,
            sources_config,
            triton_url,
            triton_models_dir,
            model_config,
            nms_iou_threshold
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

    pub fn set_source_ids(&mut self, source_ids: Vec<String>) {
        let mut source_confs: HashMap<String, f32> = HashMap::new();
        let mut source_inf_frames: HashMap<String, usize> = HashMap::new();

        for source_id in source_ids.iter() {
            source_confs.insert(
                source_id.to_string(), 
                self.sources_config.conf_default
            );

            source_inf_frames.insert(
                source_id.to_string(), 
                self.sources_config.inf_frame_default
            );
        }

        // Set to config
        self.sources_config.ids = source_ids;
        self.sources_config.confs = source_confs;
        self.sources_config.inf_frames = source_inf_frames;
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

    pub fn source_image_test(&self) -> &str {
        &self.source_image_test
    }

    pub fn sources_config(&self) -> &SourcesConfig {
        &self.sources_config
    }

    pub fn triton_url(&self) -> &str {
        &self.triton_url
    }

    pub fn triton_models_dir(&self) -> &str {
        &self.triton_models_dir
    }

    pub fn model_config(&self) -> &ModelConfig {
        &self.model_config
    }

    pub fn nms_iou_threshold(&self) -> f32 {
        self.nms_iou_threshold
    }
}