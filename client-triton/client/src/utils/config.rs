//! Responsible for holding all application configuration under one place
//! for easy access and setting format for same variables

use dotenvy::from_path;
use std::path::{Path};
use std::env;
use tracing_subscriber::{fmt, EnvFilter};
use std::collections::HashMap;
use std::str::FromStr;
use anyhow::{self, Result, Context};

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

/// Represents all the configuation variables used by the application
pub struct AppConfig {
    local: bool,
    environment: Environment,
    gpu_name: String,
    source_image_test: String,
    source_ids: Vec<String>,
    source_confs: HashMap<String, f32>,
    source_conf_default: f32,
    source_inf_frames: HashMap<String, usize>,
    source_inf_frame_default: usize,
    triton_url: String,
    triton_models_dir: String,
    model_name: String,
    model_version: String,
    model_input_name: String,
    model_input_shape: [i64; 3],
    model_output_name: String,
    model_output_shape: [i64; 2],
    model_precision: InferencePrecision,
    nms_iou_threshold: f32,
    nms_conf_thrershold: f32,
    s3_access_key: String,
    s3_secret_key: String,
    s3_endpoint: String,
    s3_region: String,
    s3_models_bucket: String,
    s3_model_path: String
}

impl AppConfig {
    /// Creates a new instance of the configuration object
    pub fn new(local: bool, environment: Environment) -> Result<Self> {
        // Load variables from local env file
        if local {
            AppConfig::load_env_file(environment)?;
        }

        // Initiate app logging
        AppConfig::init_logging();

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

        // Inference model
        let triton_url = env::var("TRITON_URL")
            .context("TRITON_URL variable not found")?;
        let triton_models_dir = env::var("TRITON_MODELS_DIR")
            .context("TRITON_MODELS_DIR variable not found")?;
        let model_name = env::var("MODEL_NAME")
            .context("MODEL_NAME variable not found")?;
        let model_version = env::var("MODEL_VERSION")
            .context("MODEL_VERSION variable not found")?;
        let model_input_name = env::var("MODEL_INPUT_NAME")
            .context("MODEL_INPUT_NAME variable not found")?;
        let model_output_name = env::var("MODEL_OUTPUT_NAME")
            .context("MODEL_OUTPUT_NAME variable not found")?;
        let model_precision: InferencePrecision = env::var("MODEL_PRECISION")
            .context("MODEL_PRECISION variable not found")?
            .parse()
            .context("Must be valid precision")?;

        // Model input
        let model_input_shape: Vec<i64> = AppConfig::parse_list(
            &env::var("MODEL_INPUT_SHAPE")
            .context("MODEL_INPUT_SHAPE variable not found")?
        );
        let model_input_shape: [i64; 3] = model_input_shape
            .try_into()
            .map_err(|_| anyhow::anyhow!("Input must be exactly 3 in length (e.g. 3, 640, 640)"))?;

        // Model output
        let model_output_shape: Vec<i64> = AppConfig::parse_list(
            &env::var("MODEL_OUTPUT_SHAPE")
            .context("MODEL_OUTPUT_SHAPE variable not found")?
        );
        let model_output_shape: [i64; 2] = model_output_shape
            .try_into()
            .map_err(|_| anyhow::anyhow!("Output must be exactly 2 in length (e.g. 84, 8400)"))?;

        // Detection processing
        let nms_iou_threshold: f32  = env::var("NMS_IOU_THRESHOLD")
            .context("NMS_IOU_THRESHOLD variable not found")?
            .parse()
            .context("NMS_IOU_THRESHOLD must be a float")?;

        let nms_conf_thrershold: f32 = env::var("NMS_CONF_THRESHOLD")
            .context("NMS_CONF_THRESHOLD variable not found")?
            .parse()
            .context("NMS_CONF_THRESHOLD must be a float")?;

        // S3 information
        let s3_access_key = env::var("S3_ACCESS_KEY")
            .context("S3_ACCESS_KEY variable not found")?;
        let s3_secret_key = env::var("S3_SECRET_KEY")
            .context("S3_SECRET_KEY variable not found")?;
        let s3_endpoint = env::var("S3_ENDPOINT")
            .context("S3_ENDPOINT variable not found")?;
        let s3_region = env::var("S3_REGION")
            .context("S3_REGION variable not found")?;
        let s3_models_bucket = env::var("S3_MODELS_BUCKET")
            .context("S3_MODELS_BUCKET variable not found")?;
        let s3_model_path = env::var("S3_MODEL_PATH")
            .context("S3_MODEL_PATH variable not found")?;

        Ok(Self {
            local,
            environment,
            gpu_name,
            source_image_test,
            source_ids,
            source_confs,
            source_conf_default,
            source_inf_frames,
            source_inf_frame_default,
            triton_url,
            triton_models_dir,
            model_name,
            model_version,
            model_input_name,
            model_input_shape,
            model_output_name,
            model_output_shape,
            model_precision,
            nms_iou_threshold,
            nms_conf_thrershold,
            s3_access_key,
            s3_secret_key,
            s3_endpoint,
            s3_region,
            s3_models_bucket,
            s3_model_path
        })
    }

    /// Loads environment variables from a local .env file
    fn load_env_file(environment: Environment) -> Result<()> {
        let base_dir = Path::new(file!()).parent()
            .context("Error getting config parent directory")?;

        let env_file = match environment {
            Environment::Production => ".env_test",
            Environment::NonProduction => ".env_test"
        };

        let env_path = base_dir.join(format!("../secrets/{}", env_file))
            .canonicalize()
            .context("Local environment variable not found!")?;
        
        // Load variables to environment
        from_path(env_path)
            .expect("Error loading local env file");

        Ok(())
    }

    /// Initiates structured logging
    fn init_logging() {
        tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::from_default_env())
            .json()
            .with_timer(fmt::time::UtcTime::rfc_3339())
            // .with_thread_ids(true)
            // .with_thread_names(true)
            .init();
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

    pub fn source_ids(&self) -> &Vec<String> {
        &self.source_ids
    }

    pub fn source_confs(&self) -> &HashMap<String, f32> {
        &self.source_confs
    }

    pub fn source_conf_default(&self) -> f32 {
        self.source_conf_default
    }

    pub fn source_inf_frames(&self) -> &HashMap<String, usize> {
        &self.source_inf_frames
    }

    pub fn source_inf_frame_default(&self) -> usize {
        self.source_inf_frame_default
    }

    pub fn triton_url(&self) -> &str {
        &self.triton_url
    }

    pub fn triton_models_dir(&self) -> &str {
        &self.triton_models_dir
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn model_version(&self) -> &str {
        &self.model_version
    }

    pub fn model_input_name(&self) -> &str {
        &self.model_input_name
    }

    pub fn model_input_shape(&self) -> &[i64; 3] {
        &self.model_input_shape
    }

    pub fn model_output_name(&self) -> &str {
        &self.model_output_name
    }

    pub fn model_output_shape(&self) -> &[i64; 2] {
        &self.model_output_shape
    }

    pub fn model_precision(&self) -> InferencePrecision {
        self.model_precision
    }

    pub fn nms_iou_threshold(&self) -> f32 {
        self.nms_iou_threshold
    }

    pub fn nms_conf_thrershold(&self) -> f32 {
        self.nms_conf_thrershold
    }

    pub fn s3_access_key(&self) -> &str {
        &self.s3_access_key
    }

    pub fn s3_secret_key(&self) -> &str {
        &self.s3_secret_key
    }

    pub fn s3_endpoint(&self) -> &str {
        &self.s3_endpoint
    }

    pub fn s3_region(&self) -> &str {
        &self.s3_region
    }

    pub fn s3_models_bucket(&self) -> &str {
        &self.s3_models_bucket
    }

    pub fn s3_model_path(&self) -> &str {
        &self.s3_model_path
    }
}