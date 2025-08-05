use dotenvy::from_path;
use std::path::Path;
use std::io::Error;
use std::env;
use std::process::Command;
use tracing_subscriber::{fmt, EnvFilter};
use std::collections::HashMap;
use std::str::FromStr;

// Custom modules
use crate::inference::InferencePrecision;

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum Environment {
    Production,
    NonProduction
}

#[derive(Debug)]
pub struct Config {
    local: bool,
    environment: Environment,
    source_ids: Vec<String>,
    source_confs: HashMap<String, f32>,
    source_conf_default: f32,
    source_inf_frames: HashMap<String, usize>,
    source_inf_frame_default: usize,
    model_name: String,
    model_version: String,
    model_input_name: String,
    model_input_shape: [i64; 3],
    model_output_name: String,
    model_output_shape: [i64; 2],
    nms_iou_threshold: f32,
    nms_conf_thrershold: f32,
    model_precision: InferencePrecision
}

impl Config {
    pub fn new(local: bool, environment: Environment) -> Result<Self, Error> {
        // Load variables from local env file
        if local {
            Config::load_env_file(environment);
        }

        // Initiate app logging
        Config::init_logging();

        // Streams
        let source_ids: Vec<String> = Config::parse_list(
            &env::var("SOURCE_IDS")
            .expect("SOURCES_IDS variable not found")
        );

        // Append confidence threshold for each source
        // Check if source has a prefrred setting and assign default value if not
        let mut source_confs: HashMap<String, f32> = Config::parse_key_values(
            &env::var("SOURCE_CONFS")
            .expect("SOURCES_IDS variable not found")
        );
        let source_conf_default: f32  = env::var("SOURCE_CONF_DEFAULT")
            .expect("SOURCE_CONF_DEFAULT variable not found")
            .parse()
            .expect("SOURCE_CONF_DEFAULT must be a float");

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
        let mut source_inf_frames: HashMap<String, usize> = Config::parse_key_values(
            &env::var("SOURCE_INF_FRAMES")
            .expect("SOURCE_INF_FRAMES variable not found")
        );
        let source_inf_frame_default: usize  = env::var("SOURCE_INF_FRAME_DEFAULT")
            .expect("SOURCE_INF_FRAME_DEFAULT variable not found")
            .parse()
            .expect("SOURCE_INF_FRAME_DEFAULT must be a positive integer");
        
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
        let model_name = env::var("MODEL_NAME")
            .expect("MODEL_NAME variable not found");
        let model_version = env::var("MODEL_VERSION")
            .expect("MODEL_VERSION variable not found");
        let model_input_name = env::var("MODEL_INPUT_NAME")
            .expect("MODEL_INPUT_NAME variable not found");
        let model_output_name = env::var("MODEL_OUTPUT_NAME")
            .expect("MODEL_OUTPUT_NAME variable not found");
        let model_precision: InferencePrecision = env::var("MODEL_PRECISION")
            .expect("MODEL_PRECISION variable not found")
            .parse()
            .expect("Must be valid precision");

        // Model input
        let model_input_shape: Vec<i64> = Config::parse_list(
            &env::var("MODEL_INPUT_SHAPE")
            .expect("MODEL_INPUT_SHAPE variable not found")
        );
        let model_input_shape: [i64; 3] = model_input_shape
            .try_into()
            .expect("Input must be exactly 3 in length (e.g. 3, 640, 640)");

        // Model output
        let model_output_shape: Vec<i64> = Config::parse_list(
            &env::var("MODEL_OUTPUT_SHAPE")
            .expect("MODEL_OUTPUT_SHAPE variable not found")
        );
        let model_output_shape: [i64; 2] = model_output_shape
            .try_into()
            .expect("Output must be exactly 2 in length (e.g. 84, 8400)");

        // Detection processing
        let nms_iou_threshold: f32  = env::var("NMS_IOU_THRESHOLD")
            .expect("NMS_IOU_THRESHOLD variable not found")
            .parse()
            .expect("NMS_IOU_THRESHOLD must be a float");

        let nms_conf_thrershold: f32 = env::var("NMS_CONF_THRESHOLD")
            .expect("NMS_CONF_THRESHOLD variable not found")
            .parse()
            .expect("NMS_CONF_THRESHOLD must be a float");

        Ok(Self {
            local,
            environment,
            source_ids,
            source_confs,
            source_conf_default,
            source_inf_frames,
            source_inf_frame_default,
            model_name,
            model_version,
            model_input_name,
            model_input_shape,
            model_output_name,
            model_output_shape,
            model_precision,
            nms_iou_threshold,
            nms_conf_thrershold
        })
    }

    fn load_env_file(environment: Environment) {
        let base_dir = Path::new(file!()).parent()
            .expect("Error getting config parent directory");

        let env_file = match environment {
            Environment::Production => ".env_test",
            Environment::NonProduction => ".env_test"
        };

        let env_path = base_dir.join(format!("./secrets/{}", env_file)).canonicalize()
            .expect("Local environment variable not found!");

        from_path(env_path).expect("Error loading local env file");
    }

    fn init_logging() {
        tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::from_default_env())
            .json()
            .with_timer(fmt::time::UtcTime::rfc_3339())
            // .with_thread_ids(true)
            // .with_thread_names(true)
            .init();
    }

    fn get_gpu() -> String {
        let output = Command::new("nvidia-smi")
            .args(&["--query-gpu=name", "--format=csv,noheader"])
            .output()
            .expect("failed to execute nvidia-smi");

        String::from_utf8_lossy(&output.stdout).to_string()
    }

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

    fn parse_list<T>(input: &str) -> Vec<T>
    where
        T: FromStr,
        T::Err: std::fmt::Debug,
    {
        input
            .split(',')
            .map(|s| s.trim().parse::<T>().expect("Error parsing list"))
            .collect()
    }
}

impl Config {
    pub fn is_local(&self) -> bool {
        self.local
    }

    pub fn environment(&self) -> Environment {
        self.environment
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

    pub fn nms_iou_threshold(&self) -> f32 {
        self.nms_iou_threshold
    }

    pub fn nms_conf_thrershold(&self) -> f32 {
        self.nms_conf_thrershold
    }

    pub fn model_precision(&self) -> InferencePrecision {
        self.model_precision
    }
}