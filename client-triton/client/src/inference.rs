///! Responsible for performing inference with Nvidia Triton Server
///! 
///! Performs operations using gRPC protocol for minimal latency between
///! our application and Triton Server.
///! Allows us to dynamically load models(multiple instances) depending on amount of video sources we have

use triton_client::Client;
use triton_client::inference::{ModelInferRequest, RepositoryModelLoadRequest, ModelRepositoryParameter, RepositoryModelUnloadRequest};
use triton_client::inference::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};
use triton_client::inference::model_repository_parameter::{ParameterChoice};
use std::collections::HashMap;
use serde_json::json;
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::OnceCell;
use anyhow::{self, Result, Context};
use tokio::time::{Duration, interval};

// Custom modules
pub mod source;
pub mod queue;
use crate::utils::{
    self,
    GPUStats,
    config::{AppConfig, ModelConfig, TritonConfig}
};

// Variables
pub static INFERENCE_MODEL: OnceCell<Arc<InferenceModel>> = OnceCell::const_new();
pub static GPU_STATS_INTERVAL: Duration = Duration::from_secs(3);

/// Returns the inference model instance, if initiated
pub fn get_inference_model() -> Result<&'static Arc<InferenceModel>> {
    Ok(
        INFERENCE_MODEL
            .get()
            .context("Infernece model is not initiated!")?
    )
}

/// Initiates a single instance of a model for inference
pub async fn init_inference_model(app_config: &AppConfig) -> Result<()> {
    if let Ok(_) = get_inference_model() {
        anyhow::bail!("Model is already initiated!")
    }

    // Create new instance
    let client_instance = InferenceModel::new(
        app_config.triton_config().clone(),
        app_config.model_config().clone(),
    )
        .await
        .context("Error creating model client")?;

    // Set global variable
    INFERENCE_MODEL.set(Arc::new(client_instance))
        .map_err(|_| anyhow::anyhow!("Error setting model instance"))?;

    Ok(())
}

pub async fn start_model_instances(app_config: &AppConfig) -> Result<()> {
    let client_instance = get_inference_model()?;

    // Calculate total "load units" - how much processing capacity we need
    // Each source contributes fractional load based on its frame rate
    let total_load: f32 = app_config
        .sources_config()
        .sources
        .values()
        .map(|source_config| 1.0 / source_config.inf_frame as f32)
        .sum();
    
    // Get target batch size from config
    let target_batch_size = app_config.model_config()
        .perf_batch_sizes
        .iter()
        .max()
        .copied()
        .unwrap_or(app_config.model_config().max_batch_size) as f32;
    
    // Calculate instances needed
    // Divide total load by target batch size to get base instances
    // Add small overhead for arrival variance
    let batch_efficiency = 0.60; // Assume X% batch fill rate in practice
    let instances = (total_load / (target_batch_size * batch_efficiency))
        .ceil()
        .max(1.0) as usize;

    // Clear previous model instances
    if let Ok(_) = client_instance.unload_model().await {
        tracing::warn!("Unloaded previous model instances")
    }

    // Initiate model instances
    client_instance.load_model(instances).await
        .context("Error loading model instances")?;

    tracing::info!("Initiated {} model instances", instances);

    Ok(())
}

/// Represents the inference model precision type
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum InferencePrecision {
    FP32,
    FP16
}

impl InferencePrecision {
    pub fn to_string(&self) -> String {
        match self {
            InferencePrecision::FP32 => "FP32",
            InferencePrecision::FP16 => "FP16",
        }.to_string()
    }
}

impl FromStr for InferencePrecision {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_uppercase().as_str() {
            "FP32" => Ok(InferencePrecision::FP32),
            "FP16" => Ok(InferencePrecision::FP16),
            _ => anyhow::bail!("Invalid precision")
        }
    }
}

/// Represents type of inference model
#[derive(Clone)]
pub enum InferenceModelType {
    YOLO,
    DINO
}

impl FromStr for InferenceModelType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_uppercase().as_str() {
            "YOLO" => Ok(InferenceModelType::YOLO),
            "DINO" => Ok(InferenceModelType::DINO),
            _ => anyhow::bail!("Invalid model type")
        }
    }
}

/// Represents an instance of an inference model
pub struct InferenceModel {
    client: Client,
    triton_config: TritonConfig,
    model_config: ModelConfig,
    base_request: ModelInferRequest,
    stats_handle: tokio::task::JoinHandle<()>
}

impl InferenceModel {
    /// Create new instance of inference model
    /// 
    /// Creates a new Triton Server client for inference
    /// Initiate all values for fast inference, including a pre-made request body for inference
    /// Reports statistics about GPU utilization
    pub async fn new(
        triton_config: TritonConfig,
        model_config: ModelConfig
    ) -> Result<Self> {
        //Create client instance
        let client = Client::new(&triton_config.url, None)
            .await
            .context("Error creating triton client instance")?;

        // Check if server is ready
        let server_ready = client.server_ready()
            .await
            .context("Error getting model ready status")?;

        if !server_ready.ready {
            anyhow::bail!("Triton server is not ready");
        }

        // Create base inference request
        let mut batch_input_shape = vec![1];
        batch_input_shape.extend(&model_config.input_shape);

        let base_request = ModelInferRequest {
            model_name: triton_config.model_name.to_string(),
            model_version: triton_config.model_version.to_string(),
            id: String::new(),
            parameters: HashMap::new(),
            inputs: vec![
                InferInputTensor {
                    name: model_config.input_name.to_string(),
                    datatype: model_config.precision.to_string(),
                    shape: batch_input_shape,
                    parameters: HashMap::new(),
                    contents: None
                }
            ],
            outputs: vec![
                InferRequestedOutputTensor {
                    name: model_config.output_name.to_string(),
                    parameters: HashMap::new(),
                }
            ],
            raw_input_contents: Vec::new()
        };


        // Spawn seperate task to monitor GPU stats
        let stats_interval = GPU_STATS_INTERVAL.clone();
        let stats_handle = tokio::spawn(async move {
            let mut interval = interval(stats_interval);
            
            loop {
                interval.tick().await;
                
                // NVML Is execution blocking, running it seperately
                let gpu_stats = tokio::task::spawn_blocking(|| {
                    let result = utils::get_gpu_statistics();

                    match result {
                        Ok(stats) => {
                            InferenceModel::process_gpu_stats(stats);
                        },
                        Err(e) => {
                            tracing::warn!(
                                error=e.to_string(),
                                "Error getting GPU utilization information"
                            )
                        }
                    }
                }).await;

                if let Err(e) = gpu_stats {
                    tracing::warn!(
                        error=e.to_string(),
                        "Error getting GPU utilization information"
                    )
                }

            }
        });

        Ok(Self { 
            client,
            triton_config,
            model_config,
            base_request,
            stats_handle
        })
    }

    /// Unloads running instances of a given model
    pub async fn unload_model(&self) -> Result<()> {
        // Unload previous instances of model we're about to load
        self.client.repository_model_unload(RepositoryModelUnloadRequest { 
            repository_name: "".to_string(), 
            model_name: self.triton_config().model_name.to_string(), 
            parameters: HashMap::new()
        })
            .await
            .context("Error unloading previous triton model instances")?;

        Ok(())
    }
    
    /// Loads given amount of instances of a given model
    pub async fn load_model(&self, instances: usize) -> Result<()> {
        let model_config = json!({
            "name": &self.triton_config().model_name.to_string(),
            "platform": "tensorrt_plan",
            "max_batch_size": &self.model_config().max_batch_size,
            "input": [
                {
                    "name": &self.model_config().input_name,
                    "data_type": format!("TYPE_{}", &self.model_config().precision.to_string()),
                    "dims": &self.model_config().input_shape
                }
            ],
            "output": [
                {
                    "name": &self.model_config().output_name,
                    "data_type": format!("TYPE_{}", &self.model_config().precision.to_string()),
                    "dims": &self.model_config().output_shape
                }
            ],
            "instance_group": [
                {
                    "kind": "KIND_GPU",
                    "count": instances,
                    "gpus": [0]
                }
            ],
            "dynamic_batching": {
                "max_queue_delay_microseconds": self.model_config().max_queue_delay,
                "preferred_batch_size": &self.model_config().perf_batch_sizes
            },
            "optimization": {
                "execution_accelerators": {
                "gpu_execution_accelerator": [
                    {
                        "name": "tensorrt",
                        "parameters": {
                            "key": "precision_mode",
                            "value": &self.model_config().precision.to_string()
                        }
                    }
                ]
                },
                "input_pinned_memory": {
                    "enable": true
                },
                "output_pinned_memory": {
                    "enable": true
                },
                "gather_kernel_buffer_threshold": 0
            },
            "model_transaction_policy": {
                "decoupled": false
            },
            "model_warmup": [
                {
                    "name": "warmup_random",
                    "batch_size": self.model_config().max_batch_size,
                    "inputs":  {
                        &self.model_config().input_name: {
                            "dims": &self.model_config().input_shape,
                            "data_type": format!("TYPE_{}", &self.model_config().precision.to_string()),
                            "random_data": true
                        }
                    }
                }
            ]
        });

        // Define model config
        let mut parameters = HashMap::new();
        parameters.insert("config".to_string(), ModelRepositoryParameter{ 
            parameter_choice: Some(ParameterChoice::StringParam(model_config.to_string()))
        });

        // Load selected model
        self.client.repository_model_load(RepositoryModelLoadRequest { 
            repository_name: "".to_string(), 
            model_name: self.triton_config().model_name.to_string(), 
            parameters: parameters
        })
            .await
            .context("Error loading triton model instances")?;

        Ok(())
    }

    /// Performs inference on a raw input, returning raw model results
    pub async fn infer(&self, raw_input: Vec<u8>) -> Result<Vec<u8>> {        
        // tracing::info!("inference!");
        // Create new inference request
        let mut inference_request = self.base_request.clone();
        inference_request.raw_input_contents.push(raw_input);

        // Perform inference
        let inference_result = self.client.model_infer(inference_request)
            .await
            .context("Error sending triton inference request")?;

        // Return inference results
        Ok(
            inference_result.raw_output_contents
                .into_iter()
                .next()
                .context("Error getting inference results for raw input")?
        )
    }

    /// Get Triton Server alive status
    pub async fn is_alive(&self) -> bool {
        let server_alive = &self.client.server_live().await;
        
        match server_alive {
            Ok(response) => return response.live,
            Err(_) => return false

        }
    }

    pub fn process_gpu_stats(stats: GPUStats) {
        tracing::info!(
            name=stats.name,
            uuid=stats.uuid,
            serial=stats.serial,
            memory_total_mb=stats.memory_total,
            memory_used_mb=stats.memory_used,
            memory_free_mb=stats.memory_free,
            util_perc=stats.util_perc,
            memory_perc=stats.memory_perc,
            "GPU utilization information"
        );
    }
}

impl InferenceModel {
    pub fn client(&self) -> &Client {
        &self.client
    }

    pub fn triton_config(&self) -> &TritonConfig {
        &self.triton_config
    }

    pub fn model_config(&self) -> &ModelConfig {
        &self.model_config
    }

    pub fn base_request(&self) -> &ModelInferRequest {
        &self.base_request
    }

    pub fn stats_handle(&self) -> &tokio::task::JoinHandle<()> {
        &self.stats_handle
    }
}

impl Drop for InferenceModel {
    fn drop(&mut self) {
        // Abort tokio task
        self.stats_handle.abort();
    }
}