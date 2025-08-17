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
use tokio::time::{Duration, interval, Instant};

// Custom modules
use crate::utils::config::AppConfig;
pub mod processing;
pub mod source;
use crate::utils;

/// Static singleton instance for model inference
pub static INFERENCE_MODEL: OnceCell<Arc<InferenceModel>> = OnceCell::const_new();

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
        app_config.triton_url().to_string(),
        app_config.model_name().to_string(), 
        app_config.model_version().to_string(), 
        app_config.model_input_name().to_string(), 
        app_config.model_input_shape().clone(),
        app_config.model_output_name().to_string(), 
        app_config.model_output_shape().clone(),
        app_config.model_precision(),
        app_config.nms_iou_threshold()
    )
        .await
        .context("Error creating model client")?;

    // Set global variable
    INFERENCE_MODEL.set(Arc::new(client_instance))
        .map_err(|_| anyhow::anyhow!("Error setting model instance"))?;

    Ok(())
}

pub async fn start_model_instances(instances: usize) -> Result<()> {
    let client_instance = get_inference_model()?;

    // Clear previous model instances
    if let Ok(_) = client_instance.unload_model().await {
        tracing::warn!("Unloaded previous model instances")
    }

    // Initiate model instances
    client_instance.load_model(instances).await
        .context("Error loading model instances")?;

    Ok(())

}

/// Represents raw frame before performing inference on it
#[derive(Clone)]
pub struct InferenceFrame {
    pub data: Vec<u8>,
    pub height: usize,
    pub width: usize,
    pub added: Instant
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

/// Represents a single bbox instance from the model inference results
#[derive(Clone, Copy)]
pub struct InferenceResult {
    pub bbox: [f32; 4],
    pub class: usize, 
    pub score: f32
}

impl InferenceResult {
    pub fn class_name(&self) -> &'static str {
        match self.class {
            0 => "person",
            1 => "bicycle",
            2 => "car",
            3 => "motorcycle",
            4 => "airplane",
            5 => "bus",
            _ => Box::leak(self.class.to_string().into_boxed_str())
        }
    }
}

/// Represents an instance of an inference model
pub struct InferenceModel {
    client: Client,
    triton_url: String,
    model_name: String,
    model_version: String,
    input_name: String,
    input_shape: [i64; 3],
    output_name: String,
    output_shape: [i64; 2],
    precision: InferencePrecision,
    nms_iou_threshold: f32,
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
        triton_url: String,
        model_name: String, 
        model_version: String, 
        input_name: String, 
        input_shape: [i64; 3],
        output_name: String, 
        output_shape: [i64; 2],
        precision: InferencePrecision,
        nms_iou_threshold: f32
    ) -> Result<Self> {
        //Create client instance
        let client = Client::new(&triton_url, None)
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
        batch_input_shape.extend(&input_shape);

        let base_request = ModelInferRequest {
            model_name: model_name.clone(),
            model_version: model_version.clone(),
            id: String::new(),
            parameters: HashMap::new(),
            inputs: vec![
                InferInputTensor {
                    name: input_name.clone(), // Adjust based on your model
                    datatype: precision.to_string(),
                    shape: batch_input_shape,
                    parameters: HashMap::new(),
                    contents: None
                }
            ],
            outputs: vec![
                InferRequestedOutputTensor {
                    name: output_name.clone(), // Adjust based on your model
                    parameters: HashMap::new(),
                }
            ],
            raw_input_contents: Vec::new()
        };


        // Spawn seperate task to monitor GPU stats
        let stats_interval = Duration::from_secs(5);
        let stats_handle = tokio::spawn(async move {
            let mut interval = interval(stats_interval);
            
            loop {
                interval.tick().await;
                
                // NVML Is execution blocking, running it seperately
                let gpu_stats = tokio::task::spawn_blocking(|| {
                    utils::get_gpu_statistics()
                }).await;

                match gpu_stats {
                    Ok(result) => {
                        match result {
                            Ok(stats) => {
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
                            },
                            Err(e) => {
                                tracing::warn!(
                                    error=e.to_string(),
                                    "Error getting GPU utilization information"
                                )
                            }
                        }
                    },
                    Err(e) => {
                        tracing::warn!(
                            error=e.to_string(),
                            "Error getting GPU utilization information"
                        )
                    }
                };

            }
        });

        Ok(Self { 
            client,
            triton_url,
            model_name,
            model_version,
            input_name,
            input_shape,
            output_name,
            output_shape,
            precision,
            nms_iou_threshold,
            base_request,
            stats_handle
        })
    }

    /// Unloads running instances of a given model
    pub async fn unload_model(&self) -> Result<()> {
        // Unload previous instances of model we're about to load
        self.client.repository_model_unload(RepositoryModelUnloadRequest { 
            repository_name: "".to_string(), 
            model_name: self.model_name.clone(), 
            parameters: HashMap::new()
        })
            .await
            .context("Error unloading previous triton model instances")?;

        Ok(())
    }

    /// Loads given amount of instances of a given model
    pub async fn load_model(&self, instances: usize) -> Result<()> {
        let model_config = json!({
            "name": self.model_name,
            "platform": "tensorrt_plan",
            "max_batch_size": 16,
            "input": [
                {
                    "name": &self.input_name,
                    "data_type": format!("TYPE_{}", self.precision.to_string()),
                    "dims": &self.input_shape
                }
            ],
            "output": [
                {
                    "name": self.output_name,
                    "data_type": format!("TYPE_{}", self.precision.to_string()),
                    "dims": &self.output_shape
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
                "max_queue_delay_microseconds": 500,
                "preferred_batch_size": [2, 4, 8, 12, 16]
            },
            "optimization": {
                "execution_accelerators": {
                "gpu_execution_accelerator": [
                    {
                        "name": "tensorrt",
                        "parameters": {
                            "key": "precision_mode",
                            "value": self.precision.to_string()
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
                    "batch_size": 10,
                    "inputs":  {
                        &self.input_name: {
                            "dims": &self.input_shape,
                            "data_type": format!("TYPE_{}", self.precision.to_string()),
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
            model_name: self.model_name.clone(), 
            parameters: parameters
        })
            .await
            .context("Error loading triton model instances")?;

        Ok(())
    }

    /// Performs inference on a raw image, returning raw model results
    pub async fn infer(&self, image: Vec<u8>) -> Result<Vec<u8>> {
        // Create new inference request
        let mut inference_request = self.base_request.clone();
        inference_request.raw_input_contents.push(image);

        // Perform inference
        let inference_result = self.client.model_infer(inference_request)
            .await
            .context("Error sending triton inference request")?;

        // Return inference results
        Ok(
            inference_result.raw_output_contents
                .into_iter()
                .next()
                .context("Error getting inference results for image")?
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
}

impl InferenceModel {
    pub fn client(&self) -> &Client {
        &self.client
    }

    pub fn triton_url(&self) -> &str {
        &self.triton_url
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn model_version(&self) -> &str {
        &self.model_version
    }

    pub fn input_name(&self) -> &str {
        &self.input_name
    }

    pub fn input_shape(&self) -> &[i64; 3] {
        &self.input_shape
    }

    pub fn output_name(&self) -> &str {
        &self.output_name
    }

    pub fn output_shape(&self) -> &[i64; 2] {
        &self.output_shape
    }

    pub fn precision(&self) -> InferencePrecision {
        self.precision
    }

    pub fn nms_iou_threshold(&self) -> f32 {
        self.nms_iou_threshold
    }

    pub fn base_request(&self) -> &ModelInferRequest {
        &self.base_request
    }
}

impl Drop for InferenceModel {
    fn drop(&mut self) {
        // Abort tokio task
        self.stats_handle.abort();
    }
}