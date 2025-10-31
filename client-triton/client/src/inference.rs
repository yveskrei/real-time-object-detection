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
use std::sync::Arc;
use tokio::sync::OnceCell;
use anyhow::{self, Result, Context};
use std::time::{Duration, Instant};

// Custom modules
pub mod source;
pub mod queue;
use crate::utils::{
    self,
    GPUStats,
    config::{AppConfig, ModelConfig, TritonConfig}
};
use crate::utils::config::{InferenceModelType, InferencePrecision};

// Variables
pub static INFERENCE_MODELS: OnceCell<HashMap<InferenceModelType, Arc<InferenceModel>>> = OnceCell::const_new();
pub static GPU_STATS_INTERVAL: Duration = Duration::from_secs(200);

/// Returns the inference model instance, if initiated
pub fn get_inference_model(model_type: InferenceModelType) -> Result<&'static Arc<InferenceModel>> {
    Ok(
        INFERENCE_MODELS
            .get()
            .context("Infernece models are not initiated!")?
            .get(&model_type)
            .context("Infernece model is not initiated!")?
    )
}

/// Initiates a single instance of a model for inference
pub async fn init_inference_models(app_config: &AppConfig) -> Result<()> {
    if let Some(_) = INFERENCE_MODELS.get() {
        anyhow::bail!("Models are already initiated!")
    }

    // Create model instances
    let mut models: HashMap<InferenceModelType, Arc<InferenceModel>> = HashMap::new();
    for (model_type, model_config) in app_config.inference_config().models.iter() {
        // Create single instance
        let client_instance = InferenceModel::new(
            app_config.triton_config().clone(),
            model_config.clone(),
        )
            .await
            .context("Error creating model client")?;

        models.insert(
            model_type.clone(),
            Arc::new(client_instance)
        );
    }

    // Set global variable
    INFERENCE_MODELS.set(models)
        .map_err(|_| anyhow::anyhow!("Error setting model instances"))?;

    Ok(())
}

pub async fn start_models_instances(app_config: &AppConfig) -> Result<()> {
    // Calculate total "load units" - how much processing capacity we need
    // Each source contributes fractional load based on its frame rate
    let instances: u32 = app_config
        .sources_config()
        .sources
        .len() as u32;

    // Load same amount of instances for each model type
    for model_type in app_config.inference_config().models.keys() {
        let client_instance = get_inference_model(model_type.clone())?;

        // Clear previous model instances
        if let Ok(_) = client_instance.unload_model().await {
            tracing::warn!("Unloaded previous model instances for type {}", model_type.to_string());
        }

        // Initiate model instances
        client_instance.load_model(instances).await
            .context("Error loading model instances")?;

        tracing::info!("Initiated {} model instances for type {}", instances, model_type.to_string());
    }

    Ok(())
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
            model_name: model_config.name.to_string(),
            model_version: "1".to_string(),
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

        let stats_handle = tokio::task::spawn_blocking(move || {
            loop {
                let measure_time = Instant::now();

                // Get GPU statistics
                let stats_result = utils::get_gpu_statistics();

                match stats_result {
                    Ok(stats) => {
                        InferenceModel::process_gpu_stats(stats);
                    },
                    Err(e) => {
                        tracing::warn!(
                            error=e.to_string(),
                            "Error getting GPU utilization information"
                        )
                    }
                };

                // Sleep if time remains
                let remainder = measure_time.elapsed();
                if remainder < stats_interval {
                    let duration = stats_interval - remainder;
                    std::thread::sleep(duration);
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
            model_name: self.model_config().name.to_string(), 
            parameters: HashMap::new()
        })
            .await
            .context("Error unloading previous triton model instances")?;

        Ok(())
    }
    
    /// Loads given amount of instances of a given model
    pub async fn load_model(&self, instances: u32) -> Result<()> {
        let model_config = json!({
            "name": &self.model_config().name,
            "platform": "tensorrt_plan",
            "max_batch_size": &self.model_config().batch_max_size,
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
                "max_queue_delay_microseconds": self.model_config().batch_max_queue_delay,
                "preferred_batch_size": &self.model_config().batch_preferred_sizes,
                "preserve_ordering": false
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
                    "batch_size": self.model_config().batch_max_size,
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
            model_name: self.model_config().name.to_string(), 
            parameters: parameters
        })
            .await
            .context("Error loading triton model instances")?;

        Ok(())
    }

    /// Performs inference on a single raw input, returning raw model results
    pub async fn infer_single(&self, raw_input: Vec<u8>) -> Result<Vec<u8>> {        
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
                .context("Error getting inference result for input")?
        )
    }

    /// Performs inference on many raw inputs, returning raw model results
    /// Automatically batches requests up to max_batch_size
    pub async fn infer_many(&self, raw_inputs: Vec<Vec<u8>>) -> Result<Vec<Vec<u8>>> {
        let max_batch_size = self.model_config.batch_max_size as usize;
        let mut all_results = Vec::with_capacity(raw_inputs.len());
        
        // Calculate output size per sample
        let output_size_per_sample: usize = self.model_config.output_shape
            .iter()
            .map(|&dim| dim as usize)
            .product::<usize>() * match self.model_config.precision {
                InferencePrecision::FP16 => 2,
                InferencePrecision::FP32 => 4,
            };
        
        for chunk in raw_inputs.chunks(max_batch_size) {
            let batch_size = chunk.len();
            
            // Concatenate batch into single blob
            let total_bytes = chunk.iter().map(|v| v.len()).sum();
            let mut concatenated = Vec::with_capacity(total_bytes);
            for input in chunk {
                concatenated.extend_from_slice(input);
            }
            
            // Create inference request
            let mut inference_request = self.base_request.clone();
            inference_request.inputs[0].shape = {
                let mut shape = vec![batch_size as i64];
                shape.extend(&self.model_config.input_shape);
                shape
            };
            inference_request.raw_input_contents = vec![concatenated];

            // Perform inference
            let inference_result = self.client.model_infer(inference_request)
                .await
                .context("Error sending triton inference request")?;
            
            // Split concatenated output back into individual results
            let output_blob = &inference_result.raw_output_contents[0];
            for i in 0..batch_size {
                let start = i * output_size_per_sample;
                let end = start + output_size_per_sample;
                all_results.push(output_blob[start..end].to_vec());
            }
        }
        
        Ok(all_results)
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