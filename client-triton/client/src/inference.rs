use triton_client::Client;
use triton_client::inference::{ModelInferRequest, RepositoryModelLoadRequest, ModelRepositoryParameter};
use triton_client::inference::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};
use triton_client::inference::model_repository_parameter::{ParameterChoice};
use std::collections::HashMap;
use std::io::{Error, ErrorKind};
use serde_json::json;
use std::str::FromStr;
use std::sync::{Arc, OnceLock};

// Custom modules
use crate::config::Config;

// Static inference model
pub static INFERENCE_MODEL: OnceLock<Arc<InferenceModel>> = OnceLock::new();
pub fn get_inference_model() -> &'static Arc<InferenceModel> {
    INFERENCE_MODEL
        .get()
        .expect("Infernece model is not initiated!")
}
pub async fn init_inference_model(app_config: &Config) -> Result<(), Error> {
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
        .map_err(|e| Error::new(ErrorKind::Other, e))?;

    // Initiate model instances
    client_instance.load_model(app_config.source_ids().len())
        .await
        .map_err(|e| Error::new(ErrorKind::Other, e))?;

    // Set global variable
    INFERENCE_MODEL.set(Arc::new(client_instance))
        .map_err(|_| Error::new(ErrorKind::Other, "Model is already initiated"))?;

    Ok(())
}

#[derive(Clone, Debug)]
pub struct InferenceFrame {
    pub data: Vec<u8>,
    pub height: usize,
    pub width: usize
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
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
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "FP32" => Ok(InferencePrecision::FP32),
            "FP16" => Ok(InferencePrecision::FP16),
            _ => Err(format!("Invalid precision: {}", s)),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct InferenceResult {
    pub bbox: [f32; 4],
    pub class: usize, 
    pub score: f32
}

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
    base_request: ModelInferRequest
}

impl InferenceModel {
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
    ) -> Result<Self, Error> {
        //Create client instance
        let client = Client::new(&triton_url, None).await
        .map_err(|e| Error::new(ErrorKind::ConnectionRefused, e))?;

        // Check if server is ready
        let server_ready = client.server_ready().await
        .map_err(|e| Error::new(ErrorKind::ConnectionRefused, e))?;

        if !server_ready.ready {
            return Err(Error::new(ErrorKind::ConnectionRefused, "Triton server is not ready"));
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
            raw_input_contents: vec![]
        };

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
            base_request
        })
    }

    pub async fn load_model(&self, instances: usize) -> Result<(), Error> {
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
                "max_queue_delay_microseconds": 1000,
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
                }
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
        }).await
            .map_err(|e| Error::new(ErrorKind::Other, e))?;

        Ok(())
    }

    pub async fn infer(&self, image: &[u8]) -> Result<Vec<u8>, Error> {
        // Create new inference request
        let mut inference_request = self.base_request.clone();
        inference_request.raw_input_contents = vec![image.to_vec()];

        // Perform inference
        let inference_result = self.client.model_infer(inference_request).await
            .map_err(|e| Error::new(ErrorKind::Other, e))?;

        // Return inference results
        Ok(
            inference_result.raw_output_contents
                .into_iter()
                .next()
                .expect("Error getting inference results for image")
        )
    }

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