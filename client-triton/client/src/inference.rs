use triton_client::Client;
use triton_client::inference::{ModelInferRequest, InferTensorContents, RepositoryModelLoadRequest, ModelRepositoryParameter};
use triton_client::inference::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};
use triton_client::inference::model_repository_parameter::{ParameterChoice};
use std::collections::HashMap;

use std::io::{Error, ErrorKind};
use half::f16;
use serde_json::json;

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum Precision {
    FP32,
    FP16
}

pub struct InferenceModel {
    triton: Client,
    model_name: String,
    model_version: String,
    input_name: String,
    output_name: String,
    precision: Precision
}

impl InferenceModel {
    pub async fn new(
        model_name: String, 
        model_version: String, 
        input_name: String, 
        output_name: String, 
        precision: Precision
    ) -> Result<Self, Error> {
        //Create client instance
        let triton = Client::new("http://localhost:8001/", None).await
        .map_err(|e| Error::new(ErrorKind::ConnectionRefused, e))?;

        // Check if server is ready
        let server_ready = triton.server_ready().await
        .map_err(|e| Error::new(ErrorKind::ConnectionRefused, e))?;

        if !server_ready.ready {
            return Err(Error::new(ErrorKind::ConnectionRefused, "Triton server is not ready"));
        }

        println!("Triton Server is ready!");

        Ok(Self { 
            triton,
            model_name,
            model_version,
            input_name,
            output_name,
            precision
        })
    }

    pub async fn load_model(&self, instances: i32) -> Result<(), Error> {
        let precision_name = match self.precision {
            Precision::FP32 => "FP32",
            Precision::FP16 => "FP16"
        };

        let model_config = json!({
            "name": self.model_name,
            "platform": "tensorrt_plan",
            "max_batch_size": 16,
            "input": [
                {
                    "name": self.input_name,
                    "data_type": format!("TYPE_{}", precision_name),
                    "dims": [3, 640, 640]
                }
            ],
            "output": [
                {
                    "name": self.output_name,
                    "data_type": format!("TYPE_{}", precision_name),
                    "dims": [84, 8400]
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
                            "value": precision_name
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
            }
        });

        // Define model config
        let mut parameters = HashMap::new();
        parameters.insert("config".to_string(), ModelRepositoryParameter{ 
            parameter_choice: Some(ParameterChoice::StringParam(model_config.to_string()))
        });

        // Load selected model
        self.triton.repository_model_load(RepositoryModelLoadRequest { 
            repository_name: "".to_string(), 
            model_name: self.model_name.clone(), 
            parameters: parameters
        }).await
        .map_err(|e| Error::new(ErrorKind::Other, e))?;

        println!("Model {} is ready for inference!", self.model_name);

        Ok(())
    }

    pub async fn infer(&self, image: &[f32]) -> Result<(), Error> {
        // Define input/output precision
        let precision_name = match self.precision {
            Precision::FP32 => "FP32",
            Precision::FP16 => "FP16"
        };
        let image_bytes = InferenceModel::get_image_bytes(image, self.precision);

        let inference_request = ModelInferRequest {
            model_name: self.model_name.clone(),
            model_version: self.model_version.clone(),
            id: String::new(),
            parameters: HashMap::new(),
            inputs: vec![
                InferInputTensor {
                    name: self.input_name.clone(), // Adjust based on your model
                    datatype: precision_name.to_string(),
                    shape: vec![1, 3, 640, 640],
                    parameters: HashMap::new(),
                    contents: Some(InferTensorContents {
                        bool_contents: vec![],
                        int_contents: vec![],
                        int64_contents: vec![],
                        uint_contents: vec![],
                        uint64_contents: vec![],
                        fp32_contents: vec![], // Always empty when using bytes
                        fp64_contents: vec![],
                        bytes_contents: vec![image_bytes],
                    })
                }
            ],
            outputs: vec![
                InferRequestedOutputTensor {
                    name: self.output_name.clone(), // Adjust based on your model
                    parameters: HashMap::new(),
                }
            ],
            raw_input_contents: vec![]
        };

        // Perform inference
        let response = self.triton.model_infer(inference_request).await
        .map_err(|e| Error::new(ErrorKind::Other, e))?;
        
        println!("Inference successful!");
        println!("Model name: {}", response.model_name);
        println!("Model version: {}", response.model_version);
        
        // Process outputs
        for output in &response.outputs {
            println!("Output tensor: {}", output.name);
            println!("Shape: {:?}", output.shape);
            println!("Datatype: {}", output.datatype);
            
            // Access the output data
            if let Some(contents) = &output.contents {
                println!("FP32 values count: {}", contents.fp32_contents.len());
                println!("First few values: {:?}", 
                    contents.fp32_contents.iter().take(5).collect::<Vec<_>>());
            }
        }

        Ok(())
    }

    fn get_image_bytes(image: &[f32], precision: Precision) -> Vec<u8> {
        match precision {
            Precision::FP32 => {
                let mut bytes = Vec::with_capacity(image.len() * 4);
                for value in image {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                bytes
            },
            Precision::FP16 => {
                let mut bytes = Vec::with_capacity(image.len() * 2);
                for value in image {
                    let fp16_value = f16::from_f32(*value);
                    bytes.extend_from_slice(&fp16_value.to_le_bytes());
                }
                bytes
            }
        }
    }
}