use std::io::{Error, ErrorKind};
use std::sync::{Arc, OnceLock};

// Custom modules
use client::config::{Config, Environment};
use client::inference::{InferenceModel};
use client::source::{INFERENCE_MODEL};
use client::source::{PROCESSORS, SourceProcessor};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    //Iniaitlize config
    let app_config = Config::new(true, Environment::Production)
        .expect("Error loading config");

    //Initiate model
    let inference_model = InferenceModel::new(
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

    INFERENCE_MODEL.set(Arc::new(inference_model))
        .map_err(|_| "Inference model is already iniitated")?;

    // Initiate sources processors
    for source_id in app_config.source_ids().iter() {
        let confidence_threshold = app_config.source_confs()
            .get(source_id)
            .expect("Source does not have confidence threshold setting");
        let inference_frame = app_config.source_inf_frames()
            .get(source_id)
            .expect("Source does not have inference frame setting");
        
        // Start processor
        let processor = Arc::new(SourceProcessor::new(
            source_id.clone(), 
            *confidence_threshold, 
            *inference_frame
        ));

        PROCESSORS
            .write()
            .unwrap()
            .insert(source_id.clone(), processor);
    }

    Ok(())
}
