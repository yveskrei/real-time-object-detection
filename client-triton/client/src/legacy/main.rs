// Custom modules
use client::config::{Config, Environment};
use client::inference::{InferenceModel};
use client::utils;
use client::processing;
use std::io::{Error, ErrorKind};
use std::time::Instant;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    //Iniaitlize config
    let app_config = Config::new(true, Environment::Production)
    .expect("Error loading config");

    //Initiate model
    let inference = InferenceModel::new(
        app_config.model_name().to_string(), 
        app_config.model_version().to_string(), 
        app_config.model_input_name().to_string(), 
        app_config.model_input_shape().clone(),
        app_config.model_output_name().to_string(), 
        app_config.model_output_shape().clone(),
        app_config.model_precision()
    )
    .await
    .map_err(|e| Error::new(ErrorKind::Other, e))?;

    // Load the image from a local file
    let (image, image_height, image_width) = utils::get_image_raw("/mnt/disk_e/Programming/real-time-object-detection/client-triton/client/src/giraffes.jpg")
    .expect("Error loading sample image");

    let inference_start = Instant::now();

    // Pre-processed image
    let image_pre_proc = processing::preprocess_yolo(
        &image, 
        image_height, 
        image_width,
        inference.input_shape(),
        inference.precision()
    ).unwrap();

    let pre_processing_time = inference_start.elapsed();

    // Perform inference on image
    let results = inference.infer(&image_pre_proc).await
    .map_err(|e| Error::new(ErrorKind::Other, e))?;

    let inference_time = inference_start.elapsed() - pre_processing_time;

    let bboxes = processing::postprocess_yolo(
        &results, 
        image_height,
        image_width,
        inference.output_shape(),
        inference.precision(),
        app_config.source_conf_default(),
        app_config.nms_iou_thrershold()
    )
    .unwrap();

    let post_processing_time = inference_start.elapsed() - inference_time - pre_processing_time;
    let total_time = inference_start.elapsed();

    tracing::info!(
        pre_processing=pre_processing_time.as_micros(),
        inference=inference_time.as_micros(),
        post_processing=post_processing_time.as_micros(),
        total=total_time.as_micros(),
        "Inference times in microseconds"
    );

    println!("Total detections: {}", bboxes.len());
    println!("Detections: {:?}", bboxes);

    Ok(())
}
