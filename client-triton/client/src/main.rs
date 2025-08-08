use std::time::{Duration, Instant};
use anyhow::{Result, Context};

// Custom modules
use client::utils::config::{AppConfig, Environment};
use client::inference::{self, source};
use client::utils;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    //Iniaitlize config
    let app_config = AppConfig::new(true, Environment::Production)
        .context("Error loading config")?;

    // Download model to local machine
    inference::load_inference_model(&app_config)
        .await
        .context("Error loading inference model locally")?;

    //Initiate inference client
    inference::init_inference_model(&app_config)
        .await
        .context("Error initiating inference model")?;

    // Initiate sources processors
    source::init_source_processors(&app_config)
        .await
        .context("Error initiating source processors")?;

    // Perform test inference
    let (image, image_height, image_width) = utils::get_image_raw("/mnt/disk_e/Programming/real-time-object-detection/client-triton/giraffes.jpg")
        .context("Error loading sample image")?;
    let processor1 = source::get_source_processor("2025")?;
    let processor2 = source::get_source_processor("3033")?;
    let processor3 = source::get_source_processor("2026")?;
    let processor4 = source::get_source_processor("2027")?;
    let processor5 = source::get_source_processor("2028")?;
    
    loop {
        // Trigger frame processing
        processor1.process_frame(&image.clone(), image_height, image_width);
        tokio::time::sleep(Duration::from_millis(2)).await;
        processor2.process_frame(&image.clone(), image_height, image_width);
        tokio::time::sleep(Duration::from_millis(2)).await;
        processor3.process_frame(&image.clone(), image_height, image_width);
        tokio::time::sleep(Duration::from_millis(2)).await;
        processor4.process_frame(&image.clone(), image_height, image_width);
        tokio::time::sleep(Duration::from_millis(2)).await;
        processor5.process_frame(&image.clone(), image_height, image_width);
        tokio::time::sleep(Duration::from_millis(2)).await;
    }

    Ok(())
}
