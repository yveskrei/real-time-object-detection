use tokio::time::{Duration, Instant};
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
    let (image, image_height, image_width) = utils::get_image_raw(app_config.source_image_test())
        .context("Error loading test image")?;

    // Initialize all processors
    let mut processors = Vec::new();
    for source_id in app_config.source_ids() {
        let processor = source::get_source_processor(source_id).await?;
        processors.push(processor);
    }

    let frame_interval = Duration::from_millis(34); // 30fps = ~33.33ms, using 34ms
    let stagger_delay = Duration::from_millis(2);   // 2ms delay between processors

    loop {
        let frame_start = Instant::now();
        
        // Process each processor with staggered timing
        for processor in &processors {
            processor.process_frame(&image.clone(), image_height, image_width);
            tokio::time::sleep(stagger_delay).await;
        }
        
        // Calculate remaining time to maintain 30fps
        let elapsed = frame_start.elapsed();
        if elapsed < frame_interval {
            let remaining = frame_interval - elapsed;
            tokio::time::sleep(remaining).await;
        }
    }
}
