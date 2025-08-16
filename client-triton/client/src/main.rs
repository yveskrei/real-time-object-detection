use tokio::time::{Duration, Instant};
use anyhow::{Result, Context};

// Custom modules
use client::utils::config::{AppConfig, Environment};
use client::inference::{self, source};
use client::utils;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    //Iniaitlize config
    let mut app_config = AppConfig::new(true, Environment::Production)
        .context("Error loading config")?;

    //Initiate inference client
    inference::init_inference_model(&app_config)
        .await
        .context("Error initiating inference model")?;
    
    // Perform test inference
    let (image, image_height, image_width) = utils::get_image_raw(app_config.source_image_test())
        .context("Error loading test image")?;
    
    // Test variables
    let max_streams = 10; // Adjust this to your desired maximum
    let test_duration = Duration::from_secs(15); // 60 second test
    let frame_interval = Duration::from_millis(34); // 30fps = ~33.33ms, using 34ms
    let stagger_delay = Duration::from_millis(2);   // 2ms delay between processors

    for stream_count in 1..=max_streams {
        tracing::info!("\n=== Starting performance test with {} streams ===", stream_count);
        
        // Create source IDs for current test
        let source_ids: Vec<String> = (1..=stream_count)
            .map(|i| format!("test_source_{}", i))
            .collect();

        // Set new source ids
        app_config.set_source_ids(source_ids);

        //Initiate inference client
        inference::start_model_instances(app_config.source_ids().len())
            .await
            .context("Error initiating inference model instances")?;

        // Initiate sources processors
        source::init_source_processors(&app_config)
            .await
            .context("Error initiating source processors")?;

        // Initialize all processors
        let mut processors = Vec::new();
        for source_id in app_config.source_ids() {
            let processor = source::get_source_processor(source_id).await?;
            processors.push(processor);
        }

        tracing::info!("Initialized {} processors", processors.len());

        // Start performance test
        let test_start = Instant::now();
        
        while test_start.elapsed() < test_duration {
            let frame_start = Instant::now();
            
            // Process each processor with staggered timing
            for processor in &processors {
                processor.process_frame(&image.clone(), image_height, image_width);
                tokio::time::sleep(stagger_delay).await;
            }
            
            // Calculate remaining time to maintain required frame interval
            let elapsed = frame_start.elapsed();
            if elapsed < frame_interval {
                let remaining = frame_interval - elapsed;
                tokio::time::sleep(remaining).await;
            }
        }
    }

    Ok(())
}
