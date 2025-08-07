use std::io::{Error, ErrorKind};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

// Custom modules
use client::config::{Config, Environment};
use client::inference::{self, InferenceModel};
use client::source::{self, SourceProcessor};
use client::utils;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    //Iniaitlize config
    let app_config = Config::new(true, Environment::Production)
        .expect("Error loading config");

    //Initiate inference client
    inference::init_inference_model(&app_config);

    // Initiate sources processors
    source::init_source_processors(&app_config);

    // Perform test inference
    let interval = Duration::from_millis(1000);
    let (image, image_height, image_width) = utils::get_image_raw("/mnt/disk_d/Programming/real-time-object-detection/client-triton/giraffes.jpg")
        .expect("Error loading sample image");
    let processor1 = source::get_source_processor("2025");
    let processor2 = source::get_source_processor("3033");
    
    loop {
        let start = Instant::now();

        // Trigger frame processing
        processor1.process_frame(&image.clone(), image_height, image_width);
        processor2.process_frame(&image.clone(), image_height, image_width);
        
        // Calculate how long to sleep to maintain 34ms intervals
        let elapsed = start.elapsed();
        if elapsed < interval {
            tokio::time::sleep(interval - elapsed).await;
        }
    }

    Ok(())
}
