use anyhow::{Result, Context};

// Custom modules
use client::inference::{self, source};
use client::utils::{
    kafka,
    config::AppConfig
};
use client::client_video::ClientVideo;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    // Iniaitlize config
    let app_config = AppConfig::new()
        .context("Error loading config")?;

    client::init_tokio_runtime(tokio::runtime::Handle::current())
        .await
        .context("Error initializing tokio runtime")?;

    // // Initiate Kafka producer
    // kafka::init_kafka_producer(&app_config)
    //     .await
    //     .context("Error initiating Kafka producer")?;

    // Initiate inference client
    inference::init_inference_models(&app_config)
        .await
        .context("Error initiating inference model")?;

    inference::start_models_instances(&app_config)
        .await
        .context("Error initiating inference model instances")?;

    // Initiate sources processors
    source::init_source_processors(&app_config)
        .await
        .context("Error initiating source processors")?;

    // Start receiving frames from sources
    ClientVideo::set_callbacks()
        .await
        .context("Error setting Client Video callbacks")?;

    ClientVideo::init_sources(&app_config)
        .await
        .context("Error setting Client Video callbacks")?;

    Ok(())
}
