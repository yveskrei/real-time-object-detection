use anyhow::{Result, Context};

// Custom modules
use client::inference::{self, source};
use client::utils::{
    kafka,
    config::AppConfig
};
use client::client_video;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    // Iniaitlize config
    let app_config = AppConfig::new()
        .context("Error loading config")?;

    // Initiate Kafka producer
    kafka::init_kafka_producer(&app_config)
        .await
        .context("Error initiating Kafka producer")?;

    // Initiate inference client
    inference::init_inference_model(&app_config)
        .await
        .context("Error initiating inference model")?;

    //Initiate inference client
    inference::start_model_instances(&app_config)
        .await
        .context("Error initiating inference model instances")?;

    // Initiate sources processors
    source::init_source_processors(&app_config)
        .context("Error initiating source processors")?;

    // Initiate video client
    client_video::init_client_video(&app_config)
        .context("Error initiating video client")?;

    Ok(())
}
