use anyhow::{Result, Context};

// Custom modules
use client::utils::config::{AppConfig};
use client::inference::{self, source};

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    //Iniaitlize config
    let app_config = AppConfig::new()
        .context("Error loading config")?;

    //Initiate inference client
    inference::init_inference_model(&app_config)
        .await
        .context("Error initiating inference model")?;

    //Initiate inference client
    inference::start_model_instances(1)
        .await
        .context("Error initiating inference model instances")?;

    // Initiate sources processors
    source::init_source_processors(&app_config)
        .await
        .context("Error initiating source processors")?;

    Ok(())
}