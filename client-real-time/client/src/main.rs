use anyhow::{Result, Context};

// Custom modules
use client::utils::config::AppConfig;
use client::services;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    // Iniaitlize config
    let app_config = AppConfig::new()
        .context("Error loading config")?;

    // Initiate services
    services::init_services(&app_config, tokio::runtime::Handle::current())
        .await
        .context("Error initiating services")?;

    // Start services through the global Arc
    services::get_services()?.start(&app_config).await
        .context("Error starting services")?;

    // Keep main alive; Ctrl+C (SIGINT) terminates the process via the default signal handler
    tokio::time::sleep(tokio::time::Duration::from_secs(u64::MAX)).await;

    Ok(())
}