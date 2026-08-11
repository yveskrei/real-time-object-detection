use axum::{Router, extract::DefaultBodyLimit};
use std::net::SocketAddr;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use anyhow::{Result, Context};

// Custom modules
use client::utils::config::AppConfig;
use client::handlers;
use client::services;

#[tokio::main]
async fn main() -> Result<()> {
    // Iniaitlize config
    let app_config = AppConfig::new()
        .context("Error loading config")?;
    
    services::init_services(&app_config).await
        .context("Error initiating services")?;

    // Start services through the global Arc
    services::get_services()?.start(&app_config).await
        .context("Error starting services")?;

    // Build API application
    let app = Router::new()
        .merge(handlers::routes())
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024)); // 50MB limit

    // Register port for application
    let addr = SocketAddr::from((
        [127, 0, 0, 1], 
        app_config.port()
    ));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .context("Unable to register dedicated port")?;

    tracing::info!("Server running on http://{}", addr);

    // Start application
    axum::serve(listener, app)
        .await
        .context("Cannot start application")?;

    Ok(())
}