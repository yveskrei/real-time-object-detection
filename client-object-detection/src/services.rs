use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::sync::{OnceCell, RwLock};

// Custom modules
use crate::client_video::{ClientVideo, RunMode};
use crate::inference::InferenceModels;
use crate::source::SourceProcessors;
use crate::statistics::Statistics;
use crate::utils::config::AppConfig;

// Variables
pub static SERVICES: OnceCell<Arc<Services>> = OnceCell::const_new();

/// Returns all services
pub fn get_services() -> Result<Arc<Services>> {
    Ok(SERVICES
        .get()
        .cloned()
        .context("Services are not initiated!")?)
}

/// Initiates all services required by the application
pub async fn init_services(app_config: &AppConfig, runtime: tokio::runtime::Handle) -> Result<()> {
    if let Some(_) = SERVICES.get() {
        anyhow::bail!("Services are already initiated!")
    }

    // Initiate services
    let services = Services::new(app_config, runtime)
        .await
        .context("Error initiating Services")?;

    // Set global variable before start, so started tasks can read it
    SERVICES
        .set(Arc::new(services))
        .map_err(|_| anyhow::anyhow!("Error setting services"))?;

    Ok(())
}

pub struct Services {
    runtime: tokio::runtime::Handle,
    statistics: RwLock<Statistics>,
    inference_models: InferenceModels,
    source_processors: RwLock<SourceProcessors>,
    client_video: ClientVideo,
}

impl Services {
    pub async fn new(app_config: &AppConfig, runtime: tokio::runtime::Handle) -> Result<Self> {
        // Initiate inference model
        let inference_models = InferenceModels::new(app_config)
            .await
            .context("Error initiating inference models")?;

        // Initiate source processors
        let source_processors =
            SourceProcessors::new(app_config).context("Error initiating source processors")?;

        // Initiate statistics
        let statistics = Statistics::new(app_config.inference_config().device_type)
            .context("Error intiating statistics")?;

        // Initiate client video
        let client_video = ClientVideo::new().context("Error creating client video")?;

        Ok(Self {
            runtime,
            statistics: RwLock::new(statistics),
            inference_models,
            source_processors: RwLock::new(source_processors),
            client_video,
        })
    }

    pub async fn start(&self, app_config: &AppConfig) -> Result<()> {
        // Start statistics
        self.statistics
            .write()
            .await
            .start()
            .context("Error starting statistics")?;

        // Start model
        self.inference_models
            .start(app_config)
            .await
            .context("Error starting inference model")?;

        // Boot the video library - callbacks and run mode
        self.client_video
            .init_state(RunMode::Regular)
            .await
            .context("Error initiating client video state")?;

        // Start sources. Frames begin flowing from here.
        self.client_video
            .init_sources(app_config)
            .await
            .context("Error initiating client video sources")?;

        Ok(())
    }
}

impl Services {
    pub fn runtime(&self) -> &tokio::runtime::Handle {
        &self.runtime
    }

    pub fn statistics(&self) -> &RwLock<Statistics> {
        &self.statistics
    }

    pub fn inference_models(&self) -> &InferenceModels {
        &self.inference_models
    }

    pub fn source_processors(&self) -> &RwLock<SourceProcessors> {
        &self.source_processors
    }

    pub fn client_video(&self) -> &ClientVideo {
        &self.client_video
    }
}
