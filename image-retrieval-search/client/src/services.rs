use anyhow::{Result, Context};
use std::sync::Arc;
use tokio::sync::{OnceCell, RwLock};

// Custom modules
use crate::utils::config::AppConfig;
use crate::inference::InferenceModels;
use crate::statistics::Statistics;
use crate::utils::{
    elastic::Elastic,
    redis::Redis
};

// Variables
pub static SERVICES: OnceCell<Arc<Services>> = OnceCell::const_new();

/// Returns all services
pub fn get_services() -> Result<Arc<Services>> {
    Ok(
        SERVICES
            .get()
            .cloned()
            .context("Services are not initiated!")?
    )
}

/// Initiates all services required by the application
pub async fn init_services(app_config: &AppConfig) -> Result<()> {
    if let Some(_) = SERVICES.get() {
        anyhow::bail!("Services are already initiated!")
    }

    // Initiate services
    let services = Services::new(app_config).await
        .context("Error initiating Services")?;

    // Set global variable before start, so started tasks can read it
    SERVICES.set(Arc::new(services))
        .map_err(|_| anyhow::anyhow!("Error setting services"))?;

    Ok(())
}

pub struct Services {
    elastic: Elastic,
    redis: Redis,
    statistics: RwLock<Statistics>,
    inference_models: InferenceModels,
}

impl Services {
    pub async fn new(app_config: &AppConfig) -> Result<Self> {
        // Initiate Elastic
        let elastic = Elastic::new(
            app_config.elastic_config().clone(),
            app_config.search_config().clone()
        )
            .context("Error initiating Elastic")?;

        // Initiate Redis
        let redis = Redis::new(app_config.redis_config().clone()).await
            .context("Error initiating Redis")?;
        
        // Initiate statistics
        let statistics = Statistics::new(app_config.inference_config().device_type)
            .context("Error intiating statistics")?;

        // Initiate inference models
        let inference_models = InferenceModels::new(&app_config).await
            .context("Error initiating inference models")?;

        Ok(
            Self {
                elastic,
                redis,
                statistics: RwLock::new(statistics),
                inference_models,
            }
        )
    }

    pub async fn start(&self, app_config: &AppConfig) -> Result<()> {
        // Start statistics
        self.statistics
            .write()
            .await
            .start()
            .context("Error starting statistics")?;

        // Start models
        self.inference_models
            .start(app_config)
            .await
            .context("Error starting inference models")?;

        Ok(())
    }
}

impl Services {
    pub fn elastic(&self) -> &Elastic {
        &self.elastic
    }

    pub fn redis(&self) -> &Redis {
        &self.redis
    }

    pub fn statistics(&self) -> &RwLock<Statistics> {
        &self.statistics
    }

    pub fn inference_models(&self) -> &InferenceModels {
        &self.inference_models
    }
}