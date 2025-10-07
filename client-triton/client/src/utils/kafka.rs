use rdkafka::config::ClientConfig;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::util::Timeout;
use std::time::Duration;
use anyhow::{Context, Result};
use tokio::sync::OnceCell;
use std::sync::Arc;

// Custom modules
use crate::utils::config::{KafkaConfig, AppConfig};

// Variables
pub static KAFKA_PRODUCER: OnceCell<Arc<Kafka>> = OnceCell::const_new();

/// Returns the inference model instance, if initiated
pub fn get_kafka_producer() -> Result<&'static Arc<Kafka>> {
    Ok(
        KAFKA_PRODUCER
            .get()
            .context("Kafka producer is not initiated!")?
    )
}

/// Initiates a single instance of a model for inference
pub async fn init_kafka_producer(app_config: &AppConfig) -> Result<()> {
    if let Ok(_) = get_kafka_producer() {
        anyhow::bail!("Kafka producer already initiated!")
    }

    // Create new instance
    let kafka_instance = Kafka::new(
        app_config.kafka_config().clone()
    )
        .context("Error creating new Kafka producer")?;

    // Set global variable
    KAFKA_PRODUCER.set(Arc::new(kafka_instance))
        .map_err(|_| anyhow::anyhow!("Error setting Kafka producer"))?;

    Ok(())
}

pub struct Kafka {
    config: KafkaConfig,
    producer: FutureProducer
}

impl Kafka {
    /// Creates a new Kafka producer instance
    pub fn new(config: KafkaConfig) -> Result<Self> {
        let producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", &config.brokers)
            .set("message.timeout.ms", "5000")
            .create()
            .context("Failed to create Kafka producer")?;

        Ok(
            Kafka { 
                config,
                producer,
            }
        )
    }

    /// Produces a message to the specified topic
    pub async fn produce(&self, key: Option<&str>, message: &str) -> Result<()> {
        let mut record = FutureRecord::to(&self.config.topic)
            .payload(message);
    
        if let Some(k) = key {
            record = record.key(k);
        }

        self.producer
            .send(record, Timeout::After(Duration::from_secs(5)))
            .await
            .map(|_| ())
            .map_err(|(err, _)| err)
            .context(format!("Failed to produce message to topic '{}'", &self.config.topic))?;

        Ok(())
    }
}