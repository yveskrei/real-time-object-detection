use anyhow::{Result, Context};
use redis::Client;
use redis::aio::MultiplexedConnection;

// Custom modules
use crate::utils::config::RedisConfig;

#[allow(dead_code)]
pub struct Redis {
    config: RedisConfig,
    connection: MultiplexedConnection
}

impl Redis {
    pub async fn new(config: RedisConfig) -> Result<Self> {
        // Create client
        let connection_url = format!(
            "redis://{}:{}@{}",
            &config.username,
            &config.password,
            &config.url
        );
        let client = Client::open(connection_url)
            .context("Error creating Redis client")?;

        // Create connection
        let connection = client.get_multiplexed_async_connection().await
            .context("Error creating Redis connection")?;

        Ok(
            Self {
                config,
                connection
            }
        )
    }
}

impl Redis {
    pub fn connection(&self) -> MultiplexedConnection {
        self.connection.clone()
    }
}