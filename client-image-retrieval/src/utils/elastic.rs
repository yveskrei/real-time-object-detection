use anyhow::{Context, Result};
use elasticsearch::{
    BulkParts, Elasticsearch, http::request::JsonBody, http::transport::Transport,
};
use serde_json::{Value, json};
use std::sync::Arc;

// Custom modules
use crate::processing::ResultEmbedding;
use crate::utils::config::ElasticConfig;
use crate::utils::queue::FixedSizeQueue;

// Constants
const MAX_QUEUE_SIZE: usize = 1000;
const FLUSH_INTERVAL: tokio::time::Duration = tokio::time::Duration::from_secs(2);

pub struct Elastic {
    queue: Arc<FixedSizeQueue<Value>>,
}

impl Elastic {
    /// Creates a new Elastic client instance
    pub fn new(config: ElasticConfig) -> Result<Self> {
        let transport =
            Transport::single_node(&config.url).context("Failed to create Elastic transport")?;

        let client = Elasticsearch::new(transport);

        // Create queue wrapped in Arc
        let queue = Arc::new(FixedSizeQueue::new(MAX_QUEUE_SIZE, None::<fn(Value)>));

        // Clone queue Arc for the background task
        let queue_clone = Arc::clone(&queue);
        let config_clone = config.clone();

        // Spawn background task for flushing
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(FLUSH_INTERVAL);

            loop {
                interval.tick().await;

                // Access receiver through the Arc
                let batch = queue_clone.receiver.drain().await;

                if !batch.is_empty() {
                    if let Err(e) = Self::send_bulk(&client, &config_clone, batch).await {
                        tracing::error!("Failed to send bulk request to Elastic: {:?}", e);
                    }
                }
            }
        });

        Ok(Self { queue })
    }

    /// Adds embeddings to the queue
    pub async fn populate_embeddings(
        &self,
        source_id: u32,
        timestamp: i64,
        embeddings: &[ResultEmbedding],
    ) -> Result<()> {
        for embedding in embeddings {
            // Prepare document
            let doc = json!({
                "timestamp": timestamp,
                "channel_id": source_id,
                "embedding": embedding.data
            });

            // Send to queue
            self.queue.sender.send_async(doc).await;
        }

        Ok(())
    }

    /// Sends a bulk request to Elasticsearch
    async fn send_bulk(
        client: &Elasticsearch,
        config: &ElasticConfig,
        batch: Vec<Value>,
    ) -> Result<()> {
        let mut body: Vec<JsonBody<Value>> = Vec::with_capacity(batch.len() * 2);
        let total_embeddings = batch.len();

        for doc in batch {
            // Index action
            body.push(json!({"index": { "_index": config.index_name }}).into());
            // Document source
            body.push(doc.into());
        }

        let response = client
            .bulk(BulkParts::None)
            .body(body)
            .send()
            .await
            .context("Failed to send bulk request to Elastic")?;

        // Validate request response
        let response_body: Value = response.json().await?;
        if response_body["errors"].as_bool().unwrap_or(false) {
            // Log the actual error message from the first failed item
            tracing::error!("Bulk write contained errors: {:?}", response_body);
        } else {
            tracing::info!(
                "Successfully sent bulk request to Elastic, Total {}",
                total_embeddings
            );
        }

        Ok(())
    }
}
