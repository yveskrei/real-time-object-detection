use std::collections::HashMap;
use anyhow::{Context, Result};
use elasticsearch::{Elasticsearch, http::transport::Transport};
use serde_json::json;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

// Custom modules
use crate::utils::config::{ElasticConfig, SearchType, SearchConfigOption};
use crate::processing::ResultEmbedding;

pub struct Elastic {
    client: Elasticsearch,
    elastic_config: ElasticConfig,
    search_config: HashMap<SearchType, SearchConfigOption>
}

#[derive(Debug, Deserialize, Serialize, Clone, ToSchema)]
pub struct SearchMetadata {
    /// Channel ids to restrict the search to. `u32` to match the `source_id` the
    /// ingest side writes into `channel_id` - the field is mapped `keyword`, and
    /// Elasticsearch coerces numeric terms to their string form, so a numeric term
    /// matches a document indexed from a numeric source.
    pub channel_ids: Option<Vec<u32>>,
    pub timestamp_start: Option<i64>,
    pub timestamp_end: Option<i64>
}

impl Elastic {
    /// Creates a new Elastic client instance
    pub fn new(
        elastic_config: ElasticConfig, 
        search_config: HashMap<SearchType, SearchConfigOption>
    ) -> Result<Self> {
        let transport = Transport::single_node(&elastic_config.url)
            .context("Failed to create Elastic transport")?;
        
        let client = Elasticsearch::new(transport);

        Ok(Self {
            client,
            elastic_config,
            search_config
        })
    }

    /// Performs a KNN search on the index with the given embedding and configuration
    pub async fn search_disk_bbq(
        &self,
        embedding: ResultEmbedding,
        search_type: SearchType,
        metadata: SearchMetadata
    ) -> Result<Vec<serde_json::Value>> {
        // Determine search configuration
        let search_config = self.search_config.get(&search_type)
            .context("Search configuration not found")?;

        let mut must_clauses = Vec::new();

        // Filter by channel IDs
        if let Some(ids) = metadata.channel_ids {
            if !ids.is_empty() {
                must_clauses.push(json!({
                    "terms": {
                        "channel_id": ids
                    }
                }));
            }
        }

        // Filter by timestamps (start and end)
        if metadata.timestamp_start.is_some() || metadata.timestamp_end.is_some() {
            let mut range_query = json!({});
            
            if let Some(start) = metadata.timestamp_start {
                range_query["gte"] = json!(start);
            }
            
            if let Some(end) = metadata.timestamp_end {
                range_query["lte"] = json!(end);
            }

            must_clauses.push(json!({
                "range": {
                    "timestamp": range_query
                }
            }));
        }

        let filter = if !must_clauses.is_empty() {
            Some(json!({
                "bool": {
                    "must": must_clauses
                }
            }))
        } else {
            None
        };

        let mut knn_query = json!({
            "field": "embedding",
            "query_vector": embedding.data,
            "k": search_config.output_vectors,
            "num_candidates": search_config.num_candidates,
            "visit_percentage": search_config.centriod_visit_percentage,
            "rescore_vector": {
                "oversample": search_config.vector_oversample_multiplier
            }
        });

        if let Some(f) = filter {
            knn_query["filter"] = f;
        }

        let body = json!({
            "timeout": "25s",
            "size": search_config.output_vectors,
            "knn": knn_query,
            "_source": {
                "excludes": ["embedding"]
            }
        });

        let response = self.client
            .search(elasticsearch::SearchParts::Index(&[&self.elastic_config.index_name]))
            .body(body)
            .send()
            .await
            .context("Failed to execute search request")?;

        let response_body = response.json::<serde_json::Value>().await
            .context("Failed to parse search response")?;

        // Return search results
        let hits = response_body["hits"]["hits"]
            .as_array()
            .map(|h| h.to_vec())
            .unwrap_or_default();

        Ok(hits)
    }
}
