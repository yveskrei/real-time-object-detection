use anyhow::{Context, Result};
use axum::extract::Query;
use serde::Deserialize;
use serde_json::json;
use utoipa::ToSchema;

// Custom modules
use crate::handlers::TAG_RETRIEVAL;
use crate::handlers::api::{ApiResponse, ApiResult};
use crate::processing;
use crate::utils::config::SearchType;
use crate::utils::elastic::SearchMetadata;

#[derive(ToSchema, Deserialize)]
pub struct ImageSearchRequest {
    pub image_id: String,
    pub search_type: SearchType,

    #[schema(nullable, example = json!("1,2"))]
    pub channel_ids: Option<String>,

    #[schema(nullable, example = json!(1767899017))]
    pub timestamp_start: Option<i64>,

    #[schema(nullable, example = json!(1767899017))]
    pub timestamp_end: Option<i64>,
}

/// Parses the comma separated `channel_ids` query parameter into channel ids
fn parse_channel_ids(raw: Option<&str>) -> Result<Option<Vec<u32>>> {
    let Some(raw) = raw else {
        return Ok(None);
    };

    let channel_ids = raw
        .split(',')
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .map(|segment| {
            segment
                .parse::<u32>()
                .with_context(|| format!("Invalid channel id '{}'", segment))
        })
        .collect::<Result<Vec<u32>>>()?;

    Ok((!channel_ids.is_empty()).then_some(channel_ids))
}

/// Search images similar to a given one
#[utoipa::path(
    get,
    path = "/search",
    tag = TAG_RETRIEVAL,
    params(
        ("image_id" = String, Query, description = "ID of the reference image"),
        ("search_type" = SearchType, Query, description = "Type of search"),
        ("channel_ids" = Option<String>, Query, description = "Comma separated channel IDs"),
        ("timestamp_start" = Option<i64>, Query, description = "Start timestamp"),
        ("timestamp_end" = Option<i64>, Query, description = "End timestamp"),
    ),
    responses(
        (status = 200, description = "Search successful"),
        (status = 400, description = "Invalid input")
    )
)]
pub async fn search_image(
    Query(request): Query<ImageSearchRequest>,
) -> ApiResult<serde_json::Value> {
    let channel_ids = match parse_channel_ids(request.channel_ids.as_deref()) {
        Ok(channel_ids) => channel_ids,
        Err(e) => {
            tracing::warn!(
                error=%e,
                "Rejected search with invalid channel_ids"
            );

            return Ok(ApiResponse::bad_request(
                "Invalid channel_ids, expected a comma separated list of whole numbers",
            ));
        }
    };

    // Construct Elastic search parameters
    let metadata = SearchMetadata {
        channel_ids,
        timestamp_start: request.timestamp_start,
        timestamp_end: request.timestamp_end,
    };

    match processing::search::search_image(request.image_id, request.search_type, metadata).await {
        Ok(candidates) => {
            let results: Vec<serde_json::Value> = candidates
                .into_iter()
                .map(|c| {
                    json!({
                        "score": c["_score"],
                        "metadata": c["_source"]
                    })
                })
                .collect();

            Ok(ApiResponse::success_with_message(
                "Image processed successfully",
                json!({
                    "count": results.len(),
                    "candidates": results
                }),
            ))
        }
        Err(e) => {
            tracing::error!(
                error=%e,
                "Could not process search"
            );

            Ok(ApiResponse::bad_request("Error searching candidates"))
        }
    }
}
