use anyhow::{Context, Result};
use bincode::{Decode, Encode, config};
use redis::AsyncCommands;
use std::sync::Arc;
use uuid::Uuid;

// Custom modules
use crate::processing;
use crate::processing::{RawFrame, ResultEmbedding};
use crate::services;
use crate::utils::config::{ModelPurpose, SearchType};
use crate::utils::elastic::SearchMetadata;

#[derive(Encode, Decode)]
pub struct ImageMetadata {
    pub embedding: ResultEmbedding,
    pub model_type: ModelPurpose,
}

pub async fn upload_image(image: RawFrame, model_type: ModelPurpose) -> Result<String> {
    let services = services::get_services()?;

    // Get image embedding
    let inference_model = services.inference_models().model(model_type.clone())?;
    let (inference_stats, embedding) =
        processing::dino::process_frame(&inference_model, Arc::new(image)).await?;

    // Assign new image
    let image_id = Uuid::new_v4();
    let image_metadata = ImageMetadata {
        embedding,
        model_type,
    };

    // Upload image to redis
    let metadata_bytes: Vec<u8> = bincode::encode_to_vec(&image_metadata, config::standard())
        .context("Error converting image metadata to bytes")?;

    services
        .redis()
        .connection()
        .set_ex::<_, _, ()>(
            format!("retrieval_{}", image_id.to_string()),
            metadata_bytes,
            120, // seconds
        )
        .await
        .context("Error uploading image to redis")?;

    // Add to global statistics
    services
        .statistics()
        .read()
        .await
        .processing_stats()
        .accumulate(&inference_stats);

    Ok(image_id.to_string())
}

pub async fn search_image(
    image_id: String,
    search_type: SearchType,
    search_metadata: SearchMetadata,
) -> Result<Vec<serde_json::Value>> {
    let services = services::get_services()?;

    // Get image data from redis
    let metadata_bytes: Option<Vec<u8>> = services
        .redis()
        .connection()
        .get(format!("retrieval_{}", &image_id))
        .await
        .context("Error retrieving image!")?;
    let metadata_bytes = metadata_bytes.context("Image is not found!")?;

    if metadata_bytes.is_empty() {
        anyhow::bail!("Error reading image metadata, empty!")
    }

    let (image_metadata, _): (ImageMetadata, usize) =
        bincode::decode_from_slice(&metadata_bytes, config::standard())
            .context("Error reading image metadata bytes")?;

    // Search for similar images
    let search_results = services
        .elastic()
        .search_disk_bbq(image_metadata.embedding, search_type, search_metadata)
        .await?;

    Ok(search_results)
}
