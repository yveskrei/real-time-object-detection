use axum::extract::Multipart;
use utoipa::ToSchema;
use serde_json::json;
use axum::body::Bytes;
use anyhow::{Result, Context};

// Custom modules
use crate::processing::{self, RawFrame};
use crate::handlers::api::ApiResponse;
use crate::handlers::api::ApiResult;
use crate::utils::config::ModelPurpose;
use crate::handlers::TAG_RETRIEVAL;
use crate::utils;

/// Represents input data for image upload
#[derive(ToSchema)]
pub struct ImageUploadRequest {
    #[schema(value_type = String, format = Binary)]
    pub image: Bytes,
    pub model_type: ModelPurpose
}

pub struct ImageUpload {
    pub image: RawFrame,
    pub model_type: ModelPurpose
}

impl ImageUpload {
    async fn from_multipart(mut multipart: Multipart) -> Result<Self> {
        let mut image: Option<Bytes> = None;
        let mut model_type: Option<ModelPurpose> = None;

        while let Ok(Some(field)) = multipart.next_field().await {
            if let Some(name) = field.name() {
                if name == "image" {
                    let bytes = field.bytes().await
                        .context("Failed to read bytes for 'image' field")?;

                    image = Some(bytes);
                } else if name == "model_type" {
                    let text = field.text().await
                        .context("Failed to read text for 'model_type' field")?;
                
                    let mt = text.parse::<ModelPurpose>()
                        .context("Invalid model type provided:")?;
                
                    model_type = Some(mt);
                }
            }
        }

        let image = image.context("Missing required 'image' field")?;
        let model_type = model_type.context("Missing required 'model_type' field")?;

        // Parse given image
        let (rgb_bytes, width, height) = utils::parse_image(&image)
            .context("Error parsing input image")?;
        let image = RawFrame {
            data: rgb_bytes,
            height,
            width
        };

        Ok(Self { 
            image, 
            model_type
        })
    }
}

/// Upload image to perform search on
#[utoipa::path(
    post,
    path = "/upload",
    tag = TAG_RETRIEVAL,
    request_body(content = ImageUploadRequest, content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "Upload successful"),
        (status = 400, description = "Invalid input")
    )
)]
pub async fn upload_image(multipart: Multipart) -> ApiResult<serde_json::Value> {
    match ImageUpload::from_multipart(multipart).await {
        Ok(upload_data) => {
            match processing::search::upload_image(
                upload_data.image, 
                upload_data.model_type
            ).await {
                Ok(image_id) => {
                    Ok(ApiResponse::success_with_message(
                        "Image uploaded successfully",
                        json!({
                            "image_id": image_id
                        })
                    ))
                }
                Err(e) => {
                    tracing::error!(
                        error=%e,
                        "Error uploading image"
                    );

                    Ok(ApiResponse::bad_request("Could not upload image"))
                }
            }
        },
        Err(_) => {
            Ok(ApiResponse::bad_request("Could not process input"))
        }
    }
}
