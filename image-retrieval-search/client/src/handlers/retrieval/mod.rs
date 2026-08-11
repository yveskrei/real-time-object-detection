use axum::{Router, routing};

// Custom modules
pub mod post;
pub mod get;

// Route functions
use crate::handlers::retrieval::post::upload_image;
use crate::handlers::retrieval::get::search_image;

pub fn routes() -> Router {
    Router::new()
        .route("/upload", routing::post(upload_image))
        .route("/search", routing::get(search_image))
}
