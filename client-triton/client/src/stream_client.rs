// Custom modules
use crate::inference::{InferenceResult};

pub fn populate_results(source_id: &str, bboxes: &[InferenceResult]) {
    tracing::info!(
        source_id=source_id,
        bboxes=bboxes.len(),
        "Got bboxes to populate!"
    )
}