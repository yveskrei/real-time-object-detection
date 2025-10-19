use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use anyhow::{Context, Result};

// Response models matching the backend
#[derive(Debug, Deserialize, Serialize)]
pub struct StreamStatus {
    pub video_id: i32,
    pub is_streaming: bool,
    pub stream_url: Option<String>,
    pub stream_start_time_ms: Option<i64>,
    pub vlc_command: Option<String>,
    pub pid: Option<i32>,
    pub error: Option<String>,
}

/// HTTP session for communicating with the player backend
#[derive(Clone)]
pub struct PlayerSession {
    client: Client,
    base_url: String,
}

impl PlayerSession {
    /// Create a new player session from environment variable
    pub fn new() -> Result<Self> {
        let base_url = env::var("PLAYER_BACKEND_URL")
            .context("PLAYER_BACKEND_URL variable is not set")?;
        
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .context("Failed to build HTTP client")?;
        
        Ok(Self { client, base_url })
    }

    /// Get the base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Get stream status for a video
    pub async fn get_stream_status(&self, video_id: i32) -> Result<StreamStatus> {
        let url = format!("{}/streams/status/{}", self.base_url, video_id);
        
        let response = self.client
            .get(&url)
            .send()
            .await
            .context("Failed to send stream status request")?;

        // Check if request was successful
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("Backend returned error {}: {}", status, error_text);
        }

        let status: StreamStatus = response
            .json()
            .await
            .context("Failed to parse stream status response")?;

        Ok(status)
    }
}

impl Default for PlayerSession {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            let client = Client::new();
            Self {
                client,
                base_url: "".to_string(),
            }
        })
    }
}