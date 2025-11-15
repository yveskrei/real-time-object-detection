use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use anyhow::{Context, Result};

// Info for the raw video stream (backend provides this in 'udp' field)
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct RawStreamInfo {
    // Made protocol and host optional to match the backend's response.
    pub protocol: Option<String>,
    pub host: Option<String>,
    pub port: u16,
    pub width: u32,
    pub height: u32,
    pub pix_fmt: String, // Note: Backend *says* rgb24 but *streams* yuv420p
    pub fps: f64,
    pub bytes_per_pixel: u16,
    pub frame_size_bytes: u32,
}

// Info for the DASH stream
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct DashInfo {
    pub manifest_url: String,
}

// Response models matching the backend
#[derive(Debug, Deserialize, Serialize)]
pub struct StreamStatus {
    pub video_id: i32,
    pub is_streaming: bool,
    pub stream_start_time_ms: Option<i64>,
    pub pid: Option<i32>,
    pub error: Option<String>,
    pub clients: Option<i32>,
    pub status: Option<String>,
    
    // CHANGED: Renamed 'tcp' to 'udp' to match python backend response
    pub udp: Option<RawStreamInfo>, // This field holds raw stream (UDP) info
    
    pub dash: Option<DashInfo>,
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