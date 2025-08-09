//! Responsible for connecting to remote S3 storage

use aws_config::{Region};
use aws_sdk_s3::{Client, Config};
use aws_sdk_s3::config::{Credentials, SharedCredentialsProvider};
use anyhow::{Result, Context};
use std::path::Path;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

/// Represents an instance of S3 client connection
pub struct S3Client {
    _access_key: String,
    _secret_key: String,
    _endpoint: String,
    _region: String,
    client: Client
}

impl S3Client {
    pub fn new(
        access_key: String,
        secret_key: String,
        endpoint: String,
        region: String,
    ) -> Self {
        // Create connection credentials
        let creds = Credentials::new(
            access_key.to_string(),
            secret_key.to_string(),
            None,
            None,
            "",
        );

        // Build the S3 config
        let config = Config::builder()
            .endpoint_url(endpoint.to_string())
            .region(Region::new(region.to_string()))
            .credentials_provider(SharedCredentialsProvider::new(creds))
            .force_path_style(true)
            .build();

        // Create the S3 client
        let client = Client::from_conf(config);

        Self {
            _access_key: access_key,
            _secret_key: secret_key,
            _endpoint: endpoint,
            _region: region,
            client
        }
    }

    /// Downloads a given file from S3 storage and saves it in a local path
    pub async fn download_s3_file(
        &self,
        bucket: &str,
        key: &str,
        local_path: &str,
    ) -> Result<()> {
        // Get the object from S3
        let response = self.client
            .get_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .context("Error getting file from S3 path")?;

        // Create the local directory if it doesn't exist
        if let Some(parent) = Path::new(local_path).parent() {
            tokio::fs::create_dir_all(parent).await
                .context("Error creating parents to save path")?;
        }

        // Create the local file
        let mut file = File::create(local_path).await
            .context("Error creating local file path")?;

        // Stream the S3 object body to the local file
        let mut stream = response.body;
        
        while let Some(bytes) = stream.try_next().await
            .context("Error getting S3 file bytes")? {
            file.write_all(&bytes).await
                .context("Error writing bytes to local file")?;
        }

        // Ensure all data is written to disk
        file.flush().await
            .context("Could not write local file data")?;

        Ok(())
    }
}