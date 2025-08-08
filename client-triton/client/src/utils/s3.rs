use aws_config::{Region};
use aws_sdk_s3::{Client, Config};
use aws_sdk_s3::config::{Credentials, SharedCredentialsProvider};
use anyhow::{Result, Context};
use std::path::Path;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

pub struct S3Client {
    access_key: String,
    secret_key: String,
    endpoint: String,
    region: String,
    client: Client
}

impl S3Client {
    pub fn new(
        access_key: String,
        secret_key: String,
        endpoint: String,
        region: String,
    ) -> Self {
        // Create credentials
        let creds = Credentials::new(
            access_key.to_string(),
            secret_key.to_string(),
            None, // session token (not needed for MinIO)
            None, // expiration
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
            access_key,
            secret_key,
            endpoint,
            region,
            client
        }
    }

    pub async fn download_s3_file(
        &self,
        bucket: &str,
        key: &str,
        local_path: &str,
    ) -> Result<()> {
        // Get the object from S3
        let resp = self.client
            .get_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await?;

        // Create the local directory if it doesn't exist
        if let Some(parent) = Path::new(local_path).parent() {
            tokio::fs::create_dir_all(parent).await
                .context("Error creating parents to save path")?;
        }

        // Create the local file
        let mut file = File::create(local_path).await
            .context("Error creating local file path")?;

        // Stream the S3 object body to the local file
        let mut stream = resp.body;
        
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