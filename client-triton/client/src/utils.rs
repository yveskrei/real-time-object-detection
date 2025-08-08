use image::{ImageReader, GenericImageView};
use anyhow::{Result, Context};

// Custom modules
pub mod config;
pub mod s3;

pub fn get_image_raw(path: &str) -> Result<(Vec<u8>, usize, usize)> {
    let image = ImageReader::open(path)
        .context("Error opening image from path")?
        .decode()
        .context("Error decoding image")?;

    // Get dimensions
    let (width, height) = image.dimensions();
    let width = width as usize;
    let height = height as usize;

    // Convert to RGB8 if needed
    let img_rgb8 = image.to_rgb8();

    // Get raw pixel data
    Ok((img_rgb8.into_raw(), height, width))
}