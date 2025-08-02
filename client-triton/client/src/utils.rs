use std::io::{Error, ErrorKind};
use image::{ImageReader, GenericImageView};

pub fn get_image_raw(path: &str) -> Result<(Vec<u8>, usize, usize), Error> {
    let image = ImageReader::open(path).unwrap().decode().unwrap();

    // Get dimensions
    let (width, height) = image.dimensions();
    let width = width as usize;
    let height = height as usize;

    // Convert to RGB8 if needed
    let img_rgb8 = image.to_rgb8();

    // Get raw pixel data
    Ok((img_rgb8.into_raw(), height, width))
}