use anyhow::{Result, Context};

use library::stream;

fn main() -> Result<()> {
    stream::init_ffmpeg()
        .context("Cannot initiate ffmpeg")?;
    stream::start_stream(1)
        .context("Cannot start stream")?;

    Ok(())
}