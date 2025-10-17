use crate::player_proxy::{PlayerSession, TOKIO_RUNTIME};
use ffmpeg_next as ffmpeg;
use anyhow::{Context, Result, anyhow};
use std::time::{SystemTime, UNIX_EPOCH};

/// Initialize FFmpeg (call once at library load)
pub fn init_ffmpeg() -> Result<()> {
    ffmpeg::init().context("Failed to initialize FFmpeg")?;
    
    // Set log level to only show errors and fatal messages (suppress warnings)
    ffmpeg::log::set_level(ffmpeg::log::Level::Error);
    
    Ok(())
}

/// Start receiving frames from a streaming video
/// Returns an error if the stream is not active
pub fn start_stream(video_id: i32) -> Result<()> {
    // Create session
    let session = PlayerSession::new()?;
    
    // Check stream status using the runtime
    let status = TOKIO_RUNTIME.block_on(async {
        session.get_stream_status(video_id).await
    })?;

    // Check if stream is active
    if !status.is_streaming {
        return Err(anyhow!(
            "Stream for video {} is not active. Please start the stream first using the backend API.",
            video_id
        ));
    }

    // Get stream URL and start time
    let stream_url = status.stream_url
        .ok_or_else(|| anyhow!("Stream is active but no URL available"))?;
    let stream_start_time_ms = status.stream_start_time_ms
        .ok_or_else(|| anyhow!("Stream is active but no start time available"))?;
    
    println!("Stream is active for video {}!", video_id);
    println!("  Stream URL: {}", stream_url);
    println!("  Stream start time: {} ms", stream_start_time_ms);
    println!("  PID: {}", status.pid.unwrap_or(0));

    // Give the stream a moment to stabilize
    std::thread::sleep(std::time::Duration::from_millis(500));

    println!("\nConnecting to stream and reading frames...");
    println!("Stream start timestamp (from metadata): {} ms\n", stream_start_time_ms);

    // Open the input stream with FFmpeg
    let mut ictx = ffmpeg::format::input(&stream_url)
        .context("Failed to open stream URL")?;

    // Find the video stream
    let input_stream = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or_else(|| anyhow!("No video stream found"))?;
    
    let stream_index = input_stream.index();

    println!("Connected to video stream (index: {})", stream_index);
    println!("Reading frames...\n");

    let mut frame_count = 0;

    // Read packets and frames
    for (stream, packet) in ictx.packets() {
        if stream.index() == stream_index {
            // Get current system time in milliseconds
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .context("Failed to get system time")?
                .as_millis() as i64;

            // Calculate timestamp relative to stream start
            let timestamp_since_stream_start = now - stream_start_time_ms;

            // Get packet presentation timestamp (in stream time base)
            let pts = packet.pts().unwrap_or(0);
            let time_base = stream.time_base();
            
            // Convert PTS to milliseconds
            let pts_ms = (pts as f64 * time_base.numerator() as f64 * 1000.0) 
                / time_base.denominator() as f64;

            frame_count += 1;

            println!(
                "Frame #{}: Stream timestamp: {} ms | PTS: {} ({}ms) | Time since stream start: {} ms",
                frame_count,
                stream_start_time_ms,
                pts,
                pts_ms as i64,
                timestamp_since_stream_start
            );
        }
    }

    println!("\nStream ended. Total frames received: {}", frame_count);

    Ok(())
}