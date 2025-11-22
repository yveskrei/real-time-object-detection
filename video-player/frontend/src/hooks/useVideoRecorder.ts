import { useState, useRef, useEffect, useCallback } from 'react';
import { drawBBoxes } from '../utils/drawing';
import type { BBox } from '../types';
import * as Mp4Muxer from 'mp4-muxer';

interface UseVideoRecorderProps {
    videoRef: React.RefObject<HTMLVideoElement | null>;
    bboxes: BBox[];
    originalWidth: number;
    originalHeight: number;
    minConfidence: number;
}

export const useVideoRecorder = ({
    videoRef,
    bboxes,
    originalWidth,
    originalHeight,
    minConfidence
}: UseVideoRecorderProps) => {
    const [isRecording, setIsRecording] = useState(false);
    const [recordingDuration, setRecordingDuration] = useState(0);

    const encoderRef = useRef<VideoEncoder | null>(null);
    const chunksRef = useRef<{ chunk: EncodedVideoChunk, meta?: EncodedVideoChunkMetadata }[]>([]);
    const decoderConfigRef = useRef<VideoDecoderConfig | null>(null);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const workerRef = useRef<Worker | null>(null);
    const startTimeRef = useRef<number>(0);

    // Configuration
    const FPS = 30;
    const BITRATE = 8_000_000; // 8 Mbps
    const BUFFER_DURATION_SEC = 30;

    // Cleanup function
    const cleanup = useCallback(() => {
        if (workerRef.current) {
            workerRef.current.terminate();
            workerRef.current = null;
        }

        if (encoderRef.current) {
            if (encoderRef.current.state !== 'closed') {
                encoderRef.current.close();
            }
            encoderRef.current = null;
        }

        setIsRecording(false);
        setRecordingDuration(0);
        chunksRef.current = [];
        decoderConfigRef.current = null;
    }, []);

    // Auto-start recording when video dimensions are available
    useEffect(() => {
        if (!videoRef.current || originalWidth === 0 || originalHeight === 0) return;

        // If already recording with same dimensions, skip
        if (isRecording && canvasRef.current?.width === originalWidth && canvasRef.current?.height === originalHeight) {
            return;
        }

        // Cleanup previous recording if any
        cleanup();

        // Create offscreen canvas
        const canvas = document.createElement('canvas');
        canvas.width = originalWidth;
        canvas.height = originalHeight;
        canvasRef.current = canvas;

        const ctx = canvas.getContext('2d', { alpha: false });
        if (!ctx) return;

        // Initialize VideoEncoder
        const initEncoder = async () => {
            try {
                const encoder = new VideoEncoder({
                    output: (chunk, meta) => {
                        if (meta?.decoderConfig) {
                            decoderConfigRef.current = meta.decoderConfig;
                        }

                        chunksRef.current.push({ chunk, meta });

                        // Prune old chunks based on timestamp
                        // chunk.timestamp is in microseconds
                        const maxDurationMicros = BUFFER_DURATION_SEC * 1_000_000;
                        const lastTimestamp = chunksRef.current[chunksRef.current.length - 1].chunk.timestamp;

                        // Simple pruning: remove chunks older than buffer window relative to latest
                        while (chunksRef.current.length > 0 &&
                            (lastTimestamp - chunksRef.current[0].chunk.timestamp) > maxDurationMicros) {
                            chunksRef.current.shift();
                        }

                        // Update duration for UI (approximate based on buffer fullness)
                        const currentDuration = (chunksRef.current.length / FPS);
                        setRecordingDuration(Math.min(Math.round(currentDuration), BUFFER_DURATION_SEC));
                    },
                    error: (e) => {
                        console.error('VideoEncoder error:', e);
                    }
                });

                // Configure encoder
                // Use Baseline Profile (avc1.42001e) to avoid B-frames
                const config: VideoEncoderConfig = {
                    codec: 'avc1.42001e',
                    width: originalWidth,
                    height: originalHeight,
                    bitrate: BITRATE,
                    framerate: FPS,
                };

                // Check support
                const support = await VideoEncoder.isConfigSupported(config);
                if (!support.supported) {
                    console.warn('Baseline Profile not supported, trying High Profile');
                    config.codec = 'avc1.4d002a';
                }

                encoder.configure(config);
                encoderRef.current = encoder;
                setIsRecording(true);
                startTimeRef.current = performance.now();

                // Start Worker Timer
                // We use a Web Worker to drive the loop because requestAnimationFrame 
                // is throttled in background tabs, causing recording to stop.
                const blob = new Blob([`
                    let interval;
                    self.onmessage = function(e) {
                        if (e.data === 'start') {
                            interval = setInterval(function() {
                                self.postMessage('tick');
                            }, ${1000 / FPS});
                        } else if (e.data === 'stop') {
                            clearInterval(interval);
                        }
                    };
                `], { type: 'text/javascript' });

                const worker = new Worker(URL.createObjectURL(blob));
                workerRef.current = worker;

                let frameCount = 0;
                let lastFrameTime = 0;
                const frameInterval = 1000 / FPS;

                worker.onmessage = () => {
                    const now = performance.now();

                    if (!videoRef.current || !canvasRef.current || !encoderRef.current || encoderRef.current.state === 'closed') return;

                    // Throttle to FPS (still useful if worker drifts or bursts)
                    if (now - lastFrameTime < frameInterval - 2) { // -2ms tolerance
                        return;
                    }

                    lastFrameTime = now - ((now - lastFrameTime) % frameInterval);

                    if (videoRef.current.readyState >= 2) {
                        // Draw
                        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
                        drawBBoxes(
                            ctx,
                            latestBBoxes.current,
                            originalWidth,
                            originalHeight,
                            canvas.width,
                            canvas.height,
                            minConfidence
                        );

                        // Encode
                        // Timestamp in microseconds
                        const timestamp = (now - startTimeRef.current) * 1000;
                        const duration = (1000 / FPS) * 1000; // Duration in microseconds

                        const frame = new VideoFrame(canvas, {
                            timestamp,
                            duration
                        });

                        // Force keyframe every 2 seconds
                        const keyFrame = frameCount % (FPS * 2) === 0;

                        encoderRef.current.encode(frame, { keyFrame });
                        frame.close();

                        frameCount++;
                    }
                };

                worker.postMessage('start');

            } catch (e) {
                console.error("Failed to initialize VideoEncoder:", e);
            }
        };

        initEncoder();

        return () => {
            // Cleanup handled by parent effect dependency change or unmount calling cleanup()
        };
    }, [originalWidth, originalHeight, videoRef, cleanup, isRecording, minConfidence]);

    // Ref for latest bboxes
    const latestBBoxes = useRef(bboxes);
    useEffect(() => {
        latestBBoxes.current = bboxes;
    }, [bboxes]);

    const saveRecording = useCallback(async () => {
        // Flush any pending frames in the encoder to ensure we get the very latest footage
        if (encoderRef.current && encoderRef.current.state === 'configured') {
            try {
                await encoderRef.current.flush();
            } catch (e) {
                console.warn("Failed to flush encoder:", e);
            }
        }

        if (chunksRef.current.length === 0) return;

        try {
            // Sort chunks by timestamp to ensure monotonicity
            // This is safe for Baseline profile (no B-frames)
            chunksRef.current.sort((a, b) => a.chunk.timestamp - b.chunk.timestamp);

            // Find the first keyframe
            const firstKeyFrameIndex = chunksRef.current.findIndex(c => c.chunk.type === 'key');

            if (firstKeyFrameIndex === -1) {
                console.error("No keyframe found in buffer");
                return;
            }

            // Drop frames before the first keyframe to ensure valid playback start
            const validChunks = chunksRef.current.slice(firstKeyFrameIndex);

            if (validChunks.length === 0) return;

            // Ensure the first chunk has decoderConfig
            // If it was dropped from the buffer, we restore it from our cache
            if (!validChunks[0].meta?.decoderConfig && decoderConfigRef.current) {
                validChunks[0].meta = {
                    ...validChunks[0].meta,
                    decoderConfig: decoderConfigRef.current
                };
            }

            // Create Muxer
            const muxer = new Mp4Muxer.Muxer({
                target: new Mp4Muxer.ArrayBufferTarget(),
                video: {
                    codec: 'avc', // 'avc' for H.264
                    width: originalWidth,
                    height: originalHeight
                },
                firstTimestampBehavior: 'offset', // Fixes "first chunk must have timestamp 0" error
                fastStart: 'in-memory' // Move moov atom to front
            });

            chunksRef.current = validChunks; // Update ref to reflect what we are saving (optional, but good for consistency)

            // Feed chunks to muxer
            chunksRef.current.forEach(({ chunk, meta }) => {
                // Ensure duration is valid, fallback to 1/30s if missing
                const duration = chunk.duration || ((1000 / FPS) * 1000);

                // Pass original timestamp, muxer handles offset
                muxer.addVideoChunk(chunk, meta, chunk.timestamp, duration);
            });

            // Finalize
            muxer.finalize();

            // Get buffer
            const { buffer } = muxer.target;

            // Download
            const blob = new Blob([buffer], { type: 'video/mp4' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `recording-${new Date().toISOString()}.mp4`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

        } catch (e) {
            console.error("Muxing failed:", e);
            alert("Failed to save video: " + e);
        }
    }, [originalWidth, originalHeight]);

    return {
        isRecording,
        recordingDuration,
        stopRecording: cleanup,
        saveRecording
    };
};
