import { useEffect, useRef, useImperativeHandle, forwardRef } from 'react';
import * as dashjs from 'dashjs';
import toast from 'react-hot-toast';

interface VideoPlayerProps {
    manifestUrl: string;
    onResolutionChange: (width: number, height: number) => void;
    onError?: (error: string) => void;
}

export const VideoPlayer = forwardRef<HTMLVideoElement, VideoPlayerProps>(({ manifestUrl, onResolutionChange, onError }, ref) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const playerRef = useRef<dashjs.MediaPlayerClass | null>(null);

    useImperativeHandle(ref, () => videoRef.current!);

    useEffect(() => {
        if (!manifestUrl || !videoRef.current) return;

        // Prevent re-initialization if player already exists for this manifest
        if (playerRef.current) {
            return;
        }

        const player = dashjs.MediaPlayer().create();
        player.clearDefaultUTCTimingSources();

        // Configure for low-latency live streaming
        player.updateSettings({
            streaming: {
                utcSynchronization: {
                    enabled: false
                },
                delay: {
                    liveDelay: 6.0 // Increased to 6.0s (3 segments) for stability
                },
                liveCatchup: {
                    mode: 'liveCatchupModeDefault',
                    enabled: false, // Disabled to prevent speed-up/high FPS
                    maxDrift: 0,
                    playbackRate: {
                        min: 0,
                        max: 0
                    }
                },
                manifestUpdateRetryInterval: 1000, // Check for manifest updates every 1 second
                retryIntervals: {
                    MPD: 1000, // Retry MPD download every 1s if failed
                },
                retryAttempts: {
                    MPD: 2, // Give up after 2 retries
                },
                abr: {
                    limitBitrateByPortal: true,
                },
                buffer: {
                    bufferTimeAtTopQuality: 30,
                    bufferTimeAtTopQualityLongForm: 60,
                    bufferToKeep: 20,      // Keep 20 seconds behind playhead
                    fastSwitchEnabled: true
                }
            }
        });

        player.initialize(videoRef.current, manifestUrl, true);
        playerRef.current = player;

        // Error handling
        player.on(dashjs.MediaPlayer.events.ERROR, (e: any) => {
            // Only toast critical errors to avoid spam
            if (e.error === 'capability' || e.error === 'mediasource' || e.error === 'key_session') {
                toast.error(`Playback Error: ${e.event ? e.event.message : 'Unknown error'}`);
            } else {
                // Log other errors for debugging
                console.error('Dash.js Error:', e);
            }

            // Handle stream end (404 on manifest update usually means stream stopped)
            // Broadened check: any download error
            if (e.error === 'download') {
                if (onError) onError('Stream stopped');
            }
        });

        // Handle resolution changes
        const handleResize = () => {
            if (videoRef.current) {
                onResolutionChange(videoRef.current.videoWidth, videoRef.current.videoHeight);
            }
        };

        videoRef.current.addEventListener('loadedmetadata', handleResize);
        videoRef.current.addEventListener('resize', handleResize);

        return () => {
            // Proper cleanup sequence to avoid DOMException
            if (videoRef.current) {
                videoRef.current.removeEventListener('loadedmetadata', handleResize);
                videoRef.current.removeEventListener('resize', handleResize);
            }

            // Reset player before destroying
            if (playerRef.current) {
                try {
                    playerRef.current.reset();
                } catch (e) {
                    console.warn('Error resetting player:', e);
                }
                playerRef.current = null;
            }
        };
    }, [manifestUrl]); // Only re-run when manifestUrl changes

    return (
        <video
            ref={videoRef}
            disablePictureInPicture
            className="w-full h-full bg-black"
            style={{ objectFit: 'contain', maxWidth: '100%', maxHeight: '100%' }}
        />
    );
});

VideoPlayer.displayName = 'VideoPlayer';
