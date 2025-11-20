import { useEffect, useRef, useImperativeHandle, forwardRef } from 'react';
import * as dashjs from 'dashjs';
import toast from 'react-hot-toast';

interface VideoPlayerProps {
    manifestUrl: string;
    onResolutionChange: (width: number, height: number) => void;
}

export const VideoPlayer = forwardRef<HTMLVideoElement, VideoPlayerProps>(({ manifestUrl, onResolutionChange }, ref) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const playerRef = useRef<dashjs.MediaPlayerClass | null>(null);

    useImperativeHandle(ref, () => videoRef.current!);

    useEffect(() => {
        if (!manifestUrl || !videoRef.current) return;

        // Prevent re-initialization if player already exists for this manifest
        if (playerRef.current) {
            console.log('Player already initialized, skipping...');
            return;
        }

        console.log('Initializing dash.js player for:', manifestUrl);
        const player = dashjs.MediaPlayer().create();

        // Configure for low-latency live streaming
        player.updateSettings({
            streaming: {
                delay: {
                    liveDelay: 4.0 // 4.0 seconds target latency for better buffering
                },
                liveCatchup: {
                    mode: 'liveCatchupModeDefault',
                    enabled: true,
                    maxDrift: 0,
                    playbackRate: {
                        min: -0.5,
                        max: 0.5
                    }
                },
                manifestUpdateRetryInterval: 5000, // Check for manifest updates every 5 seconds
                retryIntervals: {
                    MPD: 5000, // Retry MPD download every 5s if failed
                },
                retryAttempts: {
                    MPD: 3,
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
            console.error('Dash.js Error:', e);
            // Only toast critical errors to avoid spam
            if (e.error === 'capability' || e.error === 'mediasource' || e.error === 'key_session') {
                toast.error(`Playback Error: ${e.event ? e.event.message : 'Unknown error'}`);
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
            console.log('Cleaning up dash.js player');
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
            controls
            className="w-full h-full bg-black"
            style={{ objectFit: 'contain', maxWidth: '100%', maxHeight: '100%' }}
        />
    );
});

VideoPlayer.displayName = 'VideoPlayer';
