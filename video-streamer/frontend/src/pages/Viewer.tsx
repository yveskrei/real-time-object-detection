import React, { useEffect, useRef, useState, useCallback } from 'react';
import toast from 'react-hot-toast';
import { listVideos, getStreamStatus } from '../api/streams';
import type { Video, BBox } from '../types';
import { VideoPlayer } from '../components/VideoPlayer';
import { BBoxOverlay } from '../components/BBoxOverlay';
import { useWebSocket } from '../hooks/useWebSocket';
import { RefreshCw, Square, Eye, EyeOff, Play, Pin, Monitor, Activity } from 'lucide-react';
import clsx from 'clsx';

export const Viewer: React.FC = () => {
    const [streams, setStreams] = useState<Video[]>([]);
    const [selectedStreamId, setSelectedStreamId] = useState<number | null>(null);
    const [manifestUrl, setManifestUrl] = useState<string | null>(null);

    // BBox Config
    const [minConfidence, setMinConfidence] = useState(0.0);
    const [retentionFrames, setRetentionFrames] = useState(1);
    const [showBBoxes, setShowBBoxes] = useState(true);
    const [showControls, setShowControls] = useState(true);

    // Video State
    const [originalRes, setOriginalRes] = useState({ width: 0, height: 0 });
    const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
    const [videoOffset, setVideoOffset] = useState({ x: 0, y: 0 });

    // Refs
    const videoRef = useRef<HTMLVideoElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const requestRef = useRef<number>(null);

    // WebSocket
    const { isConnected, bboxBuffer } = useWebSocket(selectedStreamId);

    const [activeBBoxes, setActiveBBoxes] = useState<BBox[]>([]);

    // Fetch streams
    const fetchStreams = useCallback(async () => {
        try {
            const allVideos = await listVideos();
            const active = allVideos.filter(v => v.is_streaming);
            setStreams(active);
        } catch (e) {
            console.error(e);
        }
    }, []);

    useEffect(() => {
        fetchStreams();
        const interval = setInterval(fetchStreams, 5000);
        return () => clearInterval(interval);
    }, [fetchStreams]);

    // Handle Stream Selection
    const handleStreamSelect = async (id: number) => {
        try {
            const status = await getStreamStatus(id);
            if (status.dash?.manifest_url) {
                setSelectedStreamId(id);
                setManifestUrl(status.dash.manifest_url);
            } else {
                toast.error('No DASH manifest found for this stream');
            }
        } catch (e) {
            console.error(e);
            toast.error('Failed to get stream info');
        }
    };

    const handleStopWatching = () => {
        setSelectedStreamId(null);
        setManifestUrl(null);
        setOriginalRes({ width: 0, height: 0 });
    };

    // Resize Observer - track actual video display size and position
    useEffect(() => {
        if (!containerRef.current || !videoRef.current) return;

        const updateSize = () => {
            if (containerRef.current && videoRef.current) {
                const containerRect = containerRef.current.getBoundingClientRect();
                const videoElement = videoRef.current;

                // Get video's natural dimensions
                const videoWidth = videoElement.videoWidth || originalRes.width;
                const videoHeight = videoElement.videoHeight || originalRes.height;

                if (videoWidth && videoHeight) {
                    // Calculate actual display size maintaining aspect ratio
                    const containerAspect = containerRect.width / containerRect.height;
                    const videoAspect = videoWidth / videoHeight;

                    let displayWidth, displayHeight;
                    let xOffset = 0;
                    let yOffset = 0;

                    if (containerAspect > videoAspect) {
                        // Container is wider - video is limited by height
                        displayHeight = containerRect.height;
                        displayWidth = displayHeight * videoAspect;
                        xOffset = (containerRect.width - displayWidth) / 2;
                    } else {
                        // Container is taller - video is limited by width
                        displayWidth = containerRect.width;
                        displayHeight = displayWidth / videoAspect;
                        yOffset = (containerRect.height - displayHeight) / 2;
                    }

                    setContainerSize({
                        width: displayWidth,
                        height: displayHeight
                    });
                    setVideoOffset({ x: xOffset, y: yOffset });
                } else {
                    // Fallback to container size
                    setContainerSize({
                        width: containerRect.width,
                        height: containerRect.height
                    });
                    setVideoOffset({ x: 0, y: 0 });
                }
            }
        };

        // Set initial size immediately
        updateSize();

        const observer = new ResizeObserver(() => {
            updateSize();
        });

        observer.observe(containerRef.current);

        // Also update when video metadata loads
        const videoElement = videoRef.current;
        if (videoElement) {
            videoElement.addEventListener('loadedmetadata', updateSize);
            videoElement.addEventListener('resize', updateSize);
        }

        return () => {
            observer.disconnect();
            if (videoElement) {
                videoElement.removeEventListener('loadedmetadata', updateSize);
                videoElement.removeEventListener('resize', updateSize);
            }
        };
    }, [selectedStreamId, originalRes]); // Re-run when stream or resolution changes

    // Animation Loop
    const animate = useCallback(() => {
        if (!videoRef.current || !selectedStreamId) {
            requestRef.current = requestAnimationFrame(animate);
            return;
        }

        const currentTime = videoRef.current.currentTime;
        const buffer = bboxBuffer.current;

        // Simple PTS calculation: video.currentTime is in seconds, PTS is in 90kHz
        const currentPts = currentTime * 90000;

        const ptsPerFrame = 3000; // ~30fps (90000 / 30)

        // Retention: n frames means show bbox for n frames AFTER it first appears
        // If retention = 1, only show when PTS matches exactly (within tolerance)
        // If retention = 2, show for current frame + 1 more frame
        const tolerance = ptsPerFrame * 2; // Small tolerance for matching
        const retentionWindow = ptsPerFrame * retentionFrames; // How long to keep showing after match

        const activeBBoxes: BBox[] = [];

        // Find bboxes where: bbox.pts <= currentPts < (bbox.pts + retentionWindow)
        for (let i = buffer.length - 1; i >= 0; i--) {
            const msg = buffer[i];

            // Check if this bbox's PTS is within our retention window
            if (msg.pts <= currentPts + tolerance && msg.pts >= currentPts - retentionWindow) {
                activeBBoxes.push(...msg.bboxes);
            }
        }

        // Reduced logging frequency
        if (Math.random() < 0.005) {
            console.log('PTS:', currentPts.toFixed(0), 'Active bboxes:', activeBBoxes.length, 'Retention frames:', retentionFrames);
        }

        setActiveBBoxes(activeBBoxes);
        requestRef.current = requestAnimationFrame(animate);
    }, [selectedStreamId, retentionFrames, bboxBuffer]);

    useEffect(() => {
        requestRef.current = requestAnimationFrame(animate);
        return () => {
            if (requestRef.current) cancelAnimationFrame(requestRef.current);
        };
    }, [animate]);

    // Helper to get full URL
    const getFullManifestUrl = (path: string) => {
        if (path.startsWith('http')) return path;
        const baseUrl = localStorage.getItem('backend_url') || 'http://localhost:8702';
        return `${baseUrl}${path}`;
    };

    return (
        <div className="h-[calc(100vh-8rem)] flex flex-col gap-6">
            {/* Stream Selection Bar */}
            <div className="card p-4 flex items-center justify-between bg-surface/50 backdrop-blur-sm">
                <div className="flex items-center gap-4 flex-1">
                    <div className="relative w-64">
                        <select
                            className="input w-full appearance-none cursor-pointer pr-10 truncate"
                            value={selectedStreamId || ''}
                            onChange={(e) => {
                                const val = e.target.value;
                                if (val) handleStreamSelect(parseInt(val));
                                else handleStopWatching();
                            }}
                            disabled={!!selectedStreamId}
                            style={{ textOverflow: 'ellipsis' }}
                        >
                            <option value="">Select Active Stream...</option>
                            {streams.map(s => (
                                <option key={s.id} value={s.id}>ID {s.id}: {s.name}</option>
                            ))}
                        </select>
                        <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-zinc-500">
                            <Monitor className="w-4 h-4" />
                        </div>
                    </div>

                    {!selectedStreamId && (
                        <button onClick={fetchStreams} className="btn btn-ghost p-2 rounded-full" title="Refresh Streams">
                            <RefreshCw className="w-4 h-4" />
                        </button>
                    )}
                </div>

                {selectedStreamId && (
                    <div className="flex items-center gap-3">
                        <div className={clsx(
                            "flex items-center px-3 py-1.5 rounded-full text-xs font-medium border transition-colors",
                            isConnected
                                ? "bg-green-500/10 text-green-400 border-green-500/20"
                                : "bg-red-500/10 text-red-400 border-red-500/20"
                        )}>
                            <Activity className="w-3 h-3 mr-1.5" />
                            {isConnected ? 'Live Connection' : 'Disconnected'}
                        </div>

                        <button
                            onClick={handleStopWatching}
                            className="btn btn-danger text-xs px-3 py-1.5"
                        >
                            <Square className="w-3 h-3 mr-1.5" />
                            Stop Watching
                        </button>
                    </div>
                )}
            </div>

            {/* Main Viewer Area */}
            <div className="flex-1 flex gap-6 min-h-0">
                {/* Video Player Container */}
                <div className="flex-1 bg-black rounded-xl overflow-hidden relative shadow-2xl border border-zinc-800 group">
                    {!selectedStreamId ? (
                        <div className="absolute inset-0 flex flex-col items-center justify-center text-zinc-600 bg-zinc-950">
                            <div className="p-6 rounded-full bg-zinc-900 mb-4">
                                <Play className="w-12 h-12 opacity-50" />
                            </div>
                            <p className="text-lg font-medium">Select a stream to start viewing</p>
                        </div>
                    ) : (
                        <div ref={containerRef} className="relative w-full h-full">
                            {manifestUrl && (
                                <VideoPlayer
                                    ref={videoRef}
                                    manifestUrl={getFullManifestUrl(manifestUrl)}
                                    onResolutionChange={(w, h) => setOriginalRes({ width: w, height: h })}
                                />
                            )}
                            <BBoxOverlay
                                bboxes={activeBBoxes}
                                originalWidth={originalRes.width}
                                originalHeight={originalRes.height}
                                width={containerSize.width}
                                height={containerSize.height}
                                minConfidence={minConfidence}
                                show={showBBoxes}
                                offsetX={videoOffset.x}
                                offsetY={videoOffset.y}
                            />

                            {/* Overlay Controls */}
                            <div className={clsx(
                                "absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-black/90 via-black/50 to-transparent transition-opacity duration-300",
                                showControls ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                            )}>
                                <div className="flex items-center justify-between max-w-3xl mx-auto bg-zinc-900/90 backdrop-blur-md rounded-xl p-4 border border-white/10 shadow-xl">
                                    <div className="flex items-center gap-6">
                                        <div className="flex flex-col gap-1">
                                            <label className="text-[10px] uppercase tracking-wider font-bold text-zinc-500">Confidence</label>
                                            <div className="flex items-center gap-3">
                                                <input
                                                    type="range"
                                                    min="0"
                                                    max="1"
                                                    step="0.05"
                                                    value={minConfidence}
                                                    onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
                                                    className="w-24 accent-primary h-1.5 bg-zinc-700 rounded-lg appearance-none cursor-pointer"
                                                />
                                                <span className="text-xs font-mono text-primary w-8 text-right">{(minConfidence * 100).toFixed(0)}%</span>
                                            </div>
                                        </div>

                                        <div className="w-px h-8 bg-zinc-700" />

                                        <div className="flex flex-col gap-1">
                                            <label className="text-[10px] uppercase tracking-wider font-bold text-zinc-500">Retention</label>
                                            <div className="flex items-center gap-2">
                                                <input
                                                    type="number"
                                                    min="1"
                                                    max="30"
                                                    value={retentionFrames}
                                                    onChange={(e) => setRetentionFrames(parseInt(e.target.value))}
                                                    className="w-12 bg-zinc-800 border border-zinc-700 rounded px-2 py-0.5 text-xs text-center focus:border-primary outline-none"
                                                />
                                                <span className="text-xs text-zinc-400">frames</span>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="flex items-center gap-3">
                                        <button
                                            onClick={() => setShowBBoxes(!showBBoxes)}
                                            className={clsx(
                                                "flex items-center px-3 py-1.5 rounded-lg text-xs font-medium transition-all",
                                                showBBoxes
                                                    ? "bg-primary text-white shadow-lg shadow-primary/25"
                                                    : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
                                            )}
                                        >
                                            {showBBoxes ? <Eye className="w-3 h-3 mr-1.5" /> : <EyeOff className="w-3 h-3 mr-1.5" />}
                                            {showBBoxes ? 'Overlay On' : 'Overlay Off'}
                                        </button>

                                        <button
                                            onClick={() => setShowControls(!showControls)}
                                            className={clsx(
                                                "p-2 rounded-lg transition-colors",
                                                showControls ? "text-primary bg-primary/10" : "text-zinc-400 hover:text-white"
                                            )}
                                            title="Toggle Controls Pin"
                                        >
                                            <Pin className="w-4 h-4" />
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};
