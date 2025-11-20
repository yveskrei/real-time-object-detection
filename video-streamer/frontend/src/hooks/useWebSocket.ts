import { useEffect, useRef, useState, useCallback } from 'react';
import { getBackendUrl } from '../api/client';
import type { BBoxMessage } from '../types';

export const useWebSocket = (videoId: number | null) => {
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const wsRef = useRef<WebSocket | null>(null);
    const bboxBufferRef = useRef<BBoxMessage[]>([]);

    const connect = useCallback(() => {
        if (videoId === null) return;

        const backendUrl = getBackendUrl();
        const wsUrl = backendUrl.replace(/^http/, 'ws').replace(/^https/, 'wss') + `/ws/${videoId}`;

        console.log(`Connecting to WebSocket: ${wsUrl}`);
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log('WebSocket Connected');
            setIsConnected(true);
            setError(null);
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                console.log('WebSocket message received:', data.type, data);

                if (data.type === 'bboxes') {
                    console.log('BBox message - PTS:', data.pts, 'BBoxes count:', data.bboxes?.length || 0);
                    // Add to buffer
                    // We keep a buffer of recent messages to sync with video
                    bboxBufferRef.current.push(data);

                    // Limit buffer size (e.g., keep last 500 messages)
                    if (bboxBufferRef.current.length > 500) {
                        bboxBufferRef.current.shift();
                    }
                    console.log('BBox buffer size now:', bboxBufferRef.current.length);
                } else if (data.type === 'stream_info') {
                    console.log('Stream info received:', data);
                } else if (data.type === 'pong') {
                    // Heartbeat response, don't log
                } else if (data.type === 'error') {
                    console.error('WebSocket Error Message:', data.message);
                } else {
                    console.log('Unknown message type:', data.type);
                }
            } catch (e) {
                console.error('Failed to parse WebSocket message', e, event.data);
            }
        };

        ws.onerror = (e) => {
            console.error('WebSocket Error:', e);
            setError('Connection failed');
        };

        ws.onclose = () => {
            console.log('WebSocket Disconnected');
            setIsConnected(false);
        };

        wsRef.current = ws;
    }, [videoId]);

    useEffect(() => {
        if (videoId !== null) {
            connect();
        }

        return () => {
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }
            bboxBufferRef.current = [];
        };
    }, [videoId, connect]);

    return {
        isConnected,
        error,
        bboxBuffer: bboxBufferRef,
    };
};
