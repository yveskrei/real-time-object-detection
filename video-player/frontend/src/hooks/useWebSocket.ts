import { useEffect, useRef, useState, useCallback } from 'react';
import { getBackendUrl } from '../api/client';
import type { BBoxMessage } from '../types';

export const useWebSocket = (videoId: number | null, onDisconnect?: () => void) => {
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const wsRef = useRef<WebSocket | null>(null);
    const bboxBufferRef = useRef<BBoxMessage[]>([]);

    const connect = useCallback(() => {
        if (videoId === null) return;

        const backendUrl = getBackendUrl();
        const wsUrl = backendUrl.replace(/^http/, 'ws').replace(/^https/, 'wss') + `/ws/${videoId}`;

        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            setIsConnected(true);
            setError(null);
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                if (data.type === 'bboxes') {
                    // Add to buffer
                    // We keep a buffer of recent messages to sync with video
                    bboxBufferRef.current.push(data);

                    // Limit buffer size (e.g., keep last 500 messages)
                    if (bboxBufferRef.current.length > 500) {
                        bboxBufferRef.current.shift();
                    }
                } else if (data.type === 'stream_info') {
                    // Stream info received
                } else if (data.type === 'pong') {
                    // Heartbeat response
                } else if (data.type === 'error') {
                    console.error('WebSocket Error Message:', data.message);
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
            setIsConnected(false);
            if (onDisconnect) onDisconnect();
        };

        wsRef.current = ws;
    }, [videoId, onDisconnect]);

    useEffect(() => {
        if (videoId !== null) {
            connect();
        }

        return () => {
            if (wsRef.current) {
                // Prevent triggering onDisconnect during manual cleanup
                wsRef.current.onclose = null;
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
