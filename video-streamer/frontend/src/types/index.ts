export interface Video {
    id: number;
    name: string;
    is_streaming: boolean;
}

export interface StreamStatus {
    is_streaming: boolean;
    stream_start_time_ms: number | null;
    dash: {
        manifest_url: string;
    } | null;
    udp: {
        port: number;
    } | null;
}

export interface BBox {
    top_left_corner: number;
    bottom_right_corner: number;
    class_name: string;
    confidence: number;
}

export interface BBoxMessage {
    pts: number;
    bboxes: BBox[];
    stream_start_time_ms?: number;
    video_id?: number;
    timestamp?: number;
}

export interface StreamInfoMessage {
    type: 'stream_info';
    // Add fields if needed
}
