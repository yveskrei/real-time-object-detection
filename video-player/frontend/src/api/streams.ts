import { apiClient } from './client';
import type { Video, StreamStatus } from '../types';

export const listVideos = async (): Promise<Video[]> => {
    const response = await apiClient.get<Video[]>('/videos/');
    return response.data;
};

export const uploadVideo = async (file: File, name: string): Promise<Video> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', name);
    const response = await apiClient.post<Video>('/videos/upload', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });
    return response.data;
};

export const deleteVideo = async (videoId: number): Promise<void> => {
    await apiClient.delete(`/videos/${videoId}`);
};

export const startStream = async (videoId: number): Promise<void> => {
    await apiClient.post('/streams/start', { video_id: videoId });
};

export const stopStream = async (videoId: number): Promise<void> => {
    await apiClient.post(`/streams/stop/${videoId}`);
};

export const getStreamStatus = async (videoId: number): Promise<StreamStatus> => {
    const response = await apiClient.get<StreamStatus>(`/streams/status/${videoId}`);
    return response.data;
};
