import axios from 'axios';

// Get backend URL from environment variable, with localStorage override support
const BASE_URL = localStorage.getItem('backend_url') || import.meta.env.VITE_BACKEND_URL || 'http://localhost:8702';

export const apiClient = axios.create({
    baseURL: BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const setBackendUrl = (url: string) => {
    localStorage.setItem('backend_url', url);
    apiClient.defaults.baseURL = url;
};

export const getBackendUrl = () => apiClient.defaults.baseURL || '';
