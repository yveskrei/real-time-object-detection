import axios from 'axios';

// Get backend URL from environment variable, with localStorage override support

let url = localStorage.getItem('backend_url') || import.meta.env.VITE_BACKEND_URL;

// Ensure URL has protocol if it's not a relative path
if (url && !url.startsWith('http') && !url.startsWith('/')) {
    url = `http://${url}`;
}

const BASE_URL = url;

if (!BASE_URL) {
    console.warn('VITE_BACKEND_URL is not defined in .env file');
}

export const apiClient = axios.create({
    baseURL: BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const setBackendUrl = (url: string) => {
    if (url && !url.startsWith('http') && !url.startsWith('/')) {
        url = `http://${url}`;
    }
    localStorage.setItem('backend_url', url);
    apiClient.defaults.baseURL = url;
};

export const getBackendUrl = () => apiClient.defaults.baseURL || '';
