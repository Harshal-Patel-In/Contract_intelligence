// API Base URL configuration
// In production (GitHub Pages), use VITE_API_BASE_URL or default to Render URL
// In development (Vite dev server on port 5173), proxy to local FastAPI on port 8000

const isProduction = window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1';

export const API_BASE_URL = isProduction
  ? (import.meta.env.VITE_API_BASE_URL || 'https://contract-intelligence-api-uuse.onrender.com')
  : 'http://127.0.0.1:8000';