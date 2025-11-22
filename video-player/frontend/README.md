# Video Streamer Frontend

Modern React-based video streaming application with real-time bounding box overlay.

## Tech Stack

- **React** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **dash.js** for DASH video playback
- **WebSocket** for real-time bbox updates

## Setup

### 1. Install Dependencies

```bash
npm install
```

### 2. Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and set your backend URL:

```env
VITE_BACKEND_URL=http://localhost:8702
```

### 3. Run Development Server

```bash
npm run dev
```

The app will be available at `http://localhost:5173`

### 4. Build for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

## Configuration

### Environment Variables

- `VITE_BACKEND_URL`: Backend API URL (default: `http://localhost:8702`)

### Runtime Configuration

You can also override the backend URL at runtime using localStorage:

```javascript
localStorage.setItem('backend_url', 'http://your-backend-url:port');
```

## Features

- **Live DASH Streaming**: Adaptive bitrate video streaming
- **Real-time BBox Overlay**: WebSocket-based bounding box display synchronized with video PTS
- **Frame-based Retention**: Configurable bbox persistence
- **Aspect Ratio Preservation**: Video and bboxes maintain correct proportions
- **Modern UI**: Dark theme with glassmorphism effects
- **Stream Management**: Start, stop, and monitor video streams

## Project Structure

```
frontend/
├── src/
│   ├── api/          # API client and endpoints
│   ├── components/   # Reusable React components
│   ├── hooks/        # Custom React hooks
│   ├── pages/        # Page components
│   ├── types/        # TypeScript type definitions
│   └── App.tsx       # Main application component
├── .env              # Environment configuration (gitignored)
├── .env.example      # Example environment configuration
└── package.json      # Dependencies and scripts
```

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build locally
- `npm run lint` - Run ESLint

### Hot Module Replacement

Vite provides instant HMR for a smooth development experience. Changes to React components will update instantly without losing state.

## Backend Integration

This frontend connects to the video-streamer backend API. Make sure the backend is running before starting the frontend:

```bash
# In the backend directory
cd ../backend
source ../venv/bin/activate
uvicorn main:app --reload --port 8702
```
