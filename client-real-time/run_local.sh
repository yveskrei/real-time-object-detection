#!/bin/bash
set -e

# Run triton server
docker compose -f ../docker-compose.yml up -d

# Wait for triton server to be ready
while true; do
    status_code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v2/health/ready)
    if [ "$status_code" -eq 200 ]; then
        echo "Triton Server is ready!"
        break
    else
        echo "Triton Server not ready yet. Waiting..."
        sleep 2
    fi
done

# Build client up front so APP_PID below is the real binary, not a cargo wrapper
(cd client && cargo build --release)

# Start Triton client application
export PLAYER_BACKEND_URL="http://127.0.0.1:8702"
export RUST_LOG=INFO
(cd client && exec ./target/release/client) &
APP_PID=$!

# Define cleanup function
cleanup() {
    if kill -0 "$APP_PID" 2>/dev/null; then
        # Ask the app to shut down cleanly (it listens for SIGINT)
        kill -INT "$APP_PID" 2>/dev/null || true

        # Give it up to 5 seconds to exit gracefully
        for _ in $(seq 1 50); do
            kill -0 "$APP_PID" 2>/dev/null || break
            sleep 0.1
        done

        # Force kill if still alive
        kill -KILL "$APP_PID" 2>/dev/null || true
    fi
    wait "$APP_PID" 2>/dev/null || true

    # Stop Triton Server
    docker compose -f ../docker-compose.yml down
}

# Cleanup on any exit path (Ctrl+C, SIGTERM, normal exit)
trap cleanup EXIT

# Wait for application to finish
wait "$APP_PID"
