#!/bin/bash

# Remove current instances of triton server
docker compose down

# Run triton server
docker compose up -d

# Wait for triton server to be ready
while true; do
    # Fetch the URL and capture the HTTP status code
    status_code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v2/health/ready)
    
    if [ "$status_code" -eq 200 ]; then
        echo "Triton Server is ready! (Status: $status_code)"
        break
    else
        echo "Triton Server not ready yet (Status: $status_code)"
        sleep 2  # Wait 2 seconds before trying again
    fi
done

# Set environment variables
export PLAYER_BACKEND_URL="http://127.0.0.1:8702"
export RUST_LOG=info,rdkafka=off,librdkafka=off

# Start binary
cd client
cargo run --release