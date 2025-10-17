#!/bin/bash

# Configuration
PORT=8702
BACKEND_URL="http://localhost:$PORT"

# Trap to kill all background processes on exit
trap 'kill $(jobs -p)' EXIT

# Run backend
(cd backend && uvicorn main:app --reload --port $PORT) &

# Wait for backend to be healthy
echo "Waiting for backend to be ready on port $PORT..."
until curl -s $BACKEND_URL/health > /dev/null 2>&1 && [ $(curl -s -o /dev/null -w "%{http_code}" $BACKEND_URL/health) -eq 200 ]; do
    echo "Backend not ready yet, retrying in 1 second..."
    sleep 1
done
echo "Backend is ready!"

# Run frontend with backend URL as argument
(cd frontend && python main.py $BACKEND_URL) &

# Wait for all background processes
wait