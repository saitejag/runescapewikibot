#!/bin/bash

# Start Redis Stack server in the background
redis-stack-server &
REDIS_PID=$!

# Start the FastAPI app in the background
python rwiki_bot_api.py &
FASTAPI_PID=$!

# Trap Ctrl+C (SIGINT) to stop both Redis and FastAPI when interrupted
trap "kill $REDIS_PID $FASTAPI_PID; exit" SIGINT

# Wait for both Redis and FastAPI processes
wait $REDIS_PID
wait $FASTAPI_PID

