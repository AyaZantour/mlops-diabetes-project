#!/bin/bash

echo "🧪 Testing API Deployment Workflow"
echo "=================================="

# Test v1
echo -e "\n📌 Step 1: Testing v1 API..."
docker-compose up -d api
sleep 5

echo "Making prediction request to v1..."
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [0.038, 0.050, 0.061, 0.021, -0.044, -0.034, -0.043, -0.002, 0.019, -0.017]}'

# Update to v2
echo -e "\n\n📌 Step 2: Updating to v2..."
docker-compose stop api

# Modify docker-compose to use v2
export MODEL_PATH=/app/models/model_v2.pkl
export MODEL_VERSION=v2

docker-compose up -d api
sleep 5

echo "Making prediction request to v2..."
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [0.038, 0.050, 0.061, 0.021, -0.044, -0.034, -0.043, -0.002, 0.019, -0.017]}'

# Rollback to v1
echo -e "\n\n📌 Step 3: Rolling back to v1..."
docker-compose stop api

export MODEL_PATH=/app/models/baseline.pkl
export MODEL_VERSION=v1

docker-compose up -d api
sleep 5

echo "Making prediction request after rollback..."
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [0.038, 0.050, 0.061, 0.021, -0.044, -0.034, -0.043, -0.002, 0.019, -0.017]}'

echo -e "\n\n✅ Deployment test complete!"
