#!/bin/bash

# Prompt for the tag
read -p "Enter the tag for the Docker image: " tag

# Check if a tag was provided
if [ -z "$tag" ]; then
  echo "Error: No tag provided."
  exit 1
fi

# Execute the Docker build command
docker build -t fajyz/flux-app-training-serverless-worker:"$tag" --platform linux/amd64 .

# Check if the build was successful
if [ $? -eq 0 ]; then
  echo "Docker image built successfully with tag: $tag"
else
  echo "Docker build failed."
fi
