#!/bin/bash

# Accept tag as command line argument
tag="$1"

# Check if a tag was provided
if [ -z "$tag" ]; then
  echo "Error: No tag provided. Usage: $0 <tag>"
  exit 1
fi

# Execute the Docker build command
depot build --push -t fajyz/flux-app-training-serverless-worker:"$tag" --platform linux/amd64 .

# Check if the build was successful
if [ $? -eq 0 ]; then
  echo "Docker image built successfully with tag: $tag"
else
  echo "Docker build failed."
fi
