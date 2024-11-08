# Stage 1: Base image with common dependencies
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI repository
RUN git clone https://github.com/DIRECTcut/flux-app-training-serverless-worker.git /ai-toolkit

WORKDIR /ai-toolkit

# install ai-toolkit deps
RUN git submodule update --init --recursive
# RUN python3 -m venv venv
# RUN source venv/bin/activate
# .\venv\Scripts\activate on windows
# install torch first
RUN pip3 install --upgrade --no-cache-dir torch
RUN pip3 install --upgrade -r requirements.txt

# Install runpod
RUN pip3 install runpod requests

# Go back to the root
WORKDIR /

# Add everything for now (TODO: find what we dont need and remove)
ADD ./ .
RUN chmod +x /start.sh


# Start the container
CMD /start.sh