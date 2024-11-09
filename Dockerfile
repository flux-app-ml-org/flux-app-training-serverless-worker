FROM runpod/base:0.6.2-cuda12.2.0

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
RUN python3 -m pip install --upgrade --no-cache-dir torch
RUN python3 -m pip install --upgrade -r requirements.txt

# Install runpod
RUN python3 -m pip install runpod requests

# Go back to the root
WORKDIR /

# Add everything for now (TODO: find what we dont need and remove)
ADD ./ .
RUN chmod +x /start.sh


# Start the container
CMD /start.sh