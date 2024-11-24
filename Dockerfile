# Base stage
FROM runpod/base:0.6.2-cuda12.2.0 AS base

LABEL authors="jaret"

# Install basic dependencies
RUN apt-get update

# Set up Python environment
RUN ln -s /usr/bin/python3 /usr/bin/python

# Clone the repository
WORKDIR /app
ARG CACHEBUST=1
RUN git clone https://github.com/ostris/ai-toolkit.git && \
    cd ai-toolkit && \
    git submodule update --init --recursive

# Production stage
FROM runpod/base:0.6.2-cuda12.2.0 AS final

# Install basic dependencies
RUN apt-get update

# Set up Python environment
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace/ai-toolkit
# Install Python dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

# Additional Python packages
RUN python -m pip install --no-cache-dir runpod requests

# Copy application files
WORKDIR /workspace
COPY --from=base /app/ai-toolkit /workspace/ai-toolkit
COPY rp_handler.py test_input.json /workspace/ai-toolkit/

# Set command
CMD ["python", "/workspace/ai-toolkit/rp_handler.py"]
