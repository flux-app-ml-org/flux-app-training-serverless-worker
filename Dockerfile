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

# Copy application files
WORKDIR /workspace
COPY --from=base /app/ai-toolkit /workspace/ai-toolkit
COPY rp_handler.py test_input.json /workspace/ai-toolkit/
COPY requirements.txt /workspace/ai-toolkit/additional_requirements.txt  # Copy the requirements.txt file

# Install Python dependencies
WORKDIR /workspace/ai-toolkit
RUN python -m pip install --no-cache-dir -r requirements.txt
RUN python -m pip install --no-cache-dir -r additional_requirements.txt

# Set command
CMD ["python", "/workspace/ai-toolkit/rp_handler.py"]
