# Base stage
FROM runpod/base:0.6.2-cuda12.2.0 AS base

LABEL authors="jaret"

# Install basic dependencies
RUN apt-get update

# Set up Python environment
RUN ln -s /usr/bin/python3 /usr/bin/python

# Clone the repository with specific commit
WORKDIR /app
ARG CACHEBUST=1
RUN git clone https://github.com/ostris/ai-toolkit.git && \
    cd ai-toolkit && \
    git checkout ffaf2f1 && \
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
COPY tests/ /workspace/ai-toolkit/tests/
COPY requirements-dev.txt /workspace/ai-toolkit/

# Install Python dependencies
WORKDIR /workspace/ai-toolkit
RUN python -m pip install --no-cache-dir -r requirements.txt
# FIXME: debugpy if dev
RUN python -m pip install --no-cache-dir requests runpod loki_logger_handler==1.1.1 debugpy

# Install development dependencies if DEV_MODE is set
ARG DEV_MODE=false
RUN if [ "$DEV_MODE" = "true" ] ; then python -m pip install --no-cache-dir -r requirements-dev.txt ; fi

# Set command
CMD ["python", "/workspace/ai-toolkit/rp_handler.py"]
