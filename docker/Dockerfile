# Base stage
FROM runpod/base:0.6.2-cuda12.2.0 AS base

LABEL authors="jaret"

# Install basic dependencies
RUN apt-get update && \
    apt-get install -y tmux nvtop htop

# Set up Python environment
RUN ln -s /usr/bin/python3 /usr/bin/python

# Clone the repository
WORKDIR /app
ARG CACHEBUST=1
RUN git clone https://github.com/ostris/ai-toolkit.git && \
    cd ai-toolkit && \
    git submodule update --init --recursive

# Install Python dependencies
WORKDIR /app/ai-toolkit
RUN python -m pip install -r requirements.txt

# Additional Python packages
RUN pip install jupyterlab runpod requests

# Production stage
FROM runpod/base:0.6.2-cuda12.2.0 AS final

# Copy dependencies from the base stage
COPY --from=base /usr/bin/tmux /usr/bin/tmux
COPY --from=base /usr/bin/nvtop /usr/bin/nvtop
COPY --from=base /usr/bin/htop /usr/bin/htop
COPY --from=base /usr/bin/python /usr/bin/python
COPY --from=base /usr/local/lib/python3.*/dist-packages /usr/local/lib/python3.*/dist-packages

# Copy application files
WORKDIR /workspace
COPY --from=base /app/ai-toolkit /workspace/ai-toolkit
COPY --from=base /app/ai-toolkit/requirements.txt /workspace/ai-toolkit/requirements.txt

# Add custom script
COPY rp_handler.py  test_input.json /workspace/ai-toolkit/

# Mask workspace
# TODO: unmask?
RUN mkdir /workspace

# Set command
CMD ["python", "/workspace/ai-toolkit/rp_handler.py"]
