#!/usr/bin/env bash
# TODO: took from another runpod serverless repo, check if helps with performance at all

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

python3 -u /rp_handler.py
