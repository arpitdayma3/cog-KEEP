# Base image from RunPod - choose one with PyTorch and a suitable CUDA version
# Example: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel or similar
# Let's try to find one closer to Python 3.11 if easily available, otherwise 3.10 is fine.
# For now, let's use a known stable one.
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install pget for faster downloads
RUN curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/latest/download/pget_$(uname -s)_$(uname -m)" && \
    chmod +x /usr/local/bin/pget

# Set up environment variables for model caching (similar to predict.py)
ENV MODEL_CACHE="/app/model_cache"
ENV HF_HOME="/app/model_cache"
ENV TORCH_HOME="/app/model_cache"
ENV HF_DATASETS_CACHE="/app/model_cache"
ENV TRANSFORMERS_CACHE="/app/model_cache"
ENV HUGGINGFACE_HUB_CACHE="/app/model_cache"
RUN mkdir -p /app/model_cache

# Copy the entire repository contents into the Docker image
COPY . .

# Install Python packages
# Ensure requirements.txt will be updated to include runpod and yt-dlp later
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose the port RunPod expects (if necessary, though usually handled by RunPod)
# CMD line will be provided by RunPod when creating the serverless endpoint,
# typically to run the handler script (e.g., python -u handler.py)

CMD ["python", "-u", "handler.py"]
