#!/bin/bash

# Install required packages if needed
pip install -r requirements.txt

# Set environment variables to limit memory usage
export PYTHONMALLOC=malloc
export MALLOC_TRIM_THRESHOLD_=65536
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1000  # Allow uploads up to 1GB
export OMP_NUM_THREADS=1  # Limit OpenMP threads
export RASTERIO_CACHE_SIZE=128  # 128MB rasterio cache

# Check available memory
FREE_MEM=$(free -g | awk '/^Mem:/{print $2}')
echo "System has approximately ${FREE_MEM}GB of total memory"

# Calculate memory limit (70% of available memory)
MEM_LIMIT=$((FREE_MEM * 70))
echo "Setting memory limit to ${MEM_LIMIT}00MB"

# Run with memory limit
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "Running on Linux with memory limits"
    ulimit -v $((MEM_LIMIT * 100 * 1024))  # Virtual memory limit in KB
    streamlit run app.py
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Running on macOS"
    streamlit run app.py
else
    # Windows
    echo "Running on Windows"
    streamlit run app.py
fi