# === Stage 1: Build Dependencies ===
FROM python:3.9-slim AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/root/.local/bin:$PATH"

# Install only necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app
COPY requirements.txt . 
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy model preparation script and run it
COPY prepare.py .  
RUN python3 prepare.py

# === Stage 2: Minimal Runtime Image ===
FROM python:3.9-slim

# Set environment variables
ENV PATH="/root/.local/bin:$PATH"

# Copy only required dependencies from builder
COPY --from=builder /root/.local /root/.local

# Set work directory
WORKDIR /app

# Copy the model and class mapping
COPY --from=builder /app/model.h5 /app/model.h5
COPY --from=builder /app/class.json /app/class.json

# Copy only necessary application files
COPY app.py .  

# Expose Streamlit port
EXPOSE 7860

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
