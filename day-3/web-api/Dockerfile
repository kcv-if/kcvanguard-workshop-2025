# Use official Python image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY prepare.py .  

RUN python3 prepare.py

# Copy application files
COPY app.py .

# Expose FastAPI default port
EXPOSE 7860

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]