# Use Python 3.11.9 base image
FROM python:3.11.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project
COPY . .

# Set environment PORT for Render
ENV PORT=8000
EXPOSE $PORT

# Start Django using gunicorn
CMD ["gunicorn", "backend.wsgi:application", "--bind", "0.0.0.0:8000"]
