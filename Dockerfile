# Image for local development with reflex
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Install dependencies for psutil
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    build-essential

# Install app requirements and reflex inside virtualenv
COPY requirements.txt .
# COPY .env .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the start_reflex.sh script to /usr/local/bin so it's in the PATH
COPY start_reflex.sh /usr/local/bin/start_reflex.sh
RUN chmod +x /usr/local/bin/start_reflex.sh

# Cleanup to reduce image size
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
