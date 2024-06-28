# Base image of Python application
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set pip default timeout to 1000 seconds
ENV PIP_DEFAULT_TIMEOUT=1000

# Install necessary packages
RUN apt-get update && \
    apt-get install -y build-essential python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirement file
COPY ./requirements.txt /app/requirements.txt

# Install pip, setuptools, and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install numpy and cython first
RUN pip install --no-cache-dir numpy==1.23.1 cython==0.29.24

# Install remaining dependencies
RUN pip install --no-cache-dir --prefer-binary -r /app/requirements.txt

# Copy app and models
COPY ./app /app
COPY ./model /model

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
