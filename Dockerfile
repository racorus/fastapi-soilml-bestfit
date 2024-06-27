# Base image of Python application
FROM python:3.11

# Set working directory
WORKDIR /app

# Set pip default timeout to 1000 seconds
ENV PIP_DEFAULT_TIMEOUT=1000


# Install necessary packages
RUN apt-get update && \
    apt-get install -y build-essential python3-pip python3-setuptools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Copy requirement file
COPY ./requirements.txt /app/requirements.txt

# Install pip, setuptools, and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install requirements packages
RUN pip install --no-cache-dir --upgrade --prefer-binary -r ./requirements.txt

# Copy app and models
COPY ./app /app
COPY ./model /app/model
COPY ./samples_training_data /app/samples_training_data

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

