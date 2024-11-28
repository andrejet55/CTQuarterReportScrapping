# Use a minimal slim image with Debian-based Python
FROM python:3.12-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy project files to the container
COPY . /app


# Copy and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Expose the Gradio server port
EXPOSE 7860

# Command to run your application
CMD ["python", "app.py"]
