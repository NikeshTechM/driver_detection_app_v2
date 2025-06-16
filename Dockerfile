FROM python:3.11

# Set the working directory inside the container
WORKDIR /home/app

# Copy your application files into the container
COPY app.py requirements.txt trained_knn_model.clf /home/app/



# Create necessary directories
RUN mkdir templates static

# Copy the template and static files
COPY templates/index.html /home/app/templates
COPY static/image.png /home/app/static

# Install system dependencies required by OpenCV and others
RUN apt-get update && \
    apt-get install -y build-essential cmake ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Install Python dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Define the entrypoint for the container
ENTRYPOINT ["python", "app.py"]

