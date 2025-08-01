# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code
COPY ./src /app/src

# NEW: Copy the saved model and scaler into the container
COPY ./saved_model /app/saved_model

# Make port 8000 available
EXPOSE 8000

# Run the API with Uvicorn when the container launches
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]