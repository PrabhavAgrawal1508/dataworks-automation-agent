# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app
RUN apt-get update && apt-get install -y git
# Copy only the requirements file first (for efficient Docker caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8000
EXPOSE 8000

# Define the environment variable for AIPROXY_TOKEN
ENV AIPROXY_TOKEN=""

# Run the FastAPI/Flask app (replace main:app accordingly)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

