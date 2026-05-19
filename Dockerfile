# Base image — matches your Python version
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (Docker layer caching — faster rebuilds)
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project into container
COPY . .

# Make start.sh executable
RUN chmod +x start.sh

# Expose Streamlit port
EXPOSE 8501

# Run start.sh when container starts
CMD ["./start.sh"]