FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/evaluation classification/models

# Set environment variables
ENV FLASK_APP=web.app
ENV PYTHONPATH=/app

# Expose port
EXPOSE 5000

# Run command
CMD ["python", "scripts/run_webapp.py"]