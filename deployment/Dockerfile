# Use Python 3.11 (Compatible with CatBoost)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Project Files
COPY app/ ./app/
COPY model/ ./model/
COPY frontend/ ./frontend/
COPY data/ ./data/

# Expose Port
EXPOSE 8000

# Start the App
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]