FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.docker.txt .

RUN pip install --no-cache-dir torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.docker.txt
RUN pip install --no-cache-dir pytest

COPY . .

EXPOSE 8000

CMD ["python", "-m", "src.service.cli", "--host", "0.0.0.0", "--port", "8000"]