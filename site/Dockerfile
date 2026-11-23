FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 SITE_MODE=local
# Install CPU PyTorch explicitly; avoid downloading CUDA libraries on CPU hosts.
RUN pip install --no-cache-dir "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt requirements-local.txt ./
RUN pip install --no-cache-dir -r requirements-local.txt
COPY . .
EXPOSE 8000
CMD ["sh", "-c", "python app.py --host 0.0.0.0 --port ${PORT:-8000} --threads 4"]
