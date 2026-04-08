FROM python:3.11-slim

# Metadata
LABEL name="icde" \
      version="1.0.0" \
      description="Incident Command Decision Environment — OpenEnv benchmark" \
      tags="openenv,incident-command,emergency-management"

# HF Spaces expects port 7860
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Ensure __init__ files exist
RUN touch env/__init__.py graders/__init__.py || true

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# ✅ OpenEnv + HF compliant entrypoint
CMD ["python", "app.py"]
