FROM python:3.14-slim

# Security: run as non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1001 appuser

WORKDIR /app

# System deps — curl only for healthcheck, no dev tools in final image
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Python deps — installed as root before dropping privileges
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r requirements.txt && \
    pip install --no-cache-dir python-jose[cryptography] passlib[bcrypt] slowapi bleach cryptography && \
    # Remove pip cache
    pip cache purge

# App code
COPY . .

# Create models dir with correct ownership
RUN mkdir -p /app/models /app/data && \
    chown -R appuser:appgroup /app && \
    # Remove any .env files that may have been accidentally copied
    rm -f /app/.env /app/.env.local /app/.env.production

# Drop to non-root
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run scoring service
EXPOSE 8000 8050
CMD ["python", "main.py", "scoring-service"]
