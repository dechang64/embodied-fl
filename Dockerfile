# ── Stage 1: Rust backend ──
FROM rust:1.82-slim AS rust-builder

WORKDIR /build
COPY . .
RUN cargo build --release 2>/dev/null || true

# ── Stage 2: Python runtime ──
FROM python:3.12-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash appuser

# Python dependencies
COPY python/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python application
COPY python/ .

# Copy Rust binary (if built)
COPY --from=rust-builder /build/target/release/embodied-fl /app/bin/embodied-fl 2>/dev/null || true

# Change ownership
RUN chown -R appuser:appuser /app

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run as non-root user
USER appuser

# Run Streamlit
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
