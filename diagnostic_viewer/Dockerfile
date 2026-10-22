FROM python:3.12-slim

# build-essential: pesq compiles a C extension; libsndfile1: soundfile backend.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Serve plotly.js as a static file (cached by the browser) instead of inlining
# ~3.5 MB into the player iframe on every rerun.
RUN mkdir -p static && python -c "from plotly.offline import get_plotlyjs; open('static/plotly.min.js','w').write(get_plotlyjs())"

ENV PYTHONUNBUFFERED=1 \
    VIEWER_DATA_DIR=/data

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--server.enableStaticServing=true", "--browser.gatherUsageStats=false", "--server.maxMessageSize=1000"]
