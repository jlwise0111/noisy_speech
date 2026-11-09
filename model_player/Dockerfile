# Build context is the project folder (noisy_speech/), so the image can carry the project's own
# packages: the models run with the same code that trained and scored them.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY model_player/requirements.txt model_player/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r model_player/requirements.txt

COPY speechprep speechprep
COPY unet_model/unet unet_model/unet
COPY linear_baseline/lrb linear_baseline/lrb
COPY wiener_baseline/wiener wiener_baseline/wiener
COPY diagnostic_viewer/viewer diagnostic_viewer/viewer
COPY model_player model_player

WORKDIR /app/model_player
# plotly.js as a static file (browser-cached), as in the diagnostic viewer.
RUN mkdir -p static && python -c "from plotly.offline import get_plotlyjs; open('static/plotly.min.js','w').write(get_plotlyjs())"

# Trained runs are mounted at /models and the Valentini dataset at /dataset (see docker-compose.yml).
ENV PYTHONUNBUFFERED=1 \
    PLAYER_CODE_ROOT=/app \
    PLAYER_MODELS_DIR=/models \
    PLAYER_DATASET_DIR=/dataset

EXPOSE 8502
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8502", "--server.headless=true", "--server.enableStaticServing=true", "--browser.gatherUsageStats=false", "--server.maxMessageSize=1000", "--server.maxUploadSize=200"]
