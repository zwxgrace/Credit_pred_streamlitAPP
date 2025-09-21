FROM python:3.8.17-slim

WORKDIR /CREDIT_PREDICTION_STREAMLITAPP

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "pred_catboost.py", "--server.port=8501", "--server.address=0.0.0.0"]