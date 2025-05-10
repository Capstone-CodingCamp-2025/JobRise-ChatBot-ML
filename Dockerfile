# Gunakan image dasar python yang ringan
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt
COPY main.py main.py
COPY embeddings.pkl embeddings.pkl
COPY Combined_Chatbot_Dataset.csv Combined_Chatbot_Dataset.csv

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
