FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY main.py .
COPY embeddings.pkl .
COPY Combined_Chatbot_Dataset.csv .

RUN pip install --no-cache-dir torch==2.0.1+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
