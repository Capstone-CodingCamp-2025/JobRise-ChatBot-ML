FROM pytorch/pytorch:2.0.1-cpu

WORKDIR /app

COPY requirements.txt .
COPY main.py .
COPY embeddings.pkl .
COPY Combined_Chatbot_Dataset.csv .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
