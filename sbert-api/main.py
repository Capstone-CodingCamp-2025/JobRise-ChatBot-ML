from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv("Combined_Chatbot_Dataset.csv")
questions = df['text'].tolist()
answers = df['answer'].tolist()
intents = df['intent'].tolist()

question_embeddings = model.encode(questions, convert_to_tensor=True)

class Query(BaseModel):
    text: str

@app.post("/predict")
def predict(q: Query):
    user_embedding = model.encode(q.text, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]

    best_idx = cosine_scores.argmax().item()
    best_score = cosine_scores[best_idx].item()

    return {
        "user_input": q.text,
        "matched_question": questions[best_idx],
        "matched_intent": intents[best_idx],
        "similarity_score": round(best_score, 3),
        "response": answers[best_idx]
    }
