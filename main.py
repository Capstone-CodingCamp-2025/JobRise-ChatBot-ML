from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import torch
from sentence_transformers import util

app = FastAPI()
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

questions = data["questions"]
answers = data["answers"]
intents = data["intents"]
question_embeddings = data["question_embeddings"]

class Query(BaseModel):
    text: str

@app.post("/predict")
def predict(q: Query):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')  
    user_embedding = model.encode(q.text, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
    best_idx = torch.argmax(cosine_scores).item()
    best_score = cosine_scores[best_idx].item()

    return {
        "user_input": q.text,
        "matched_question": questions[best_idx],
        "matched_intent": intents[best_idx],
        "similarity_score": round(best_score, 3),
        "response": answers[best_idx]
    }
