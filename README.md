## Chatbot API
---

### Endpoint API
**POST**  
`https://jobrise-chatbot-ml-production.up.railway.app/predict`

---

### 📥 Request Body (JSON)

```json
{
  "text": "Tulis pertanyaanmu di sini"
}

{
  "user_input": "Tulis pertanyaanmu di sini",
  "matched_question": "Pertanyaan yang paling mirip",
  "matched_intent": "Intent ID",
  "similarity_score": 0.48,
  "response": "Jawaban dari chatbot"
}

