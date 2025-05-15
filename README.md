## JANGAN DITARUH FILE APAPUN SELAIN DARI YANG SINI 

## DILARANG MENGEDIT (KARENA SANGAT SENSITIF)

## Chatbot API
---

### 🧠 Endpoint
**POST**  
`https://machine-learning-production.up.railway.app/predict`

---

### 📥 Request Body (JSON)

```json
{
  "text": "Tulis pertanyaanmu di sini"
}

``` json
{
  "user_input": "Tulis pertanyaanmu di sini",
  "matched_question": "Pertanyaan yang paling mirip",
  "matched_intent": "Intent ID",
  "similarity_score": 0.48,
  "response": "Jawaban dari chatbot"
}

