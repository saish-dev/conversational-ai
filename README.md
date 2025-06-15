# 🧠 FastAPI NLP Microservice for Intent Classification

This project is a FastAPI-based microservice for **domain-aware intent classification** using a fine-tuned RoBERTa model. It is built to support **multi-tenant** scenarios where each customer can have their own training data and domain-specific intents.

---

## 📂 Project Structure

```
nlp-service/
│
├── app/
│   ├── api/              # FastAPI endpoints (e.g., /predict)
│   ├── core/             # Core logic: training, inference, dataset utils
│   ├── models/           # Trained model artifacts (saved locally or via Azure)
│   └── main.py           # FastAPI app entrypoint
│
├── config/
│   └── settings.py       # Environment variable and settings management
│
├── Dockerfile            # Docker image for deployment
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
└── README.md             # Project documentation
```

---

## 🚀 Features

- ⚙️ **FastAPI** RESTful endpoints for real-time prediction  
- 🤖 **RoBERTa-based** intent classification  
- 🏷️ **Multi-tenant** support via domain-based tagging  
- ⬆️ Upload datasets per customer (CSV format)  
- 💾 Load/save models and label encoders dynamically  
- ☁️ Azure Blob Storage support for remote model/dataset storage  
- 🔁 Extendable for future generative AI use-cases  

---

## 🧪 API Endpoints

### 📥 `POST /predict`

Predicts the intent of a given user query based on a specified domain.

**Request Body**:
```json
{
  "text": "I need to reset my password",
  "domain": "support"
}
```

**Response**:
```json
{
  "intent": "reset_password"
}
```

---

## 📊 Dataset Format

CSV file with 3 columns:
```csv
text,intent,domain
"I forgot my password",reset_password,support
"I want a refund",request_refund,billing
```

> Each dataset should be tagged with a domain to enable multi-tenant training.

---

## ⚙️ Environment Variables (`.env`)

```env
MODEL_PATH=models/roberta
DATASET_PATH=dataset/combined_dataset.csv
LOGGING_PATH=models/logs
LABEL_ENCODER_PATH=models/roberta/label_encoder.pkl
ENVIRONMENT = "local"
```

## 🐳 Docker Setup

### ✅ First-Time Setup (or after changes to `Dockerfile` / `requirements.txt`)

```bash
docker network create conversational-ai-network
```

```bash
docker compose up -d --build
```

This will:
- Build the Docker image
- Start FastAPI Service


### 🚀 Regular Start

```bash
docker compose up -d
```

### ⛔️ Stop Containers

```bash
docker compose down
```


## 📬 Example Request

```bash
curl -X POST http://localhost:8000/predict-intent   -H "Content-Type: application/json"   -d '{"text": "Book a cab", "domain": "transport"}'
```

---

## 📘 Swagger Documentation
FastAPI automatically provides interactive API documentation powered by Swagger UI.

### 🌐 Open Swagger UI
Once your FastAPI service is running (either locally or in a Docker container), you can access the Swagger docs at:

http://localhost:5000/docs
This interface allows you to:

View all available API endpoints

See example request/response formats

Send test requests directly from the browser

🧪 Alternative: ReDoc
FastAPI also includes another documentation interface at:

http://localhost:8000/redoc
This version provides a more structured overview of your API schema.

---

## 🛠️ Notes

- Use `--build` when updating Docker dependencies.
- Don't forget to configure your `.env` file correctly for both Docker and local development.

---

## 📬 Contact & Support

Open an issue or PR to contribute, report bugs, or ask questions.

Happy developing! 🎉