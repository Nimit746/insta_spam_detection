from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import pickle
import re
import pymongo
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# Download required NLTK data
nltk.download("stopwords")
nltk.download("wordnet")

# ========== MongoDB Setup (Optional) ==========
try:
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    client = pymongo.MongoClient(
        MONGODB_URI, 
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000
    )
    # Test connection
    client.server_info()
    db = client["insta_spam_db"]
    collection = db["predictions"]
    MONGODB_AVAILABLE = True
    print("✓ MongoDB connected successfully")
except Exception as e:
    print(f"⚠ MongoDB not available: {e}")
    print("⚠ API will work without database logging")
    MONGODB_AVAILABLE = False
    collection = None

# ========== App Setup ==========
app = FastAPI()

# ========== CORS Setup ==========
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Load LSTM Model ==========
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "spam_lstm_model.h5"
model = load_model(str(MODEL_PATH))

# ========== Load Tokenizer ==========
TOKENIZER_PATH = MODEL_DIR / "tokenizer.pkl"
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# ========== Text Preprocessing ==========
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|@\w+|[^A-Za-z\s]", "", text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# ========== Request Schema ==========
class CommentInput(BaseModel):
    text: str

# ========== API Routes ==========
@app.get("/")
def root():
    return {
        "status": "running",
        "mongodb_connected": MONGODB_AVAILABLE
    }

@app.post("/predict")
def predict_spam(data: CommentInput):
    raw_text = data.text
    cleaned_text = preprocess_text(raw_text)
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequences, maxlen=100)

    try:
        prediction = model.predict(padded)[0][0]
        label = "Spam" if prediction >= 0.5 else "Not Spam"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Save to MongoDB only if available
    if MONGODB_AVAILABLE and collection is not None:
        try:
            record = {
                "input": raw_text,
                "cleaned": cleaned_text,
                "prediction": label,
                "probability": float(prediction),
                "timestamp": datetime.utcnow()
            }
            collection.insert_one(record)
        except Exception as e:
            print(f"Failed to save to MongoDB: {e}")

    return {
        "input": raw_text,
        "cleaned": cleaned_text,
        "prediction": label,
        "probability": float(prediction)
    }