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

# Download required NLTK data
nltk.download("stopwords")
nltk.download("wordnet")

# ========== MongoDB Setup ==========
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["insta_spam_db"]
collection = db["predictions"]

# ========== App Setup ==========
app = FastAPI()

# ========== Load LSTM Model ==========
model = load_model("backend/model/spam_lstm_model.h5")

# ========== Load Tokenizer ==========
with open("backend/model/tokenizer.pkl", "rb") as f:
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

# ========== API Route ==========
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

    # Save to MongoDB
    record = {
        "input": raw_text,
        "cleaned": cleaned_text,
        "prediction": label,
        "probability": float(prediction),
        "timestamp": datetime.utcnow()
    }
    collection.insert_one(record)

    return {
        "input": raw_text,
        "cleaned": cleaned_text,
        "prediction": label,
        "probability": float(prediction)
    }
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default dev server port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
