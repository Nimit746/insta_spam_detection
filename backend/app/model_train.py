import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Load preprocessed dataset
df = pd.read_csv("backend/data/Cleaned-Final-Dataset.csv")

# Check necessary columns
if 'cleaned_comment' not in df.columns or 'label' not in df.columns:
    raise ValueError("Dataset must have 'cleaned_comment' and 'label' columns.")

# Drop missing or empty values
df.dropna(subset=['cleaned_comment'], inplace=True)
df = df[df['cleaned_comment'].str.strip() != '']

# Features and labels
X = df['cleaned_comment']
y = df['label']

# Tokenization
max_words = 5000
max_len = 100

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=max_len)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42, stratify=y)

# Build LSTM model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2, callbacks=[EarlyStopping(patience=2)])

# Create model directory if it doesn't exist
os.makedirs("backend/model", exist_ok=True)

# Save model and tokenizer
model.save("backend/model/spam_lstm_model.h5")
joblib.dump(tokenizer, "backend/model/tokenizer.pkl")

print("✅ LSTM model and tokenizer saved in 'backend/model/'")
