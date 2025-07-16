# 🛡️ Spam Detection Web App

A full-stack web application to classify messages as **Spam** or **Not Spam** using a machine learning model.  
Built with a **Python backend** for ML inference and a modern **React.js (Vite + Tailwind CSS)** frontend.

---

## ⚙️ Tech Stack

### 🚀 Frontend
- [React.js](https://reactjs.org/) (bootstrapped with [Vite](https://vitejs.dev/))
- [Tailwind CSS](https://tailwindcss.com/) for utility-first styling
- Axios for API communication

### 🧠 Backend
- Python (Flask or FastAPI)
- Scikit-learn for spam classification
- Pickled ML model (`model.pkl`)
- CORS enabled for frontend-backend interaction

---

## 🧠 How It Works

1. User enters a message in the frontend.
2. Text is sent to the backend via `POST` request.
3. Backend loads the ML model and returns a prediction.
4. Frontend displays if the message is `Spam` or `Not Spam`.

---

## 🏗️ Project Structure

```
spam-detection-app/
├── backend/
│   ├── app.py              # FastAPI app
│   ├── model.pkl           # Trained ML model
│   └── requirements.txt    # Python dependencies
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── main.jsx
    │   └── components/
    ├── tailwind.config.js
    ├── index.html
    └── vite.config.js
```

---

## 🔧 Setup Instructions

### 🔙 Backend (Python)

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py                 # or: uvicorn app:app --reload (FastAPI)
```

### 🖥️ Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

---

## 🔌 API Endpoint

### POST /predict

**Request Body**:
```json
{
  "message": "You won a free iPhone!"
}
```

**Response**:
```json
{
  "prediction": "Spam"
}
```

---


## 🤝 Credits

Built collaboratively by:
- Nimit Gupta
- Sanjeevni Dhir

