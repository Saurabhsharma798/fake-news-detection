# 📰 Fake News Detection using Deep Learning (GRU vs RNN)

### 📘 Overview
This project focuses on **detecting fake and real news articles** using **Deep Learning models** —
specifically **Gated Recurrent Unit (GRU)** and **Simple Recurrent Neural Network (RNN)**.

Both models are trained on a dataset of real and fake news articles from Kaggle.
The main goal is to compare how each model performs in understanding text sequences and classifying news correctly.

---

## ⚙️ Project Workflow

### 1️⃣ Data Loading
We use two datasets:
- **Fake.csv** → Fake news articles
- **True.csv** → Real news articles

Each dataset is labeled:
- `0` → Fake
- `1` → Real

Both files are merged into one DataFrame and shuffled randomly to avoid bias.

### 2️⃣ Text Cleaning
A custom function `clean_text()` removes punctuation, numbers, stopwords, and agency names.
This keeps only meaningful words.

### 3️⃣ Combining Title and Text
The `title` and `text` columns are merged to help the model learn from both.

### 4️⃣ Tokenization and Padding
Words are converted into numbers using Keras **Tokenizer** and padded to a fixed length (300 tokens).

### 5️⃣ Train-Test Split
80% training data, 20% testing data with stratification for balance.

### 6️⃣ GRU Model
A GRU can remember important words and forget irrelevant ones.

```python
GRU_model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    GRU(128, dropout=0.3, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

### 7️⃣ Simple RNN Model
A basic RNN processes words in order but cannot remember long context.

```python
RNN_model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    SimpleRNN(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

### 8️⃣ Model Evaluation
Both models are compared using accuracy and F1-score.

```python
from sklearn.metrics import accuracy_score, f1_score

y_pred_gru = (GRU_model.predict(X_test) > 0.5).astype(int)
y_pred_rnn = (RNN_model.predict(X_test) > 0.5).astype(int)

print("GRU Accuracy:", accuracy_score(y_test, y_pred_gru))
print("RNN Accuracy:", accuracy_score(y_test, y_pred_rnn))
```

---

## 📊 Results Summary

| Model | Accuracy | Comment |
|--------|-----------|----------|
| **GRU** | ~92–94% | Retains long-term context |
| **RNN** | ~55–65% | Struggles with long text |

---

## ❓ Common Viva Questions

**Q1:** Why GRU over RNN?  
🟢 GRU uses gates to remember important information; RNN forgets earlier context.

**Q2:** What is embedding?  
🟢 Converts words into dense vectors to capture relationships.

**Q3:** What activation function is used?  
🟢 Sigmoid for binary classification (Fake vs Real).

**Q4:** Why is dropout used?  
🟢 To prevent overfitting.

**Q5:** What is the main conclusion?  
🟢 GRU gives higher accuracy and handles text sequences better than RNN.

---

## ▶️ How to Run

### 🔹 Google Colab
1. Upload `Fake.csv`, `True.csv`, and `fake_news_predictor.py`.
2. Run all cells.
3. Compare GRU and RNN accuracy.

### 🔹 Local Machine
```bash
pip install tensorflow pandas scikit-learn nltk matplotlib
python fake_news_predictor.py
```

---

## 🚀 Future Work
- Add **LSTM** or **BERT** for deeper comparison.
- Use **GloVe embeddings** for semantic meaning.
- Deploy using **Streamlit** or **FastAPI** for real-time predictions.

---

## ✨ Summary
This project demonstrates fake news detection using GRU and RNN.  
GRU clearly outperforms RNN, proving its effectiveness in capturing long-term dependencies in text data.