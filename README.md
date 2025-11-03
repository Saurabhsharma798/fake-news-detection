Fake News Detection using Deep Learning (GRU & RNN)
📘 Overview

This project detects fake and real news articles using two deep learning models —
GRU (Gated Recurrent Unit) and Simple RNN (Recurrent Neural Network).

Both models are trained on a dataset of real and fake news headlines and texts.
At the end, their performance is compared to understand how each model behaves with text data.

⚙️ Steps in the Code (Explained Simply)
1️⃣ Import Libraries and Load Dataset

We start by importing Python libraries like Pandas, TensorFlow, and Scikit-learn.
Then we load two CSV files:

Fake.csv → contains fake news articles

True.csv → contains real news articles

A new column label is added:

0 → Fake

1 → Real

Finally, both files are combined into one big dataset and shuffled randomly.

2️⃣ Data Inspection and Cleaning

We check:

The number of rows and columns

Whether any values are missing

Then we clean the text using the clean_text() function:

Remove special characters and numbers

Convert text to lowercase

Remove stopwords like the, and, is

Remove names of news agencies like BBC, Reuters

This step keeps only the meaningful words.

3️⃣ Combine Title and Article Text

The news title and main text are merged together.
This helps the model learn from both headline and content.

4️⃣ Tokenization and Padding

Neural networks can’t read words, so we convert each word into a number (token).
We then make all news articles the same length by padding shorter ones with zeros.
This makes sure every input to the model is of the same size (300 words).

5️⃣ Train–Test Split

We divide the data into:

Training set (80%) → used to teach the model

Testing set (20%) → used to check how well the model performs on unseen data

6️⃣ GRU Model

A GRU (Gated Recurrent Unit) is a smart neural network that can remember important words and forget unimportant ones while reading a sentence.

GRU architecture used:

Embedding layer (to convert words into numeric vectors)

GRU layer (128 units)

Dense layers with ReLU activation

Dropout layers (to prevent overfitting)

Output layer with sigmoid activation (gives output between 0 and 1)

We compile it with binary cross-entropy loss and Adam optimizer.

7️⃣ Training GRU Model

We train the GRU model for 5 epochs with:

Batch size = 64

Validation split = 10%

During training, we track both training and validation accuracy to make sure the model is learning properly.

8️⃣ Simple RNN Model

We also build a Simple RNN model.
It’s similar to GRU but much simpler — it doesn’t have memory gates.
This means it can’t remember long-term information as effectively.

Architecture:

Embedding layer

SimpleRNN layer (128 units)

Dense + Dropout layers

Sigmoid output

9️⃣ RNN Training and Evaluation

The RNN model is trained the same way as GRU.
After training, we check the test accuracy.
Usually, RNN performs worse (around 50–60%) because it forgets earlier context in long text.

🔟 Model Comparison

Finally, we compare GRU and RNN on:

Accuracy

F1-score

Confusion Matrix (optional visualization)

GRU is usually much better because it handles long sequences and dependencies better.

📊 Expected Results
Model	Accuracy	Comment
GRU	~90–94%	Learns long-term patterns effectively
RNN	~55–65%	Forgets older context, lower accuracy

Conclusion:
GRU outperforms RNN because it remembers important information across longer text sequences.

🧩 File Summary
File	Description
Fake.csv	Fake news dataset
True.csv	Real news dataset
fake_news_predictor.py	Full Python script with GRU and RNN training
GRU_Model.h5	Saved trained GRU model
tokenizer.pkl	Tokenizer file for converting text to tokens
🧠 Common Viva / Interview Questions (with Answers)

Q1. What is the main goal of this project?
A: To detect whether a news article is fake or real using deep learning models.

Q2. What dataset did you use?
A: The Kaggle “Fake and Real News Dataset,” containing around 44,000 news articles.

Q3. Why did you use GRU and RNN?
A: Both are sequence models that can process text word by word. GRU is an improved version of RNN that handles long-term dependencies better.

Q4. Why does RNN give lower accuracy than GRU?
A: RNN forgets earlier words due to the vanishing gradient problem, while GRU uses gates to remember important information.

Q5. What does the Embedding layer do?
A: It converts words into dense numeric vectors that represent their meaning.

Q6. What is padding and why is it used?
A: Padding makes all text inputs the same length by adding zeros, which is required for batch processing in neural networks.

Q7. What activation function is used in the last layer?
A: Sigmoid, because we have a binary classification (Fake = 0, Real = 1).

Q8. What is dropout and why do we use it?
A: Dropout randomly disables neurons during training to prevent overfitting and make the model more general.

Q9. What optimizer and loss function did you use?
A: Adam optimizer with binary cross-entropy loss.

Q10. What did you conclude from this project?
A: GRU achieved higher accuracy and generalization than RNN, proving that gated recurrent units are more effective for text classification tasks like fake news detection.

✅ Summary

This project demonstrates how deep learning models process natural language to detect fake news.
Even though both RNN and GRU belong to the same family, GRU performs better because it can remember context, forget irrelevant data, and train faster.