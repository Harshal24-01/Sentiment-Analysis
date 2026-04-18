# 🧠 Sentiment Analysis App

An end-to-end **Machine Learning + NLP project** that analyzes tweet text and classifies it into **Positive** or **Negative** sentiment with confidence scores and insights.

---

## 🚀 Features

* 🔍 Sentiment Prediction (Positive / Negative)
* 📊 Confidence Score with Visualization
* 🧠 TF-IDF + Logistic Regression Model
* 🧹 Text Preprocessing using NLP
* 📈 Keyword Insights (Top Positive & Negative Words)
* 🎨 Interactive UI built with Streamlit

---

## 📂 Project Structure

```
sentiment-analysis/
│
├── data/                 # Dataset files
│   ├── data_analysis.csv
│   ├── data_science.csv
│   └── data_visualization.csv
│
├── model/                # Saved ML model
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── src/                  # Core ML code
│   ├── preprocess.py
│   ├── train_model.py
│   ├── predict.py
│   └── keywords.py
│
├── app/                  # Streamlit UI
│   └── app.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

* **Language:** Python 3
* **Libraries:**

  * pandas
  * numpy
  * scikit-learn
  * nltk
  * streamlit
  * joblib

---

## 🧪 Model Details

* **Algorithm:** Logistic Regression
* **Feature Extraction:** TF-IDF Vectorization
* **Preprocessing:**

  * Lowercasing
  * Removing URLs, mentions, special characters
  * Stopword removal

---

## 📊 Dataset

Dataset sourced from Kaggle:

👉 https://www.kaggle.com/ruchi798/data-science-tweets

* Multiple CSV files merged
* Sentiment labels created using keyword-based approach
* Converted to **binary classification (Positive / Negative)**
* Balanced dataset using downsampling

---

## 📈 Model Performance

* Accuracy: **~94–95%**
* Balanced precision & recall for both classes

---

## ▶️ How to Run the Project

### 1. Clone Repository

```
git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis
```

---

### 2. Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

### 4. Train Model

```
cd src
python train_model.py
```

---

### 5. Run Streamlit App

```
cd ../app
streamlit run app.py
```

---

## 🧠 How It Works

1. User inputs a tweet
2. Text is cleaned and preprocessed
3. TF-IDF converts text into numerical features
4. Trained model predicts sentiment
5. App displays:

   * Sentiment result
   * Confidence score
   * Keyword insights

---

## 🎯 Example

Input:

```
I love this project, it is amazing
```

Output:

```
Positive Sentiment 😊
Confidence: 1.00
```

---

## ⚠️ Limitations

* Uses keyword-based labeling (not human-labeled data)
* May struggle with sarcasm or complex sentences
* Not using deep learning (BERT, LSTM)

---

## 🔥 Future Improvements

* Add Neutral sentiment class
* Use pre-labeled dataset
* Implement Deep Learning (BERT)
* Deploy online (Streamlit Cloud)
* Add WordCloud & advanced visualizations

---

## 👨‍💻 Author

**Harshal Suryawanshi**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
