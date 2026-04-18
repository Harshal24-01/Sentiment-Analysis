import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from preprocess import clean_text

# ---------------------------
# LOAD DATA
# ---------------------------
try:
    df1 = pd.read_csv("../data/data_analysis.csv", low_memory=False)
    df2 = pd.read_csv("../data/data_science.csv", low_memory=False)
    df3 = pd.read_csv("../data/data_visualization.csv", low_memory=False)

    df = pd.concat([df1, df2, df3], ignore_index=True)

except Exception as e:
    print("❌ Error loading dataset:", e)
    exit()

# ---------------------------
# CHECK COLUMN
# ---------------------------
if 'tweet' not in df.columns:
    print("❌ 'tweet' column not found")
    print("Available columns:", df.columns)
    exit()

# Rename for consistency
df = df.rename(columns={'tweet': 'text'})

# ---------------------------
# CREATE LABELS
# ---------------------------
print("⚠️ Creating sentiment labels...")

def create_label(text):
    positive_words = ['good', 'great', 'love', 'awesome', 'best', 'amazing', 'happy']
    negative_words = ['bad', 'worst', 'hate', 'poor', 'terrible', 'sad']

    text = str(text).lower()

    if any(word in text for word in positive_words):
        return "positive"
    elif any(word in text for word in negative_words):
        return "negative"
    else:
        return "neutral"

df['sentiment'] = df['text'].apply(create_label)

# ---------------------------
# REMOVE NEUTRAL (IMPORTANT)
# ---------------------------
df = df[df['sentiment'] != 'neutral']

# ---------------------------
# CLEAN TEXT
# ---------------------------
df['clean_text'] = df['text'].apply(clean_text)
df = df[df['clean_text'].str.strip() != ""]

# ---------------------------
# BALANCE DATASET (SAFE)
# ---------------------------
print("⚖️ Balancing dataset...")

positive_df = df[df['sentiment'] == 'positive']
negative_df = df[df['sentiment'] == 'negative']

min_count = min(len(positive_df), len(negative_df))

positive_df = positive_df.sample(min_count, random_state=42)
negative_df = negative_df.sample(min_count, random_state=42)

df = pd.concat([positive_df, negative_df]).reset_index(drop=True)

print("Class distribution after balancing:")
print(df['sentiment'].value_counts())

# ---------------------------
# FEATURES & LABELS
# ---------------------------
X = df['clean_text']
y = df['sentiment']

# ---------------------------
# TF-IDF
# ---------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# ---------------------------
# SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# ---------------------------
# MODEL
# ---------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ---------------------------
# SAVE MODEL
# ---------------------------
joblib.dump(model, "../model/model.pkl")
joblib.dump(vectorizer, "../model/vectorizer.pkl")

print("✅ Model saved successfully!")

# ---------------------------
# EVALUATION
# ---------------------------
y_pred = model.predict(X_test)

print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))