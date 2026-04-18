from preprocess import clean_text

def predict_sentiment(text, model, vectorizer):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized).max()

    return prediction, probability