def get_top_keywords(model, vectorizer, n=10):
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]

    top_positive = sorted(zip(coefs, feature_names), reverse=True)[:n]
    top_negative = sorted(zip(coefs, feature_names))[:n]

    return top_positive, top_negative