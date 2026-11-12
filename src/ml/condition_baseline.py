"""Convert symptom text to TF-IDF features and fit logistic regression."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.ml.model_selection import select_model


def train_baseline(training_text, validation_text, training_labels, validation_labels, weights):
    """Fit vocabulary and classifier on training data, then select C on validation data."""
    def make_pipeline(regularization):
        return Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
            ('clf', LogisticRegression(C=regularization, max_iter=2000, random_state=42)),
        ])

    return select_model(
        make_pipeline, training_text, validation_text, training_labels, validation_labels,
        weights, pipeline=True,
    )
