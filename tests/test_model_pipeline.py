"""Exemple de test unitaire."""

from sklearn.datasets import make_classification

from model_pipeline import train_model


def test_train_model_returns_fitted_model():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model = train_model(X, y)
    assert hasattr(model, "predict")
    assert len(model.predict(X)) == len(y)
