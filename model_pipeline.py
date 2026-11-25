"""Fonctions du pipeline ML (données, entraînement, sauvegarde, évaluation)."""

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def prepare_data(test_size=0.2, random_state=42):
    """Charge le dataset et retourne X_train, X_test, y_train, y_test."""
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train, y_train):
    """Entraîne un RandomForest."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def save_model(model, path="model.joblib"):
    """Sauvegarde le modèle sur disque."""
    joblib.dump(model, path)


def load_model(path="model.joblib"):
    """Charge le modèle depuis le disque."""
    return joblib.load(path)


def evaluate_model(model, X_test, y_test):
    """Retourne les métriques du modèle."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }
