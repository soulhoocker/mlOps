"""Atelier 3 - Pipeline Prefect : orchestration des étapes code + data/modèle."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from prefect import flow, get_run_logger, task

# ADAPTER ces imports aux noms réels de vos fonctions (model_pipeline.py / main.py)
from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

MODEL_PATH = "model.joblib"
# On ne scanne QUE nos fichiers (sinon qualité/sécurité scannent tout le venv)
CODE_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
TEST_FILE = Path("tests/test_model_pipeline.py")

TEST_TEMPLATE = '''"""Exemple de test unitaire."""

from sklearn.datasets import make_classification

from model_pipeline import train_model


def test_train_model_returns_fitted_model():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    model = train_model(X, y)
    assert hasattr(model, "predict")
    assert len(model.predict(X)) == len(y)
'''


def run_cmd(cmd, check=True):
    """Exécute une commande shell et logue la sortie."""
    logger = get_run_logger()
    logger.info("Commande : %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)  # nosec B603
    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Échec ({result.returncode}) : {' '.join(cmd)}")
    return result.returncode


# ----------------------------- TASKS : code -----------------------------
@task(name="Cloner le repo (optionnel)")
def clone_repo():
    """Étape 06 : récupère le projet depuis git si REPO_URL est défini."""
    url = os.getenv("REPO_URL")
    if not url:
        get_run_logger().info("REPO_URL non défini : étape ignorée.")
        return
    run_cmd(["git", "clone", url, "projet"])


@task(name="Installer les dépendances")
def install_dependencies():
    run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="Formatage du code (black)")
def format_code():
    run_cmd([sys.executable, "-m", "black", *CODE_FILES])


@task(name="Qualité du code (flake8)")
def check_quality():
    run_cmd([sys.executable, "-m", "flake8", "--max-line-length=100", *CODE_FILES])


@task(name="Sécurité du code (bandit)")
def check_security():
    # -ll : ne remonte que les alertes de sévérité moyenne ou haute
    run_cmd([sys.executable, "-m", "bandit", "-ll", *CODE_FILES])


@task(name="Tests unitaires (pytest)")
def run_unit_tests():
    if not TEST_FILE.exists():
        TEST_FILE.parent.mkdir(exist_ok=True)
        TEST_FILE.write_text(TEST_TEMPLATE, encoding="utf-8")
    run_cmd([sys.executable, "-m", "pytest", "tests", "-q"])


# ----------------------------- TASKS : data/modèle -----------------------
@task(name="Préparation des données")
def prepare_data_task():
    return prepare_data()  # attendu : X_train, X_test, y_train, y_test


@task(name="Entraînement du modèle")
def train_model_task(X_train, y_train):
    return train_model(X_train, y_train)


@task(name="Sauvegarder le modèle")
def save_model_task(model):
    save_model(model, MODEL_PATH)
    return MODEL_PATH


@task(name="Charger le modèle")
def load_model_task():
    return load_model(MODEL_PATH)


@task(name="Évaluer le modèle")
def evaluate_model_task(model, X_test, y_test):
    metrics = evaluate_model(model, X_test, y_test)
    get_run_logger().info("Résultats : %s", metrics)
    return metrics


# ----------------------------- FLOWS -----------------------------------
@flow(name="install")
def install_flow():
    install_dependencies()


@flow(name="code")
def code_flow():
    format_code()
    check_quality()
    check_security()
    run_unit_tests()


@flow(name="train")
def train_flow():
    X_train, X_test, y_train, y_test = prepare_data_task()
    model = train_model_task(X_train, y_train)
    return model


@flow(name="evaluate")
def evaluate_flow():
    # On re-prépare les données pour disposer du jeu de test
    _, X_test, _, y_test = prepare_data_task()
    model = load_model_task()
    return evaluate_model_task(model, X_test, y_test)


@flow(name="all")
def all_flow():
    """install -> code (format/qualité/sécurité/tests) -> 2-3-4-6."""
    install_dependencies()
    code_flow()
    X_train, X_test, y_train, y_test = prepare_data_task()
    model = train_model_task(X_train, y_train)
    save_model_task(model)
    return evaluate_model_task(model, X_test, y_test)


FLOWS = {
    "all": all_flow,
    "train": train_flow,
    "entrainement": train_flow,  # alias utilisé dans l'énoncé
    "evaluate": evaluate_flow,
    "code": code_flow,
    "install": install_flow,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Prefect")
    parser.add_argument("--flow", choices=FLOWS.keys(), default="all")
    args = parser.parse_args()
    FLOWS[args.flow]()
