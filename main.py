"""Point d'entrée : exécution manuelle des étapes ML."""

import argparse

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


def main():
    parser = argparse.ArgumentParser(description="Étapes ML")
    parser.add_argument("--train", action="store_true", help="entraîner et sauvegarder")
    parser.add_argument("--evaluate", action="store_true", help="charger et évaluer")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = prepare_data()

    if args.train:
        model = train_model(X_train, y_train)
        save_model(model)
        print("Modèle entraîné et sauvegardé.")

    if args.evaluate:
        model = load_model()
        print(evaluate_model(model, X_test, y_test))


if __name__ == "__main__":
    main()
