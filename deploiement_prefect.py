"""Étape 05 : déploiements planifiés (le "worker" du Terminal 2)."""

from prefect import serve

from pipeline_prefect import all_flow, code_flow, evaluate_flow, install_flow, train_flow

if __name__ == "__main__":
    serve(
        # Une fois par jour à 02:00
        all_flow.to_deployment(
            name="ml-pipeline-all",
            cron="0 2 * * *",
            tags=["mlops", "full-pipeline"],
        ),
        train_flow.to_deployment(name="ml-pipeline-train", tags=["mlops", "training"]),
        evaluate_flow.to_deployment(
            name="ml-pipeline-evaluate", tags=["mlops", "evaluation"]
        ),
        code_flow.to_deployment(name="ml-pipeline-code", tags=["mlops", "code"]),
        install_flow.to_deployment(name="ml-pipeline-install", tags=["mlops", "setup"]),
    )
