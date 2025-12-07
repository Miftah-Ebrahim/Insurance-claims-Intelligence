import os
import sys
import logging
from src.data.loader import load_data
from src.features.build_features import DataBuilder
from src.models.train_model import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """
    Main pipeline execution: Load -> Preprocess -> Train -> Evaluate.
    """
    logging.info("Starting End-to-End Pipeline...")

    # 1. Load Data
    data_path = "data/raw/MachineLearningRating.txt"
    if not os.path.exists(data_path):
        logging.error(f"Data file not found at {data_path}")
        return

    logging.info("Loading data...")
    df_raw = load_data(data_path)

    # 2. Preprocess
    logging.info("Building features...")
    builder = DataBuilder(df_raw)
    builder.preprocess()

    # 3. Train Models
    trainer = ModelTrainer()

    # Severity Model (Regression)
    logging.info("--- Pipeline: Claim Severity Model ---")
    X_sev, y_sev = builder.get_severity_data()
    X_train_s, X_test_s, y_train_s, y_test_s = builder.split_data(X_sev, y_sev)
    trainer.train_severity_models(X_train_s, X_test_s, y_train_s, y_test_s)

    # Probability Model (Classification)
    logging.info("--- Pipeline: Claim Probability Model ---")
    X_prob, y_prob = builder.get_probability_data()
    X_train_p, X_test_p, y_train_p, y_test_p = builder.split_data(X_prob, y_prob)
    trainer.train_probability_models(X_train_p, X_test_p, y_train_p, y_test_p)

    # 4. Save Artifacts
    if not os.path.exists("models"):
        os.makedirs("models")

    # Save best models (Example: using XGBoost if available, else RF)
    if "Severity_XGB" in trainer.models:
        trainer.save_model("Severity_XGB", "models/severity_model.pkl")
    else:
        trainer.save_model("Severity_RF", "models/severity_model.pkl")

    if "Probability_XGB" in trainer.models:
        trainer.save_model("Probability_XGB", "models/probability_model.pkl")
    else:
        trainer.save_model("Probability_RF", "models/probability_model.pkl")

    logging.info("Pipeline Complete. Models saved to 'models/' directory.")
    print("\n--- Final Results ---")
    print(trainer.get_results())


if __name__ == "__main__":
    main()
