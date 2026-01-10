"""Train a hazard classifier from cached NASA NeoWs data."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from eegnet_repl.config import Paths
from eegnet_repl.dataset import build_from_default_cache, save_dataset_csv
from eegnet_repl.logger import logger
from eegnet_repl.model import TrainedModel, build_pipeline


def save_model(model: TrainedModel, out_path: Path) -> None:
    """Persist a trained model bundle using pickle."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(model, f)
    logger.info("Saved model: %s", out_path)


def load_model(path: Path) -> TrainedModel:
    """Load a trained model bundle."""
    with path.open("rb") as f:
        obj = pickle.load(f)
    assert isinstance(obj, TrainedModel)  # noqa: S101
    return obj


def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, out_path: Path) -> None:
    """Save a confusion matrix plot."""
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_title("Confusion matrix")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train a NeoWs hazard classifier from cached data.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction (default: 0.2).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    args = parser.parse_args()

    paths = Paths.from_here()

    dataset = build_from_default_cache()
    out_csv = paths.data_processed / "neo_dataset.csv"
    save_dataset_csv(dataset, out_csv)

    df = dataset.df
    X = df[dataset.feature_cols]
    y = df[dataset.target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y if y.nunique() > 1 else None,
    )

    pipeline = build_pipeline(seed=args.seed)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    logger.info("Accuracy: %.4f", acc)
    logger.info("Classification report:\n%s", classification_report(y_test, y_pred, digits=4))

    # Save a confusion matrix figure (nice “data science artifact”)
    cm_path = paths.reports / "confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, cm_path)
    logger.info("Saved report: %s", cm_path)

    model = TrainedModel(pipeline=pipeline, feature_cols=dataset.feature_cols, target_col=dataset.target_col)
    save_model(model, paths.models / "hazard_model.pkl")


if __name__ == "__main__":
    main()
