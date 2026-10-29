"""
Data Classification Using AI  |  DecodeLabs AI Internship - Project 2
=====================================================================
Supervised learning pipeline on the Iris benchmark dataset (IPO framework):

    INPUT   -> load Iris (150 samples, 3 classes, 4 features)
    PROCESS -> shuffled train/test split (80/20) -> StandardScaler -> KNN
    OUTPUT  -> accuracy, confusion matrix, precision / recall / F1 score

Run:
    python classifier.py            # full pipeline, saves plots to outputs/
    python classifier.py --show     # also open the plot windows
    python classifier.py --predict  # then classify a flower you type in
"""

import argparse
import os

import matplotlib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.20          # 80% train / 20% test
DEFAULT_K = 5
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


# ---------------------------------------------------------------------------
# INPUT: load and understand the dataset
# ---------------------------------------------------------------------------
def load_data():
    """Load the Iris dataset and return (X, y, feature_names, class_names)."""
    iris = load_iris()
    return iris.data, iris.target, [str(f) for f in iris.feature_names], [str(c) for c in iris.target_names]


def describe_data(X, y, feature_names, class_names):
    """Print a short summary so we understand the data before modelling."""
    print("=" * 60)
    print("STEP 1: DATASET OVERVIEW")
    print("=" * 60)
    print(f"Samples   : {X.shape[0]}")
    print(f"Features  : {X.shape[1]} -> {', '.join(feature_names)}")
    print(f"Classes   : {len(class_names)} -> {', '.join(class_names)}")
    counts = np.bincount(y)
    for name, count in zip(class_names, counts):
        print(f"  {name:<11}: {count} samples")
    print("\nFeature statistics (cm):")
    print(f"  {'feature':<20}{'min':>6}{'max':>6}{'mean':>7}{'std':>6}")
    for i, name in enumerate(feature_names):
        col = X[:, i]
        print(f"  {name:<20}{col.min():>6.1f}{col.max():>6.1f}{col.mean():>7.2f}{col.std():>6.2f}")
    print(f"\nMissing values: {int(np.isnan(X).sum())}")


# ---------------------------------------------------------------------------
# PROCESS: split, scale, tune K, train
# ---------------------------------------------------------------------------
def split_data(X, y):
    """Shuffled, stratified 80/20 split (stratify keeps class balance equal)."""
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        shuffle=True,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def scale_features(X_train, X_test):
    """
    StandardScaler (mean = 0, variance = 1).
    IMPORTANT: fit on the TRAINING data only, then apply to the test data.
    Fitting on the test set would leak information (data leakage).
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


def find_best_k(X_train_scaled, y_train, k_values=range(1, 26), folds=5):
    """
    Choose K with 5-fold cross-validation on the TRAINING set only.
    Returns (best_k, {k: error_rate}).  Ties go to the K closest to DEFAULT_K (5).
    """
    errors = {}
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=folds, scoring="accuracy")
        errors[k] = 1 - scores.mean()
    best_error = min(errors.values())
    tied = [k for k, e in errors.items() if np.isclose(e, best_error)]
    best_k = min(tied, key=lambda k: (abs(k - DEFAULT_K), k))
    return best_k, errors


def train_knn(X_train_scaled, y_train, k=DEFAULT_K):
    """Instantiate -> Fit."""
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    return model


# ---------------------------------------------------------------------------
# OUTPUT: evaluate
# ---------------------------------------------------------------------------
def evaluate(model, X_test_scaled, y_test, class_names):
    """Predict -> accuracy, confusion matrix, F1 and full report."""
    predictions = model.predict(X_test_scaled)
    results = {
        "predictions": predictions,
        "accuracy": accuracy_score(y_test, predictions),
        "f1_macro": f1_score(y_test, predictions, average="macro"),
        "f1_weighted": f1_score(y_test, predictions, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "report": classification_report(y_test, predictions, target_names=class_names, digits=3),
    }
    return results


def plot_k_selection(errors, best_k, show=False):
    import matplotlib.pyplot as plt

    ks, errs = list(errors.keys()), list(errors.values())
    plt.figure(figsize=(8, 5))
    plt.plot(ks, errs, marker="o", color="#0b3d7a")
    plt.scatter([best_k], [errors[best_k]], s=250, facecolors="none",
                edgecolors="#e8561b", linewidths=2.5, label=f"Best K = {best_k}")
    plt.title("Choosing K (5-fold cross-validation on training data)")
    plt.xlabel("K value")
    plt.ylabel("Error rate")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "k_selection.png")
    plt.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close()
    return path


def plot_confusion_matrix(model, X_test_scaled, y_test, class_names, show=False):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test_scaled, y_test, display_labels=class_names, cmap="Blues", ax=ax
    )
    ax.set_title("Confusion Matrix (test set)")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close()
    return path


def compare_algorithms(X_train_scaled, X_test_scaled, y_train, y_test, best_k):
    """Bonus: compare KNN with other simple classifiers on the same split."""
    candidates = {
        f"KNN (k={best_k})": KNeighborsClassifier(n_neighbors=best_k),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    }
    rows = []
    for name, clf in candidates.items():
        clf.fit(X_train_scaled, y_train)
        pred = clf.predict(X_test_scaled)
        rows.append((name, accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro")))
    return rows


def predict_flower(model, scaler, class_names, measurements):
    """Classify one new flower: [sepal_len, sepal_wid, petal_len, petal_wid] in cm."""
    sample = scaler.transform(np.array(measurements, dtype=float).reshape(1, -1))
    return class_names[int(model.predict(sample)[0])]


def interactive_predict(model, scaler, class_names, feature_names):
    print("\n" + "=" * 60)
    print("TRY IT: classify your own flower (type 'q' to quit)")
    print("=" * 60)
    while True:
        values = []
        for name in feature_names:
            raw = input(f"  {name}: ").strip().lower()
            if raw in ("q", "quit", "exit"):
                print("Goodbye!")
                return
            try:
                values.append(float(raw))
            except ValueError:
                print("  Please enter a number.")
                break
        else:
            print(f"  -> Predicted species: {predict_flower(model, scaler, class_names, values)}\n")


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Iris classification with KNN")
    parser.add_argument("--show", action="store_true", help="open plot windows")
    parser.add_argument("--predict", action="store_true", help="classify a flower you type in")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")  # no window, just save PNGs
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. INPUT
    X, y, feature_names, class_names = load_data()
    describe_data(X, y, feature_names, class_names)

    # 2. PROCESS
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("\n" + "=" * 60)
    print("STEP 2: TRAIN / TEST SPLIT (shuffled, stratified)")
    print("=" * 60)
    print(f"Training samples: {len(X_train)} (80%)")
    print(f"Testing samples : {len(X_test)} (20%)")

    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
    print("\nSTEP 3: FEATURE SCALING (StandardScaler, fit on training data only)")
    print(f"  Train mean after scaling ~ {np.round(X_train_s.mean(axis=0), 3).tolist()}")
    print(f"  Train std  after scaling ~ {np.round(X_train_s.std(axis=0), 3).tolist()}")

    best_k, errors = find_best_k(X_train_s, y_train)
    print("\n" + "=" * 60)
    print("STEP 4: CHOOSING K (5-fold cross-validation)")
    print("=" * 60)
    print(f"Best K = {best_k}  (cv error = {errors[best_k]:.4f})")
    print(f"K = {DEFAULT_K} (slide default) cv error = {errors[DEFAULT_K]:.4f}")
    k_plot = plot_k_selection(errors, best_k, show=args.show)

    model = train_knn(X_train_s, y_train, k=best_k)
    print(f"\nSTEP 5: MODEL TRAINED -> KNeighborsClassifier(n_neighbors={best_k})")

    # 3. OUTPUT
    results = evaluate(model, X_test_s, y_test, class_names)
    print("\n" + "=" * 60)
    print("STEP 6: EVALUATION ON UNSEEN TEST DATA")
    print("=" * 60)
    print(f"Accuracy       : {results['accuracy']:.4f}")
    print(f"F1 (macro)     : {results['f1_macro']:.4f}")
    print(f"F1 (weighted)  : {results['f1_weighted']:.4f}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(f"  classes: {class_names}")
    print(results["confusion_matrix"])
    print("\nClassification report:")
    print(results["report"])
    cm_plot = plot_confusion_matrix(model, X_test_s, y_test, class_names, show=args.show)

    print("=" * 60)
    print("BONUS: ALGORITHM COMPARISON (same split)")
    print("=" * 60)
    print(f"  {'algorithm':<22}{'accuracy':>10}{'F1 macro':>10}")
    for name, acc, f1 in compare_algorithms(X_train_s, X_test_s, y_train, y_test, best_k):
        print(f"  {name:<22}{acc:>10.4f}{f1:>10.4f}")

    demo = [5.1, 3.5, 1.4, 0.2]
    print(f"\nDemo prediction for {demo} -> {predict_flower(model, scaler, class_names, demo)}")
    print(f"\nPlots saved:\n  {k_plot}\n  {cm_plot}")

    if args.predict:
        interactive_predict(model, scaler, class_names, feature_names)


if __name__ == "__main__":
    main()
