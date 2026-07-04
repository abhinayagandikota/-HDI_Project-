import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from preprocessing import load_data, preprocess_data, split_data
from model import MODELS

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..")

def plot_confusion_matrix(cm, classes, title, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")

def plot_feature_importance(model, feature_names, title, save_path):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(8, 5))
        plt.barh(range(len(indices)), importances[indices], align="center")
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Importance")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"  Saved: {save_path}")

def plot_correlation_heatmap(df, save_path):
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of HDI Features")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")

def plot_distributions(df, save_dir):
    features = ["life_expectancy", "mean_schooling_years", "expected_schooling_years", "gni_per_capita"]
    for feat in features:
        plt.figure(figsize=(7, 4))
        sns.histplot(df[feat], kde=True, bins=30)
        plt.title(f"Distribution of {feat}")
        plt.tight_layout()
        path = os.path.join(save_dir, f"dist_{feat}.png")
        plt.savefig(path)
        plt.close()
        print(f"  Saved: {path}")

def plot_pairplot(df, save_path):
    features = ["life_expectancy", "mean_schooling_years", "expected_schooling_years", "gni_per_capita", "hdi_category"]
    subset = df[features].copy()
    subset["hdi_category"] = subset["hdi_category"].astype(str)
    g = sns.pairplot(subset, hue="hdi_category", diag_kind="kde", palette="Set2")
    g.fig.suptitle("Pair Plot of HDI Features", y=1.02)
    g.savefig(save_path)
    plt.close()
    print(f"  Saved: {save_path}")

def train_and_evaluate():
    print("=" * 60)
    print("HDI Prediction – Training Pipeline")
    print("=" * 60)

    df = load_data()
    print(f"\nLoaded dataset: {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"Categories:\n{df['hdi_category'].value_counts()}")

    viz_dir = os.path.join(OUTPUT_DIR, "static", "viz")
    os.makedirs(viz_dir, exist_ok=True)

    print("\n--- Generating Visualizations ---")
    plot_correlation_heatmap(df, os.path.join(viz_dir, "correlation_heatmap.png"))
    plot_distributions(df, viz_dir)
    plot_pairplot(df, os.path.join(viz_dir, "pairplot.png"))

    X, y, scaler, le = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    feature_names = ["life_expectancy", "mean_schooling_years", "expected_schooling_years", "gni_per_capita"]
    class_names = le.classes_

    print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    results = []

    for name, builder in MODELS.items():
        print(f"\n--- Training: {name} ---")
        model = builder()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=class_names)

        print(f"Accuracy: {acc:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Classification Report:\n{report}")

        plot_confusion_matrix(cm, class_names, f"{name} - Confusion Matrix",
                              os.path.join(viz_dir, f"cm_{name.lower().replace(' ', '_')}.png"))

        if name == "Random Forest":
            plot_feature_importance(model, feature_names,
                                    f"{name} - Feature Importance",
                                    os.path.join(viz_dir, "feature_importance.png"))

        results.append((name, acc, model))

    results.sort(key=lambda x: x[1], reverse=True)
    best_name, best_acc, best_model = results[0]
    print(f"\n=== Best Model: {best_name} (Accuracy: {best_acc:.4f}) ===")

    artifacts_dir = os.path.join(OUTPUT_DIR, "src", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    model_path = os.path.join(artifacts_dir, "hdi_model.pkl")
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    encoder_path = os.path.join(artifacts_dir, "encoder.pkl")

    pickle.dump(best_model, open(model_path, "wb"))
    pickle.dump(scaler, open(scaler_path, "wb"))
    pickle.dump(le, open(encoder_path, "wb"))

    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Encoder saved to: {encoder_path}")
    print("\nTraining complete!")

if __name__ == "__main__":
    train_and_evaluate()
