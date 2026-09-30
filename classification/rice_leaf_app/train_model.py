"""Run from project root: python -m rice_leaf_app.train_model."""
import json
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version

import joblib
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from rice_leaf_app.config import APP_DIR, ARTIFACTS, CLASSES, MODEL_PATH, ROOT, SEED
from rice_leaf_app.dataset import audit_dataset
from rice_leaf_app.preprocessing import FEATURE_VERSION, image_to_features


def train():
    print("1/5 Auditing images and exact duplicates...", flush=True)
    manifest = audit_dataset()
    usable = manifest[manifest.status.eq("included")].reset_index(drop=True)
    print(f"Included {len(usable)} / {len(manifest)} images", flush=True)
    print("2/5 Extracting 404 color and texture features...", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        features = []
        for i, feature in enumerate(pool.map(image_to_features, [ROOT / p for p in usable.path]), 1):
            features.append(feature)
            if i % 2000 == 0:
                print(f"Extracted {i}/{len(usable)} images", flush=True)
        X = np.asarray(features)
    y = usable.label.to_numpy()
    train_idx = np.flatnonzero(usable.split.eq("train"))
    test_idx = np.flatnonzero(usable.split.eq("test"))
    fit_idx, val_idx = train_test_split(train_idx, test_size=0.2, stratify=y[train_idx], random_state=SEED)
    usable["development_split"] = "test"
    usable.loc[fit_idx, "development_split"] = "fit"
    usable.loc[val_idx, "development_split"] = "validation"
    usable.to_csv(ARTIFACTS / "split_manifest.csv", index=False)
    candidates = []
    print("3/5 Selecting C using training-only validation (macro F1)...", flush=True)
    for c in (1.0, 10.0):
        candidate = make_pipeline(StandardScaler(), SVC(C=c, kernel="rbf", class_weight="balanced", cache_size=512))
        candidate.fit(X[fit_idx], y[fit_idx])
        score = f1_score(y[val_idx], candidate.predict(X[val_idx]), average="macro")
        candidates.append({"C": c, "validation_macro_f1": float(score)})
        print(candidates[-1], flush=True)
    best_c = max(candidates, key=lambda row: row["validation_macro_f1"])["C"]
    print(f"4/5 Fitting final SVM C={best_c} on training split...", flush=True)
    model = make_pipeline(StandardScaler(), SVC(C=best_c, kernel="rbf", class_weight="balanced",
                                               probability=True, random_state=SEED, cache_size=512))
    model.fit(X[train_idx], y[train_idx])
    # Use the same probability argmax rule in evaluation and application.
    probabilities = model.predict_proba(X[test_idx])
    predicted = model.classes_[probabilities.argmax(axis=1)]
    labels = list(CLASSES)
    report = classification_report(y[test_idx], predicted, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y[test_idx], predicted, labels=labels)
    metrics = dict(accuracy=float(accuracy_score(y[test_idx], predicted)),
                   macro_f1=float(f1_score(y[test_idx], predicted, average="macro")),
                   weighted_f1=float(f1_score(y[test_idx], predicted, average="weighted")),
                   train_count=len(train_idx), test_count=len(test_idx), fit_count=len(fit_idx),
                   validation_count=len(val_idx), raw_count=len(manifest),
                   excluded_count=int(manifest.status.eq("excluded").sum()),
                   best_C=best_c, candidates=candidates, classes=labels, report=report,
                   confusion_matrix=cm.tolist(), feature_version=FEATURE_VERSION,
                   feature_count=X.shape[1], seed=SEED,
                   prediction_rule="argmax(predict_proba)",
                   versions={p: version(p) for p in ["numpy", "Pillow", "scikit-learn", "joblib", "gradio"]})
    counts = manifest.groupby(["label", "split"]).size().unstack(fill_value=0).add_prefix("raw_")
    counts = counts.join(usable.groupby(["label", "split"]).size().unstack(fill_value=0).add_prefix("used_"))
    counts.to_csv(ARTIFACTS / "class_counts.csv")
    per_class = pd.DataFrame({label: report[label] for label in labels}).T
    per_class.index.name = "label"
    per_class.to_csv(ARTIFACTS / "per_class_metrics.csv")
    predictions = usable.iloc[test_idx][["path", "label"]].copy()
    predictions["predicted"] = predicted
    predictions["probability"] = probabilities.max(axis=1)
    predictions["correct"] = predictions.label.eq(predictions.predicted)
    predictions.to_csv(ARTIFACTS / "test_predictions.csv", index=False)
    predictions[~predictions.correct].to_csv(ARTIFACTS / "errors.csv", index=False)
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(11, 9))
    plot = ax.imshow(cm, cmap="Greens")
    english = [CLASSES[label][0] for label in labels]
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=english,
           yticklabels=english, xlabel="Predicted class", ylabel="True class", title="Held-out test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black")
    fig.colorbar(plot, ax=ax)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts[["used_train", "used_test"]].rename(columns={"used_train":"Train", "used_test":"Test"}).plot.bar(ax=axes[0])
    axes[0].set(title="Images per class after exact deduplication", ylabel="Images")
    per_class[["precision", "recall", "f1-score"]].plot.bar(ax=axes[1], ylim=(0,1))
    axes[1].set(title="Held-out performance per class", ylabel="Score")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "class_comparison.png", dpi=150)
    plt.close(fig)
    # Small portable input examples, selected from held-out images.
    examples = APP_DIR / "examples"
    examples.mkdir(exist_ok=True)
    for label in labels:
        source = ROOT / usable[(usable.split == "test") & (usable.label == label)].iloc[0].path
        from PIL import Image, ImageOps
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((800, 800))
            image.save(examples / f"{label}.jpg", quality=92)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(dict(model=model, feature_version=FEATURE_VERSION, metrics=metrics), MODEL_PATH, compress=3)
    restored = joblib.load(MODEL_PATH)["model"]
    np.testing.assert_allclose(restored.predict_proba(X[test_idx[:10]]), probabilities[:10])
    print("5/5 Saved model, reports, plots, examples. Reload verified.", flush=True)
    print(json.dumps({k: metrics[k] for k in ["accuracy", "macro_f1", "weighted_f1", "train_count", "test_count"]}), flush=True)
    return metrics


if __name__ == "__main__":
    train()
