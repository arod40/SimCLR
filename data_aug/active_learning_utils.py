from pathlib import Path
import json
import shutil
import fire
import numpy as np
from typing import List


def convert_doccano_annotations_to_image_folder(dir: str):
    """Converts doccano annotations to image folder structure expected by ImageFolder dataset.
    Assumes 'dir' to have at least the following structure
    /dir
        all.jsonl
        /images
            /image1.jpg
            /image2.jpg
            ...
    """
    dir = Path(dir)
    labeled_dir = dir / "labeled"
    labeled_dir.mkdir(exist_ok=True, parents=True)

    with open(dir / "all.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            if data["label"]:
                image_name = data["filename"]
                image_path = dir / "images" / image_name
                if image_path.exists():
                    label = data["label"][0]
                    label_dir = labeled_dir / label
                    label_dir.mkdir(exist_ok=True, parents=True)
                    shutil.copy(image_path, label_dir / image_name)


def compute_metrics(dir: str, classes: List[str]):
    """Computes metrics afer labeling a step in the active learning.
    Assumes 'dir' to have at least the following structure
    /dir
        all.jsonl
        metadata.json
        ...
    The 'classes' list should be in the same order as the probs appear in the metadata.json file.
    """
    dir = Path(dir)
    print(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    metadata = json.loads(Path(dir / "metadata.json").read_text())

    y_pred = []
    y = []
    with open(dir / "all.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            if data["label"]:
                image_name = Path(data["filename"]).stem
                y_pred.append(np.array(metadata[image_name]).argmax())
                y.append(class_to_idx[data["label"][0]])

    y_pred = np.array(y_pred)
    y = np.array(y)

    # Compute metrics
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
        roc_auc_score,
        roc_curve,
    )

    print(classification_report(y, y_pred, target_names=classes))


if __name__ == "__main__":
    fire.Fire(
        {
            "convert": convert_doccano_annotations_to_image_folder,
            "metrics": compute_metrics,
        }
    )
