import os

import numpy as np
from loadingpy import pybar


class Evaluator:
    def __init__(self) -> None:
        self.labels_folder = os.path.join(os.getcwd(), "teacher_set", "test")
        self.label_paths = [
            elem for elem in sorted(os.listdir(self.labels_folder)) if "label" in elem
        ]

    def compute_iou(self, pred: np.ndarray, label: np.ndarray) -> float:
        p = pred.astype(float)
        g = label.astype(float)

        i = np.sum(p * g)

        u = np.sum(p) + np.sum(g) - i
        if u == 0:
            return -1
        return i / u

    def evaluate_one_student(self, student_folder: str) -> float:
        scores = []
        prediction_paths = [elem for elem in sorted(os.listdir(student_folder))]
        for pred_path, label_path in pybar(
            list(zip(prediction_paths, self.label_paths)),
            base_str=f"evaluate {os.path.basename(student_folder)}",
        ):
            score = self.compute_iou(
                pred=np.load(os.path.join(student_folder, pred_path)),
                label=np.load(os.path.join(self.labels_folder, label_path)),
            )
            if score < 0:
                print(label_path)
            scores.append(score)
        return np.mean(scores)

    def evaluate(self, all_students_folder: str) -> None:
        if os.path.exists(all_students_folder):
            for folder in os.listdir(all_students_folder):
                if os.path.isdir(os.path.join(all_students_folder, folder)):
                    score = self.evaluate_one_student(
                        student_folder=os.path.join(all_students_folder, folder)
                    )
                    print(
                        f"\tmean intersection over union (mIoU) for {folder}: {100*score:.4f}"
                    )
