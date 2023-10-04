import os
import shutil
from typing import Any

from loadingpy import pybar


class SpitPrivateData:
    def __init__(self) -> None:
        pass

    def move_everything(self, folder_src: str, folder_target: str):
        images = [elem for elem in sorted(os.listdir(folder_src)) if "image_" in elem]
        labels = [elem for elem in sorted(os.listdir(folder_src)) if "label" in elem]
        vis = [elem for elem in sorted(os.listdir(folder_src)) if "vis" in elem]
        for image, label, vi in pybar(
            list(zip(images, labels, vis)), base_str="move to teacher set"
        ):
            shutil.copyfile(
                os.path.join(folder_src, image), os.path.join(folder_target, image)
            )
            shutil.copyfile(
                os.path.join(folder_src, label), os.path.join(folder_target, label)
            )
            shutil.copyfile(
                os.path.join(folder_src, vi), os.path.join(folder_target, vi)
            )

    def move_student(
        self, folder_src: str, folder_target: str, move_labels: bool = True
    ):
        images = [elem for elem in sorted(os.listdir(folder_src)) if "image_" in elem]
        labels = [elem for elem in sorted(os.listdir(folder_src)) if "label" in elem]
        for image, label in pybar(
            list(zip(images, labels)), base_str="move to student set"
        ):
            shutil.copyfile(
                os.path.join(folder_src, image), os.path.join(folder_target, image)
            )
            if move_labels:
                shutil.copyfile(
                    os.path.join(folder_src, label), os.path.join(folder_target, label)
                )

    def __call__(
        self,
        train_folder: str,
        val_folder: str,
        test_folder: str,
        *args: Any,
        **kwds: Any
    ) -> Any:
        teacher_data = os.path.join(os.getcwd(), "teacher_set")
        if not os.path.exists(teacher_data):
            os.mkdir(teacher_data)
            os.mkdir(os.path.join(teacher_data, "train"))
            self.move_everything(
                folder_src=train_folder,
                folder_target=os.path.join(teacher_data, "train"),
            )
            os.mkdir(os.path.join(teacher_data, "val"))
            self.move_everything(
                folder_src=val_folder,
                folder_target=os.path.join(teacher_data, "val"),
            )
            os.mkdir(os.path.join(teacher_data, "test"))
            self.move_everything(
                folder_src=test_folder,
                folder_target=os.path.join(teacher_data, "test"),
            )
            shutil.make_archive("t", "zip", teacher_data)
        student_data = os.path.join(os.getcwd(), "student_set")
        if not os.path.exists(student_data):
            os.mkdir(student_data)
            os.mkdir(os.path.join(student_data, "train"))
            self.move_student(
                folder_src=train_folder,
                folder_target=os.path.join(student_data, "train"),
            )
            os.mkdir(os.path.join(student_data, "val"))
            self.move_student(
                folder_src=val_folder,
                folder_target=os.path.join(student_data, "val"),
            )
            os.mkdir(os.path.join(student_data, "test"))
            self.move_student(
                folder_src=test_folder,
                folder_target=os.path.join(student_data, "test"),
                move_labels=False,
            )
            shutil.make_archive("s", "zip", student_data)
