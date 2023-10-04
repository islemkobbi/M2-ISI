import os

import numpy as np
import torch
import torchvision
from loadingpy import pybar
from torchinfo import summary

from .architecture import Segmentor
from .dataloader import create_dataloader
from .loss import BalancedLoss


class EndNestedLoop(Exception):
    def __init__(self, message="", errors=""):
        super().__init__(message)
        self.errors = errors


class BaselineTrainer:
    def __init__(self, quiet_mode: bool) -> None:
        self.quiet_mode = quiet_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(os.path.join(os.getcwd(), "submissions")):
            os.mkdir(os.path.join(os.getcwd(), "submissions"))
        if not os.path.exists(os.path.join(os.getcwd(), "submissions", "baseline")):
            os.mkdir(os.path.join(os.getcwd(), "submissions", "baseline"))
        self.sigmoid = torch.nn.Sigmoid()

    def make_final_predictions(
        self, model: torch.nn.Module, batch_size: int = 16
    ) -> None:
        dataset = create_dataloader(
            path_to_data=os.path.join(os.getcwd(), "student_set", "test"),
            batch_size=batch_size,
        )
        model.eval()
        cpt = 0
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for inputs, original_sizes in pybar(
                    dataset, base_str="extract on the test set"
                ):
                    inputs = inputs.to(self.device)
                    predictions = torch.round(model(inputs))
                    for prediction, original_size in zip(predictions, original_sizes):
                        _, W, H = original_size
                        prediction = torchvision.transforms.Resize(
                            size=(W, H), antialias=True
                        )(prediction)
                        np.save(
                            os.path.join(
                                os.getcwd(),
                                "submissions",
                                "baseline",
                                str(cpt).zfill(6) + ".npy",
                            ),
                            torch.round(self.sigmoid(prediction)).cpu().numpy(),
                        )
                        cpt += 1

    def train(
        self, num_opt_steps: int = 20000, batch_size: int = 16, lr: float = 5.0e-3
    ) -> None:
        dataset = create_dataloader(
            path_to_data=os.path.join(os.getcwd(), "student_set", "train"),
            batch_size=batch_size,
        )
        model = Segmentor()
        model.to(self.device)
        if not self.quiet_mode:
            summary(model, input_size=(batch_size, 3, 224, 224))
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        loss_fn = BalancedLoss()
        pbar = pybar(range(num_opt_steps), base_str="training")
        current_progression = 0
        try:
            with torch.cuda.amp.autocast():
                while True:
                    for inputs, labels in dataset:
                        if not self.quiet_mode:
                            pbar.__next__()
                        current_progression += 1
                        optimizer.zero_grad()
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        predictions = model(inputs)
                        loss = loss_fn(predictions, labels)
                        loss.backward()
                        optimizer.step()
                        if not self.quiet_mode:
                            pbar.set_description(
                                description=f"loss: {loss.cpu().detach().numpy():.4f}"
                            )
                        if current_progression == num_opt_steps:
                            raise EndNestedLoop
        except EndNestedLoop:
            pass
        try:
            pbar.__next__()
        except StopIteration:
            pass
        self.make_final_predictions(model=model, batch_size=batch_size)
