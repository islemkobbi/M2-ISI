import torch


class BalancedLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(BalancedLoss, self).__init__()
        self.compute_bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.compute_bce(input, target)
        return loss
