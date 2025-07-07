import torch
import torch.nn as nn


class AbundanceScaling(nn.Module):
    def __init__(self, num_endmembers):
        super().__init__()
        self.scaling_matrix = nn.Parameter(
            torch.eye(num_endmembers),
            requires_grad=True
        )

    def forward(self, x):
        batch_size, num_endmembers, height, width = x.size()
        x_reshaped = x.view(batch_size, num_endmembers, -1)
        scaled = torch.bmm(
            self.scaling_matrix.unsqueeze(0).expand(batch_size, -1, -1),
            x_reshaped
        )
        return scaled.view(batch_size, num_endmembers, height, width)

class SumToOne(nn.Module):
    def __init__(self, initial_scale):
        super().__init__()
        self.scale = torch.tensor(initial_scale, dtype=torch.float32)

    def forward(self, x, indices_to_keep):
        output = torch.zeros_like(x)
        softmax_values = torch.softmax(x[:, indices_to_keep, :, :] * self.scale, dim=1)
        output[:, indices_to_keep, :, :] += softmax_values
        return output

class Nonnegative(nn.Module):
    def forward(self, X):
        return torch.clamp(X, min=0.)