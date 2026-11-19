import torch

def matrix_dot_vector(a, b) -> torch.Tensor:
    at = torch.tensor(a)
    bt = torch.tensor(b)

    return torch.tensor(-1) if at.shape[1] != bt.shape[0] else at@bt