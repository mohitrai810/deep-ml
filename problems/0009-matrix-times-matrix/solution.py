import torch

def matrixmul(a, b) -> torch.Tensor:
    """
    Multiply two matrices using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2D tensor of shape (m, n) or a scalar tensor -1 if dimensions mismatch.
    """
    na = torch.tensor(a)
    nb = torch.tensor(b)
    return torch.tensor(-1) if na.shape[1]!=nb.shape[0] else na@nb
