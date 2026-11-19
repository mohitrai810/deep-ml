import torch
def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2x2 matrix using PyTorch.
    Input: 2x2 tensor; Output: 1-D tensor with the two eigenvalues in descending order (highest to lowest).
    """
    m = torch.tensor(matrix,dtype=torch.float32)
    eig = torch.linalg.eigvals(m)
    real = eig.real
    s,x = torch.sort(real,descending=True)
    print(x)
    return s
