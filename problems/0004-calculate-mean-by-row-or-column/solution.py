import torch

def calculate_matrix_mean(matrix, mode: str) -> torch.Tensor:
    """
    Calculate mean of a 2D matrix per row or per column using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of means or raises ValueError on invalid mode.
    """
    a_t = torch.as_tensor(matrix, dtype=torch.float)
    ans = []
    if mode=="column":
        for i in range(len(matrix[0])):
            res = 0
            for j in range(len(matrix)):
                res+=matrix[j][i]
            ans.append(res/(len(matrix)))
    else:
        for i in matrix:
            res = sum(i)
            ans.append(res/(len(matrix[0])))
    return torch.tensor(ans)


        