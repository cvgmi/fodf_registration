import torch
import torch as nn

def scale_and_square(X, initialize, update, N=7):
    out = initialize(X, N)
    for _ in range(N):
        out = update(out)
    return out

def affine_exponential(X, N=7):
    """
    Input: Bx4x4 - a collection of 4x4 matrices.
    Output: Bx4x4 - a collection of 4x4 GL(n) matrices.

    Uses Lie group exponential function to map Lie Algebra matrices to Lie Group matrices.
    Uses scale and square to compute exponential.

    Possible improvement: initialize method below could cache identity matrices so we arent constantly
    transferring to GPU.
    """
    def initialize(X, N):
         return torch.eye(4,4)[None].to(X.device).repeat(X.shape[0],1,1)+X/(2**N)
    def update(X):
        return torch.matmul(X, X)
    return scale_and_square(X, initialize, update, N)

if __name__=='__main__':
    # affine exponential test
    X = torch.eye(4,4)[None].repeat(1000, 1, 1)
    Y = affine_exponential(X)
    print(Y)

