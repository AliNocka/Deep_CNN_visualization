import torch


class EqualityFunction(torch.autograd.Function):

  def __call__(self, x):
    return x
