import torch
from torch import jit


def _softmin(input: torch.Tensor,
             eps: float,
             dim: int) -> torch.Tensor:
    # `BxNxM` -> `BxN` (if dim=2) or `BxM` (if dim=1)
    return -eps * (-input / eps).logsumexp(dim=dim)


def test_softmin():
    input = torch.randn(4, 3, 2)
    assert _softmin(input, 0.1, 2).size() == torch.Size([4, 3])
    assert _softmin(input, 0.1, 1).size() == torch.Size([4, 2])
    # for very small eps, softmin==min
    assert torch.allclose(_softmin(input, 0.001, 2), input.min(dim=2)[0])


def _centerize(c: torch.Tensor,
               x: torch.Tensor,
               y: torch.Tensor) -> torch.Tensor:
    # `BxNxM` -> `BxNxM`
    return c - x.unsqueeze(-1) - y.unsqueeze(1)


@jit.script
def log_sinkhorn(x: torch.Tensor,
                 y: torch.Tensor,
                 a: torch.Tensor,
                 b: torch.Tensor,
                 eps: float,
                 p: int,
                 threshold: float,
                 num_inner_iter: int = 20,
                 max_iter: int = 1_000) -> torch.Tensor:
    """ The stabilized Sinkhorn algorithms of the Sinkhorn's algorithm:
    https://github.com/google-research/google-research/blob/master/soft_sort/sinkhorn.py

    :param x: the input tensor of shape `BxN`
    :param y: the target tensor of shape `BxM`
    :param a: the weight tensor associated to `x`
    :param b: the weight tensor associated to `y`
    :param eps: the value of the entropic regularization parameter
    :param p: the power to be applied to the distance kernel
    :param threshold: the value of sinkhorn error below which to stop iterating
    :param num_inner_iter:
    :param max_iter: the total number of iterations
    :return: transform matrix of shape `BxNxM`
    """

    assert x.size(0) == y.size(0)
    assert x.size() == a.size()
    assert y.size() == b.size()
    c = (x.unsqueeze(-1) - y.unsqueeze(1)).abs_().pow_(p)  # BxNxM
    log_a = a.log()
    log_b = b.log()
    alpha = torch.zeros_like(log_a)
    beta = torch.zeros_like(log_b)

    for i in range(max_iter):
        alpha += (eps * log_a + _softmin(_centerize(c, alpha, beta), eps, 2))
        beta += (eps * log_b + _softmin(_centerize(c, alpha, beta), eps, 1))

        if i % num_inner_iter == 0:
            _b = (-_centerize(c, alpha, beta) / eps).exp().sum(dim=1)
            if ((b - _b).abs() / b).sum() < threshold:
                break
    # return transport
    return (-_centerize(c, alpha, beta) / eps).exp()


def test_log_sinkhorn():
    # just check it is runnable
    x = torch.zeros(1, 3)
    x[0, 0] = 1
    y = torch.randn(1, 2)
    x[0, 1] = 1
    a = torch.ones_like(x) / 3
    b = torch.ones_like(y) / 2
    print(log_sinkhorn(x, y, a, b, 0.01, 2, 0.0001))
