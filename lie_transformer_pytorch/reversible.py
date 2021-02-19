import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states

# helpers

def sum_tuple(x, y, dim = 1):
    x = list(x)
    x[dim] += y[dim]
    return tuple(x)

def subtract_tuple(x, y, dim = 1):
    x = list(x)
    x[dim] -= y[dim]
    return tuple(x)

def set_tuple(x, dim, value):
    x = list(x).copy()
    x[dim] = value
    return tuple(x)

def map_tuple(fn, x, dim = 1):
    x = list(x)
    x[dim] = fn(x[dim])
    return tuple(x)

def chunk_tuple(fn, x, dim = 1):
    x = list(x)
    value = x[dim]
    chunks = fn(value)
    return tuple(map(lambda t: set_tuple(x, 1, t), chunks))

def cat_tuple(x, y, dim = 1, cat_dim = -1):
    x = list(x)
    y = list(y)
    x[dim] = torch.cat((x[dim], y[dim]), dim = cat_dim)
    return tuple(x)

def del_tuple(x):
    for el in x:
        if el is not None:
            del el

# following example for saving and setting rng here https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html
class Deterministic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng = False, set_rng = False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)

# heavily inspired by https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
# once multi-GPU is confirmed working, refactor and send PR back to source
class ReversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(self, x, f_args = {}, g_args = {}):
        training = self.training
        x1, x2 = chunk_tuple(lambda t: torch.chunk(t, 2, dim=2), x)
        y1, y2 = None, None

        with torch.no_grad():
            y1 = sum_tuple(self.f(x2, record_rng = training, **f_args), x1)
            y2 = sum_tuple(self.g(y1, record_rng = training, **g_args), x2)

        return cat_tuple(y1, y2, cat_dim = 2)

    def backward_pass(self, y, dy, f_args = {}, g_args = {}):
        y1, y2 = chunk_tuple(lambda t: torch.chunk(t, 2, dim=2), y)
        del_tuple(y)

        dy1, dy2 = torch.chunk(dy, 2, dim=2)
        del dy

        with torch.enable_grad():
            y1[1].requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1[1], dy2)

        with torch.no_grad():
            x2 = subtract_tuple(y2, gy1)
            del_tuple(y2)
            del gy1

            dx1 = dy1 + y1[1].grad
            del dy1
            y1[1].grad = None

        with torch.enable_grad():
            x2[1].requires_grad = True
            fx2 = self.f(x2, set_rng = True, **f_args)
            torch.autograd.backward(fx2[1], dx1)

        with torch.no_grad():
            x1 = subtract_tuple(y1, fx2)
            del fx2
            del_tuple(y1)

            dx2 = dy2 + x2[1].grad
            del dy2
            x2[1].grad = None

            x2 = map_tuple(lambda t: t.detach(), x2)
            x = cat_tuple(x1, x2, cat_dim = -1)
            dx = torch.cat((dx1, dx2), dim=2)

        return x, dx

class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, kwargs):
        ctx.kwargs = kwargs
        x = (kwargs.pop('coords'), x, kwargs.pop('mask'), kwargs.pop('edges'))

        for block in blocks:
            x = block(x, **kwargs)

        ctx.y = map_tuple(lambda t: t.detach(), x, dim = 1)
        ctx.blocks = blocks        
        return x[1]

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        kwargs = ctx.kwargs

        for block in ctx.blocks[::-1]:
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None

class SequentialSequence(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for (f, g) in self.blocks:
            x = sum_tuple(f(x), x, dim = 1)
            x = sum_tuple(g(x), x, dim = 1)
        return x

class ReversibleSequence(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList([ReversibleBlock(f, g) for (f, g) in blocks])

    def forward(self, x, **kwargs):
        x = map_tuple(lambda t: torch.cat((t, t), dim = -1), x)

        blocks = self.blocks

        coords, values, mask, edges = x
        kwargs = {'coords': coords, 'mask': mask, 'edges': edges, **kwargs}
        x = _ReversibleFunction.apply(values, blocks, kwargs)

        x = (coords, x, mask, edges)
        return map_tuple(lambda t: sum(t.chunk(2, dim = -1)) * 0.5, x)
