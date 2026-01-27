import torch as th
import numpy as np
from scipy.stats import norm

from abc import ABC, abstractmethod
from torch.nn.utils import spectral_norm
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import timm
from distillation_utils import *
from torch.optim import RAdam
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import cv2
import scipy
_src_to_module_dict = dict()
_module_to_src_dict = dict()
_import_hooks = [] 
_version = 6
_decorators = set()
import copy

import uuid
import types
import sys
from typing import Any, List, Tuple, Union, Optional
import io
import pickle
import inspect
from torch.nn.functional import silu
import shutil

if th.cuda.is_available():
    dev =  th.device("cuda")
    print(dev)
else:
    dev = th.device("cpu")


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

### Creating model and diffusion

class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self, num_heun_step):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample_t(self, batch_size, device, num_heun_step=1, time_continuous=False):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        if time_continuous:
            indices_np = np.random.rand(batch_size)
            indices = th.from_numpy(indices_np).to(device) * (1. - num_heun_step)
            weights = th.ones_like(indices).float().to(device)
        else:
            w = self.weights(num_heun_step)
            p = w / np.sum(w)
            indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
            indices = th.from_numpy(indices_np).long().to(device)
            weights_np = 1 / (len(p) * p[indices_np])
            weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights

    def sample_s(self, args, batch_size, device, indices, num_heun_step=1, time_continuous=False, N=40):
        if time_continuous:
            new_indices = th.from_numpy(np.random.rand(indices.shape[0])).to(indices.device) * (1. - indices - num_heun_step)\
                          + indices + num_heun_step
        else:
            if args.sample_s_strategy == 'smallest':
                new_indices = th.ones(indices.shape[0], device=indices.device) * (N - 1)
            elif args.sample_s_strategy == 'uniform':
                new_indices = th.from_numpy(np.random.randint(
                    low=(indices + num_heun_step).cpu().detach().numpy(), high=N,
                    size=(indices.shape[0],),
                    dtype=int)).to(indices.device)
        return new_indices

class UniformSampler(ScheduleSampler):
    def __init__(self, num_timesteps):
        self._weights = np.ones([num_timesteps])
        self.num_timesteps = num_timesteps

    def weights(self, num_heun_step=1):
        #if num_heun_step == 1:
        #    return self._weights
        #else:
        return np.ones([self.num_timesteps - num_heun_step])

class HalfLogNormalHalfUniformSampler:
    def __init__(self, args, p_mean=-1.2, p_std=1.2, even=False):
        self.args = args
        self.p_mean = p_mean
        self.p_std = p_std
        self.even = even
        if self.even:
            self.inv_cdf = lambda x: norm.ppf(x, loc=p_mean, scale=p_std)
            # self.rank, self.size = dist.get_rank(), dist.get_world_size()

    def get_t(self, t):
        t = self.args.sigma_max ** (1 / self.args.rho) + t * (
                self.args.sigma_min ** (1 / self.args.rho) - self.args.sigma_max ** (1 / self.args.rho)
        )
        t = t ** self.args.rho
        return t

    def sample(self, bs, device, num_heun_step=1):
        log_sigmas = self.p_mean + self.p_std * th.randn(bs//2, device=device)
        sigmas = th.exp(log_sigmas)
        t = th.rand(bs - bs//2, device=device) * self.args.diffusion_mult # it determines how large sigma to sample from
        sigmas = th.cat((sigmas, self.get_t(t))).view(-1)
        weights = th.ones_like(sigmas)
        return sigmas, weights

def create_named_schedule_sampler(args, name, num_timesteps):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(num_timesteps)
    elif name == 'halflognormal':
        return HalfLogNormalHalfUniformSampler(args)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


def _src_to_module(src):
    r"""Get or create a Python module for the given source code.
    """
    module = _src_to_module_dict.get(src, None)
    if module is None:
        module_name = "_imported_module_" + uuid.uuid4().hex
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
        _module_to_src_dict[module] = src
        _src_to_module_dict[src] = module
        exec(src, module.__dict__) # pylint: disable=exec-used
    return module

def _reconstruct_persistent_obj(meta):
    r"""Hook that is called internally by the `pickle` module to unpickle
    a persistent object.
    """
    meta = EasyDict(meta)
    meta.state = EasyDict(meta.state)
    for hook in _import_hooks:
        meta = hook(meta)
        assert meta is not None

    assert meta.version == _version
    module = _src_to_module(meta.module_src)

    assert meta.type == 'class'
    orig_class = module.__dict__[meta.class_name]
    decorator_class = persistent_class(orig_class)
    obj = decorator_class.__new__(decorator_class)

    setstate = getattr(obj, '__setstate__', None)
    if callable(setstate):
        setstate(meta.state) # pylint: disable=not-callable
    else:
        obj.__dict__.update(meta.state)
    return obj

def _check_pickleable(obj):
    r"""Check that the given object is pickleable, raising an exception if
    it is not. This function is expected to be considerably more efficient
    than actually pickling the object.
    """
    def recurse(obj):
        if isinstance(obj, (list, tuple, set)):
            return [recurse(x) for x in obj]
        if isinstance(obj, dict):
            return [[recurse(x), recurse(y)] for x, y in obj.items()]
        if isinstance(obj, (str, int, float, bool, bytes, bytearray)):
            return None # Python primitive types are pickleable.
        if f'{type(obj).__module__}.{type(obj).__name__}' in ['numpy.ndarray', 'torch.Tensor', 'torch.nn.parameter.Parameter']:
            return None # NumPy arrays and PyTorch tensors are pickleable.
        if is_persistent(obj):
            return None # Persistent objects are pickleable, by virtue of the constructor check.
        return obj
    with io.BytesIO() as f:
        pickle.dump(recurse(obj), f)

def _module_to_src(module):
    r"""Query the source code of a given Python module.
    """
    src = _module_to_src_dict.get(module, None)
    if src is None:
        src = inspect.getsource(module)
        _module_to_src_dict[module] = src
        _src_to_module_dict[src] = module
    return src

def is_persistent(obj):
    r"""Test whether the given object or class is persistent, i.e.,
    whether it will save its source code when pickled.
    """
    try:
        if obj in _decorators:
            return True
    except TypeError:
        pass
    return type(obj) in _decorators # pylint: disable=unidiomatic-typecheck

def persistent_class(orig_class):
    r"""Class decorator that extends a given class to save its source code
    when pickled.

    Example:

        from torch_utils import persistence

        @persistence.persistent_class
        class MyNetwork(torch.nn.Module):
            def __init__(self, num_inputs, num_outputs):
                super().__init__()
                self.fc = MyLayer(num_inputs, num_outputs)
                ...

        @persistence.persistent_class
        class MyLayer(torch.nn.Module):
            ...

    When pickled, any instance of `MyNetwork` and `MyLayer` will save its
    source code alongside other internal state (e.g., parameters, buffers,
    and submodules). This way, any previously exported pickle will remain
    usable even if the class definitions have been modified or are no
    longer available.

    The decorator saves the source code of the entire Python module
    containing the decorated class. It does *not* save the source code of
    any imported modules. Thus, the imported modules must be available
    during unpickling, also including `torch_utils.persistence` itself.

    It is ok to call functions defined in the same module from the
    decorated class. However, if the decorated class depends on other
    classes defined in the same module, they must be decorated as well.
    This is illustrated in the above example in the case of `MyLayer`.

    It is also possible to employ the decorator just-in-time before
    calling the constructor. For example:

        cls = MyLayer
        if want_to_make_it_persistent:
            cls = persistence.persistent_class(cls)
        layer = cls(num_inputs, num_outputs)

    As an additional feature, the decorator also keeps track of the
    arguments that were used to construct each instance of the decorated
    class. The arguments can be queried via `obj.init_args` and
    `obj.init_kwargs`, and they are automatically pickled alongside other
    object state. This feature can be disabled on a per-instance basis
    by setting `self._record_init_args = False` in the constructor.

    A typical use case is to first unpickle a previous instance of a
    persistent class, and then upgrade it to use the latest version of
    the source code:

        with open('old_pickle.pkl', 'rb') as f:
            old_net = pickle.load(f)
        new_net = MyNetwork(*old_obj.init_args, **old_obj.init_kwargs)
        misc.copy_params_and_buffers(old_net, new_net, require_all=True)
    """
    assert isinstance(orig_class, type)
    if is_persistent(orig_class):
        return orig_class

    assert orig_class.__module__ in sys.modules
    orig_module = sys.modules[orig_class.__module__]
    orig_module_src = _module_to_src(orig_module)

    class Decorator(orig_class):
        _orig_module_src = orig_module_src
        _orig_class_name = orig_class.__name__

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            record_init_args = getattr(self, '_record_init_args', True)
            self._init_args = copy.deepcopy(args) if record_init_args else None
            self._init_kwargs = copy.deepcopy(kwargs) if record_init_args else None
            assert orig_class.__name__ in orig_module.__dict__
            _check_pickleable(self.__reduce__())

        @property
        def init_args(self):
            assert self._init_args is not None
            return copy.deepcopy(self._init_args)

        @property
        def init_kwargs(self):
            assert self._init_kwargs is not None
            return EasyDict(copy.deepcopy(self._init_kwargs))

        def __reduce__(self):
            fields = list(super().__reduce__())
            fields += [None] * max(3 - len(fields), 0)
            if fields[0] is not _reconstruct_persistent_obj:
                meta = dict(type='class', version=_version, module_src=self._orig_module_src, class_name=self._orig_class_name, state=fields[2])
                fields[0] = _reconstruct_persistent_obj # reconstruct func
                fields[1] = (meta,) # reconstruct args
                fields[2] = None # state dict
            return tuple(fields)

    Decorator.__name__ = orig_class.__name__
    Decorator.__module__ = orig_class.__module__
    _decorators.add(Decorator)
    return Decorator

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (th.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * th.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (th.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * th.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

class AttentionOp(th.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = th.einsum('ncq,nck->nqk', q.to(th.float32), (k / np.sqrt(k.shape[1])).to(th.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = th._softmax_backward_data(grad_output=dw.to(th.float32), output=w.to(th.float32), dim=2, input_dtype=th.float32)
        dq = th.einsum('nck,nqk->ncq', k.to(th.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = th.einsum('ncq,nqk->nck', q.to(th.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk


@persistent_class
class Linear(th.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = th.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = th.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

@persistent_class
class Conv2d(th.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = th.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = th.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = th.as_tensor(resample_filter, dtype=th.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = th.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = th.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = th.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = th.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = th.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = th.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = th.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

@persistent_class
class GroupNorm(th.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = th.nn.Parameter(th.ones(num_channels))
        self.bias = th.nn.Parameter(th.zeros(num_channels))

    def forward(self, x):
        x = th.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

@persistent_class
class PositionalEmbedding(th.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = th.arange(start=0, end=self.num_channels//2, dtype=th.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = th.cat([x.cos(), x.sin()], dim=1)
        return x

@persistent_class
class FourierEmbedding(th.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', th.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = th.cat([x.cos(), x.sin()], dim=1)
        return x

@persistent_class
class UNetBlock(th.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
        training_mode='', linear_probing=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale
        self.training_mode = training_mode
        self.linear_probing = linear_probing

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        if self.training_mode == 'ctm':
            self.affine_s = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

        if linear_probing:
            self.norm0_train = GroupNorm(num_channels=in_channels, eps=eps)
            self.conv0_train = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down,
                                      resample_filter=resample_filter, **init)
            self.norm1_train = GroupNorm(num_channels=out_channels, eps=eps)
            self.conv1_train = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

            self.skip_train = None
            if out_channels != in_channels or up or down:
                kernel = 1 if resample_proj or out_channels != in_channels else 0
                self.skip_train = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down,
                                   resample_filter=resample_filter, **init)

            if self.num_heads:
                self.norm2_train = GroupNorm(num_channels=out_channels, eps=eps)
                self.qkv_train = Conv2d(in_channels=out_channels, out_channels=out_channels * 3, kernel=1,
                                  **(init_attn if init_attn is not None else init))
                self.proj_train = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb, emb_s=None, emb_t=None):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.training_mode == 'ctm':
            params_s = self.affine_s(emb_s).unsqueeze(2).unsqueeze(3).to(x.dtype)
            if not self.linear_probing:
                params = params + params_s
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(th.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(th.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = th.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale

        if self.linear_probing:
            y = self.conv0_train(silu(self.norm0_train(orig)))
            assert emb_t != None and self.training_mode == 'ctm'
            params_t = self.affine_s(emb_t).unsqueeze(2).unsqueeze(3).to(y.dtype)
            params = params_t - params_s
            params_ = params + params_s
            if self.adaptive_scale:
                scale, shift = params_.chunk(chunks=2, dim=1)
                y = silu(th.addcmul(shift, self.norm1_train(y), scale + 1))
            else:
                y = silu(self.norm1_train(y.add_(params_)))

            y = self.conv1_train(th.nn.functional.dropout(y, p=self.dropout, training=self.training))
            y = y.add_(self.skip_train(orig) if self.skip_train is not None else orig)
            y = y * self.skip_scale

            if self.num_heads:
                q, k, v = self.qkv_train(self.norm2_train(y)).reshape(y.shape[0] * self.num_heads, y.shape[1] // self.num_heads, 3,
                                                          -1).unbind(2)
                w = AttentionOp.apply(q, k)
                a = th.einsum('nqk,nck->ncq', w, v)
                y = self.proj_train(a.reshape(*y.shape)).add_(y)
                y = y * self.skip_scale
            y = y.mul(params)

            return x + y
        return x

@persistent_class
class SongUNet(th.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 4,            # Number of residual blocks per resolution.
        attn_resolutions    = [16],         # List of resolutions with self-attention.
        dropout             = 0.13,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        embedding_type      = 'fourier', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise  = 2,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type        = 'residual',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter     = [1,3,3,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
        training_mode = '',
        linear_probing=False,
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']
        self.training_mode = training_mode
        self.linear_probing = linear_probing
        self.img_resolution = img_resolution

        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels)
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        if self.training_mode.lower() == 'ctm':
            self.map_layer0_s = Linear(in_features=noise_channels, out_features=emb_channels, **init)
            self.map_layer1_s = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder.
        self.enc = th.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True,
                                                          training_mode=training_mode, linear_probing=linear_probing, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn,
                                                                training_mode=training_mode, linear_probing=linear_probing, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = th.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True,
                                                         training_mode=training_mode, linear_probing=linear_probing, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout,
                                                         training_mode=training_mode, linear_probing=linear_probing, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True,
                                                        training_mode=training_mode, linear_probing=linear_probing, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn,
                                                                training_mode=training_mode, linear_probing=linear_probing, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)
                if self.linear_probing:
                    self.dec[f'{res}x{res}_aux_norm_train'] = GroupNorm(num_channels=cout, eps=1e-6)
                    self.dec[f'{res}x{res}_aux_lin_train'] = Linear(in_features=emb_channels, out_features=out_channels*img_resolution*img_resolution, **init)
                    self.dec[f'{res}x{res}_aux_conv_train'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3,
                                                               **init_zero)

    def forward(self, x, noise_labels, noise_labels_s, class_labels):
        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (th.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))

        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))
        if noise_labels_s != None:
            emb_s = self.map_noise(noise_labels_s)
            emb_s = emb_s.reshape(emb_s.shape[0], 2, -1).flip(1).reshape(*emb_s.shape)  # swap sin/cos
            if self.map_label is not None:
                tmp = class_labels
                if self.training and self.label_dropout:
                    tmp = tmp * (th.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
                emb_s = emb_s + self.map_label(tmp * np.sqrt(self.map_label.in_features))
            emb_s = silu(self.map_layer0_s(emb_s))
            emb_s = silu(self.map_layer1_s(emb_s))
            if self.linear_probing:
                emb_t = self.map_noise(noise_labels)
                emb_t = emb_t.reshape(emb_t.shape[0], 2, -1).flip(1).reshape(*emb_t.shape)  # swap sin/cos
                emb_t = silu(self.map_layer0_s(emb_t))
                emb_t = silu(self.map_layer1_s(emb_t))

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb, emb_s=None if noise_labels_s == None else emb_s,
                          emb_t=emb_t if self.linear_probing else None) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_lin' in name:
                emb_mult = (block(emb_t) - block(emb_s)).reshape(-1, 3, self.img_resolution, self.img_resolution)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp * emb_mult + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = th.cat([x, skips.pop()], dim=1)
                x = block(x, emb, emb_s=None if noise_labels_s == None else emb_s,
                          emb_t=emb_t if self.linear_probing else None)
        return aux

@persistent_class
class EDMPrecond_CTM(th.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'SongUNet',       # Class name of the underlying model.
        teacher = False,
        teacher_model_path = '',
        training_mode = '',
        arch='ncsn',
        linear_probing=False,
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.teacher = teacher
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.eye = th.eye(self.label_dim, device=dev)
        if teacher:
            # print("I am the teacher model -----------------")
            import pickle
            # print(f'Loading network from "{teacher_model_path}"...')
            with open(teacher_model_path, 'rb') as f:
                self.model = pickle.load(f)['ema']
        else:
            # print("I am not teacher *********************")
            if arch in ['ddpmpp', 'ncsnpp']:
                resample_filter = [1,1] if arch == 'ddpmpp' else [1,3,3,1]
                channel_mult_noise = 1 if arch == 'ddpmpp' else 2
                encoder_type = 'standard' if arch == 'ddpmpp' else 'residual'
                embedding_type = 'positional' if arch == 'ddpmpp' else 'fourier'
                self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels,
                                                   out_channels=img_channels, label_dim=label_dim,
                                                   training_mode=training_mode, resample_filter=resample_filter,
                                                   channel_mult_noise=channel_mult_noise, encoder_type=encoder_type,
                                                   embedding_type=embedding_type, linear_probing=linear_probing,
                                                   **model_kwargs)
            else:
                self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels,
                                                   out_channels=img_channels, label_dim=label_dim,
                                                   training_mode=training_mode, linear_probing=linear_probing,
                                                   **model_kwargs)

    def get_c_in(self, sigma):
        return 1 / (sigma**2 + self.sigma_data**2) ** 0.5

    def unrescaling_t(self, rescaled_t):
        return th.exp(rescaled_t / 250.) - 1e-44

    def forward(self, rescaled_x, rescaled_t, s=None, teacher=False, **model_kwargs):
        class_labels = None if self.label_dim == 0 else th.zeros([1, self.label_dim], device=rescaled_x.device) \
            if model_kwargs == {} else self.eye[model_kwargs['y']].reshape(-1, self.label_dim)
        dtype = th.float16 if self.use_fp16 and rescaled_x.device.type == 'cuda' else th.float32
        if self.teacher:
            #with torch.no_grad():
            sigma = self.unrescaling_t(rescaled_t)
            c_in = append_dims(self.get_c_in(sigma), rescaled_x.ndim)
            x = rescaled_x / c_in
            D_x = self.model(x.to(dtype), sigma.flatten(), class_labels=class_labels)
            c_skip = append_dims(self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2), rescaled_x.ndim)
            c_out = append_dims(sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt(), rescaled_x.ndim)
            F_x = (D_x - c_skip * x) / c_out
        else:
            t = self.unrescaling_t(rescaled_t)
            t = t.log() / 4
            if s != None:
                s = self.unrescaling_t(s)
                s = s.log() / 4
            F_x = self.model(rescaled_x.to(dtype), t.flatten(), None if s == None else s.flatten(), class_labels=class_labels)
        #assert F_x.dtype == dtype
        return F_x

    def round_sigma(self, sigma):
        return th.as_tensor(sigma)

    def convert_to_fp16(self):
        pass

    def convert_to_fp32(self):
        pass


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, lowest=False):
        super().__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            output = self.skip_add.add(output, xs[1])

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output

def ctm_data_defaults(data_name):
    return dict(
        train_classes=-1,
        type='png',
        sigma_data=0.5,
        deterministic=False,
        num_classes=10,
    )

def ctm_loss_defaults(data_name):
    return dict(
        # CTM hyperparams
        ctm_training=True,
        consistency_weight=1.0,
        ctm_estimate_outer_type='target_model_sg',
        ctm_estimate_inner_type='model',
        ctm_target_inner_type='model_sg',
        ctm_target_matching=False,
        sample_s_strategy='uniform',
        heun_step_strategy='weighted',
        heun_step_multiplier=1.0,
        outer_parametrization='euler',
        inner_parametrization='edm',
        time_continuous=False,
        self_learn=False,
        self_learn_iterative=False,
        target_matching=False,

        # DSM hyperparams
        diffusion_training=True,
        apply_adaptive_weight=True,
        denoising_weight=1.,
        diffusion_mult = 0.7,
        diffusion_schedule_sampler='halflognormal',
        diffusion_training_frequency=1.,

        # GAN hyperparams
        d_lr=0.002,
        gan_training=False,
        gan_specific_batch=False,
        gan_micro_batch=32,
        gan_real_free=True,
        discriminator_weight=1.0,
        discriminator_start_itr=0,
        use_d_fp16=False,
        d_architecture='StyleGAN-XL',
        g_learning_period=1,
        gan_fake_outer_type='no',
        gan_fake_inner_type='',
        gan_real_inner_type='',
        gan_target_matching=False,
        data_augment=True,
        d_backbone=['deit_base_distilled_patch16_224', 'tf_efficientnet_lite0'],
        d_apply_adaptive_weight=True,
        shift_ratio=0.125,
        cutout_ratio=0.2,
        gan_training_frequency=1.,
        gaussian_filter=False,
        blur_fade_itr=1000,
        blur_init_sigma=2,
        prob_aug=1.0,
        gan_different_augment=False,
        gan_num_heun_step=17 if data_name == 'cifar10' else 39,
        gan_heun_step_strategy='uniform',
        gan_specific_time=False,
        gan_low_res_train=False,
        d_opt_load=True,
    )

def ctm_train_defaults(data_name):
    return dict(
        beta_min=0.1,
        beta_max=20.,
        multiplier=1.,
        num_heun_step=17 if data_name == 'cifar10' else 39,
        num_heun_step_random=True,

        # Network architecture
        edm_nn_ncsn=False,
        edm_nn_ddpm=True if data_name == 'cifar10' else False,
        in_channels=3,
        linear_probing=False,
        target_subtract=False,
    )

def ctm_eval_defaults(data_name):
    return dict(
        intermediate_samples=False,
        sampling_batch=64,
        sample_interval=1000 if data_name == 'cifar10' else 10,
        sampling_steps=18 if data_name == 'cifar10' else 40,
        eval_interval=1000,
        eval_num_samples=5000,
        eval_batch=500,
        #ref_path='/home/dongjun/EighthArticleExperimentalResults/CIFAR10/author_ckpt/cifar10-32x32.npz' if data_name == 'cifar10' else "",
        ref_path='/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/author_ckpt/cifar10-32x32.npz' if data_name == 'cifar10' \
            else "/home/fp084243/EighthArticleExperimentalResults/ImageNet64/author_ckpt/VIRTUAL_imagenet64_labeled.npz",
        ref_feat_path='',
        large_log=False,
        compute_ema_fids=False,
        #dm_sample_path_seed_42='/data2/dongjun/EighthArticleExperimentalResults/CIFAR10/DM/EDM-VP/fp16-seed-42/edm_heun_sampler_18_steps_ond-vp_itrs_model_ema' if data_name == 'cifar10' else "",
        dm_sample_path_seed_42='/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/DM/heun_18_seed_42_ver2' if data_name == 'cifar10' else "",
        ae_image_path_seed_42='',
        eval_seed=42,
        eval_fid=False,
        eval_similarity=True,
        save_png=False,
        check_ctm_denoising_ability=False,
        check_dm_performance=True,
        sanity_check=False,
        save_period=1000 if data_name == 'cifar10' else 10,
        clip_denoised=False,
        clip_output=True,
        gpu_usage=False,
        eval_large_nfe=True,
    )

def cm_train_defaults(data_name):
    return dict(
        #teacher_model_path="/home/dongjun/EighthArticleExperimentalResults/CIFAR10/author_ckpt/edm-cifar10-32x32-uncond-vp.pkl" if data_name == 'cifar10' else "",
        teacher_model_path="/home/acf15618av/EighthArticleExperimentalResults/CIFAR10/author_ckpt/edm-cifar10-32x32-uncond-vp.pkl" if data_name == 'cifar10' else "",
        teacher_dropout=0.0 if data_name == 'cifar10' else 0.1,
        training_mode="ctm",
        target_ema_mode="fixed",
        scale_mode="fixed",
        total_training_steps=600000,
        start_ema=0.999,
        start_scales=18 if data_name == 'cifar10' else 40,
        end_scales=18 if data_name == 'cifar10' else 40,
        distill_steps_per_iter=50000,
        loss_norm="lpips",
        port=6,
    )

def model_and_diffusion_defaults(data_name):
    """
    Defaults for image training.
    """
    res = dict(
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7,
        image_size=32 if data_name == 'cifar10' else 64,
        num_channels=192,
        num_res_blocks=3,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=64,
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False if data_name == 'cifar10' else True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
        learn_sigma=False,
        weight_schedule="uniform",
        weight_schedule_multiplier=1.,
        diffusion_weight_schedule="karras_weight",
        rescaling=False,
    )
    return res

def train_defaults(data_name):
    """
        Defaults for model training.
    """
    res = dict(
        out_dir="",
        #data_dir="/home/dongjun/EighthArticleExperimentalResults/CIFAR10/train" if data_name == 'cifar10' else "",
        data_dir="/home/acf15618av/dataset/CIFAR10/train" if data_name == 'cifar10' else "",
        schedule_sampler="uniform",
        lr=0.0004 if data_name == 'cifar10' else 0.000008,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=128 if data_name == 'cifar10' else 2048,
        batch_size=-1,
        microbatch=64 if data_name.lower() == 'cifar10' else -1,  # -1 disables microbatches
        ema_rate="0.999,0.9999" if data_name == 'cifar10' else "0.999,0.9999,0.9999432189950708",
        # comma-separated list of EMA values
        log_interval=1000,
        save_interval=1000000,
        save_check_period=1000000,
        resume_checkpoint="",
        use_fp16=True,
        fp16_scale_growth=1e-3,
        device_id=0,
        num_workers=4,
        use_MPI=False,
        map_location='cuda',
    )
    return res


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file for single GPU.
    """
    # Directly load the state dictionary from the file
    state_dict = th.load(path, **kwargs)
    return state_dict

def create_ema_and_scales_fn(
    target_ema_mode,
    start_ema,
    scale_mode,
    start_scales,
    end_scales,
    total_steps,
    distill_steps_per_iter,
):
    def ema_and_scales_fn(step):
        if target_ema_mode == "fixed" and scale_mode == "fixed":
            target_ema = start_ema
            scales = start_scales
        else:
            print("check if other options are commented out")
            print("printing the target_ema_mode:: ", target_ema_mode)
            print("scale_mode:: ", scale_mode)
            raise NotImplementedError

        return float(target_ema), int(scales)

    return ema_and_scales_fn

def load_feature_extractor(args, eval=True):
    feature_extractor = None
    if args.loss_norm == 'lpips':
        from piq import LPIPS
        feature_extractor = LPIPS(replace_pooling=True, reduction="none")
    return feature_extractor

# i am cutting the GAN projector related function from here
# for GAN projector
class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]
    
class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)

class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = th.cat((x[:, self.start_index :], readout), -1)

        return self.project(features)

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x.contiguous()

def _resize_pos_embed(self, posemb, gs_h, gs_w):
    posemb_tok, posemb_grid = (
        posemb[:, : self.start_index],
        posemb[0, self.start_index :],
    )

    gs_old = int(math.sqrt(len(posemb_grid)))

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear", align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

    posemb = th.cat([posemb_tok, posemb_grid], dim=1)

    return posemb

def forward_flex(self, x):
    b, c, h, w = x.shape

    pos_embed = self._resize_pos_embed(
        self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
    )

    B = x.shape[0]

    if hasattr(self.patch_embed, "backbone"):
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)

    if hasattr(self, "dist_token") and self.dist_token is not None:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = th.cat((cls_tokens, dist_token, x), dim=1)
    else:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = th.cat((cls_tokens, x), dim=1)

    x = x + pos_embed
    x = self.pos_drop(x)

    for blk in self.blocks:
        x = blk(x)

    x = self.norm(x)

    return x

def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == "ignore":
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == "add":
        readout_oper = [AddReadout(start_index)] * len(features)
    elif use_readout == "project":
        readout_oper = [
            ProjectReadout(vit_features, start_index) for out_feat in features
        ]
    else:
        assert (
            False
        ), "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"

    return readout_oper

activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook

def _make_vit_b16_backbone(
    model,
    features=[96, 192, 384, 768],
    size=[384, 384],
    hooks=[2, 5, 8, 11],
    vit_features=768,
    use_readout="ignore",
    start_index=1,
):
    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

    # 32, 48, 136, 384
    pretrained.layer1 = nn.Sequential(
        readout_oper[0],
        Transpose(1, 2),
        nn.Unflatten(2, th.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[0],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[0],
            out_channels=features[0],
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.layer2 = nn.Sequential(
        readout_oper[1],
        Transpose(1, 2),
        nn.Unflatten(2, th.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[1],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[1],
            out_channels=features[1],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.layer3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, th.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )

    pretrained.layer4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, th.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.Conv2d(
            in_channels=features[3],
            out_channels=features[3],
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    )

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained

def calc_dims(pretrained, is_vit=False):
    dims = []
    inp_res = 256
    tmp = th.zeros(1, 3, inp_res, inp_res)
    if not is_vit:
        tmp = pretrained.layer0(tmp)
        dims.append(tmp.shape[1:3])
        tmp = pretrained.layer1(tmp)
        dims.append(tmp.shape[1:3])
        tmp = pretrained.layer2(tmp)
        dims.append(tmp.shape[1:3])
        tmp = pretrained.layer3(tmp)
        dims.append(tmp.shape[1:3])
    else:
        tmp = forward_vit(pretrained, tmp)
        dims = [out.shape[1:3] for out in tmp]

    # split to channels and resolution multiplier
    dims = np.array(dims)
    channels = dims[:, 0]
    res_mult = dims[:, 1] / inp_res
    return channels, res_mult

def _make_vit(model, name):
    # print(name)
    if 'base' in name:
        features = [96, 192, 384, 768]
        hooks = [2, 5, 8, 11]
        vit_features = 768
    else:
        print("NOPR")
    return _make_vit_b16_backbone(
    model,
    features=features,
    size=[224, 224],
    hooks=hooks,
    vit_features=vit_features,
    start_index=2 if 'deit' in name else 1,
    )

def _make_efficientnet(model):
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(
        model.conv_stem, model.bn1, model.act1, *model.blocks[0:2]
    )
    pretrained.layer1 = nn.Sequential(*model.blocks[2:3])
    pretrained.layer2 = nn.Sequential(*model.blocks[3:5])
    pretrained.layer3 = nn.Sequential(*model.blocks[5:9])
    return pretrained

def _make_pretrained(backbone, verbose=False):
    print("printing the name of backbone::::: ", backbone)

    if backbone == "deit_base_distilled_patch16_224":
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_vit(model, backbone)
        is_vit = True
    elif backbone == "tf_efficientnet_lite0":
        model = timm.create_model(backbone, pretrained=True)
        pretrained = _make_efficientnet(model)
        is_vit = False
    pretrained.CHANNELS, pretrained.RES_MULT = calc_dims(pretrained, is_vit)
    return pretrained

def get_backbone_normstats(backbone):
    if backbone :
        return {
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
        }
    else:
        print("check if the necesarry blocks are delted. it is in pg_modules.projector.py")
        raise NotImplementedError

def _make_scratch_ccm(scratch, in_channels, cout, expand=False):
    # shapes
    out_channels = [cout, cout*2, cout*4, cout*8] if expand else [cout]*4

    scratch.layer0_ccm = nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer1_ccm = nn.Conv2d(in_channels[1], out_channels[1], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer2_ccm = nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer3_ccm = nn.Conv2d(in_channels[3], out_channels[3], kernel_size=1, stride=1, padding=0, bias=True)

    scratch.CHANNELS = out_channels

    return scratch

def _make_scratch_csm(scratch, in_channels, cout, expand):
    scratch.layer3_csm = FeatureFusionBlock(in_channels[3], nn.ReLU(False), expand=expand, lowest=True)
    scratch.layer2_csm = FeatureFusionBlock(in_channels[2], nn.ReLU(False), expand=expand)
    scratch.layer1_csm = FeatureFusionBlock(in_channels[1], nn.ReLU(False), expand=expand)
    scratch.layer0_csm = FeatureFusionBlock(in_channels[0], nn.ReLU(False))

    # last refinenet does not expand to save channels in higher dimensions
    scratch.CHANNELS = [cout, cout, cout*2, cout*4] if expand else [cout]*4

    return scratch

def _make_projector(im_res, backbone, cout, proj_type, expand=False):
    assert proj_type in [0, 1, 2], "Invalid projection type"

    ### Build pretrained feature network
    pretrained = _make_pretrained(backbone)

    # Following Projected GAN
    im_res = 256
    pretrained.RESOLUTIONS = [im_res//4, im_res//8, im_res//16, im_res//32]

    if proj_type == 0: return pretrained, None
    # print(pretrained.CHANNELS)
    # print(pretrained)
    # exit()

    ### Build CCM
    scratch = nn.Module()
    scratch = _make_scratch_ccm(scratch, in_channels=pretrained.CHANNELS, cout=cout, expand=expand)

    pretrained.CHANNELS = scratch.CHANNELS

    if proj_type == 1: return pretrained, scratch

    ### build CSM
    scratch = _make_scratch_csm(scratch, in_channels=scratch.CHANNELS, cout=cout, expand=expand)

    # CSM upsamples x2 so the feature map resolution doubles
    pretrained.RESOLUTIONS = [res*2 for res in pretrained.RESOLUTIONS]
    pretrained.CHANNELS = scratch.CHANNELS

    return pretrained, scratch

def forward_vit(pretrained, x):
    b, c, h, w = x.shape
    _ = pretrained.model.forward_flex(x)

    layer_1 = pretrained.activations["1"]
    layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]
    layer_4 = pretrained.activations["4"]

    layer_1 = pretrained.layer1[0:2](layer_1)
    layer_2 = pretrained.layer2[0:2](layer_2)
    layer_3 = pretrained.layer3[0:2](layer_3)
    layer_4 = pretrained.layer4[0:2](layer_4)

    unflatten = nn.Sequential(
        nn.Unflatten(
            2,
            th.Size(
                [
                    h // pretrained.model.patch_size[1],
                    w // pretrained.model.patch_size[0],
                ]
            ),
        )
    )

    if layer_1.ndim == 3:
        layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3:
        layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3:
        layer_3 = unflatten(layer_3)
    if layer_4.ndim == 3:
        layer_4 = unflatten(layer_4)

    layer_1 = pretrained.layer1[3 : len(pretrained.layer1)](layer_1)
    layer_2 = pretrained.layer2[3 : len(pretrained.layer2)](layer_2)
    layer_3 = pretrained.layer3[3 : len(pretrained.layer3)](layer_3)
    layer_4 = pretrained.layer4[3 : len(pretrained.layer4)](layer_4)

    return layer_1, layer_2, layer_3, layer_4

class F_Identity(nn.Module):
    def forward(self, x):
        return x

class F_RandomProj(nn.Module):
    def __init__(
        self,
        backbone="tf_efficientnet_lite3",
        im_res=256,
        cout=64,
        expand=True,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        **kwargs,
    ):
        super().__init__()
        self.proj_type = proj_type
        self.backbone = backbone
        self.cout = cout
        self.expand = expand
        self.normstats = get_backbone_normstats(backbone)

        # build pretrained feature network and random decoder (scratch)
        self.pretrained, self.scratch = _make_projector(im_res=im_res, backbone=self.backbone, cout=self.cout,
                                                        proj_type=self.proj_type, expand=self.expand)
        self.CHANNELS = self.pretrained.CHANNELS
        self.RESOLUTIONS = self.pretrained.RESOLUTIONS

    def forward(self, x):
        # predict feature maps
        if self.backbone == "deit_base_distilled_patch16_224":
        # if self.backbone:
            out0, out1, out2, out3 = forward_vit(self.pretrained, x)
        # else:
        elif self.backbone == "tf_efficientnet_lite0":

            out0 = self.pretrained.layer0(x)
            out1 = self.pretrained.layer1(out0)
            out2 = self.pretrained.layer2(out1)
            out3 = self.pretrained.layer3(out2)

        # start enumerating at the lowest layer (this is where we put the first discriminator)
        out = {
            '0': out0,
            '1': out1,
            '2': out2,
            '3': out3,
        }

        if self.proj_type == 0: return out

        out0_channel_mixed = self.scratch.layer0_ccm(out['0'])
        out1_channel_mixed = self.scratch.layer1_ccm(out['1'])
        out2_channel_mixed = self.scratch.layer2_ccm(out['2'])
        out3_channel_mixed = self.scratch.layer3_ccm(out['3'])

        out = {
            '0': out0_channel_mixed,
            '1': out1_channel_mixed,
            '2': out2_channel_mixed,
            '3': out3_channel_mixed,
        }

        if self.proj_type == 1: return out

        # from bottom to top
        out3_scale_mixed = self.scratch.layer3_csm(out3_channel_mixed)
        out2_scale_mixed = self.scratch.layer2_csm(out3_scale_mixed, out2_channel_mixed)
        out1_scale_mixed = self.scratch.layer1_csm(out2_scale_mixed, out1_channel_mixed)
        out0_scale_mixed = self.scratch.layer0_csm(out1_scale_mixed, out0_channel_mixed)

        out = {
            '0': out0_scale_mixed,
            '1': out1_scale_mixed,
            '2': out2_scale_mixed,
            '3': out3_scale_mixed,
        }

        return out

def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def NormLayer(c, mode='batch'):
    if mode == 'group':
        return nn.GroupNorm(c//2, c)
    elif mode == 'batch':
        return nn.BatchNorm2d(c)

class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes, width=1):
        super().__init__()
        self.main = nn.Sequential(
            conv2d(in_planes, out_planes*width, 4, 2, 1, bias=True),
            NormLayer(out_planes*width),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat):
        return self.main(feat)

class DownBlockPatch(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.main = nn.Sequential(
            DownBlock(in_planes, out_planes),
            conv2d(out_planes, out_planes, 1, 1, 0, bias=False),
            NormLayer(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat):
        return self.main(feat)

class SingleDisc(nn.Module):
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, patch=False):
        super().__init__()

        # midas channels
        nfc_midas = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                     256: 32, 512: 16, 1024: 8}

        # interpolate for start sz that are not powers of two
        if start_sz not in nfc_midas.keys():
            sizes = np.array(list(nfc_midas.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = nfc_midas
        else:
            nfc = {k: ndf for k, v in nfc_midas.items()}

        # for feature map discriminators with nfc not in nfc_midas
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []

        # Head if the initial input is the full modality
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]

        # Down Blocks
        DB = DownBlockPatch if patch else DownBlock
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz], nfc[start_sz//2]))
            start_sz = start_sz // 2

        layers.append(conv2d(nfc[end_sz], 1, 4, 1, 0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        return self.main(x)

class MultiScaleD(nn.Module):
    def __init__(
        self,
        channels,
        resolutions,
        num_discs=4,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        cond=0,
        patch=False,
        **kwargs,
    ):
        super().__init__()

        assert num_discs in [1, 2, 3, 4, 5]

        # the first disc is on the lowest level of the backbone
        self.disc_in_channels = channels[:num_discs]
        self.disc_in_res = resolutions[:num_discs]
        Disc = SingleDisc

        mini_discs = []
        for i, (cin, res) in enumerate(zip(self.disc_in_channels, self.disc_in_res)):
            start_sz = res if not patch else 16
            mini_discs += [str(i), Disc(nc=cin, start_sz=start_sz, end_sz=8, patch=patch)],

        self.mini_discs = nn.ModuleDict(mini_discs)

    def forward(self, features, c, rec=False):
        all_logits = []
        for k, disc in self.mini_discs.items():
            all_logits.append(disc(features[k], c).view(features[k].size(0), -1))

        all_logits = th.cat(all_logits, dim=1)
        return all_logits

def load_discriminator_and_d_feature_extractor(args):
    #assert (args.gan_training == True) == (args.d_architecture == 'StyleGAN-XL')
    if args.gan_training:
        # from pg_modules.projector import F_RandomProj
        # from pg_modules.discriminator import MultiScaleD

        backbones = ['deit_base_distilled_patch16_224', 'tf_efficientnet_lite0']
        discriminator, discriminator_feature_extractor = [], []
        if args.gan_low_res_train:
            discriminator2 = []
        backbone_kwargs = {'im_res': args.image_size}
        # print("printing the backbones: ", backbones)
        # exit()
        for i, bb_name in enumerate(backbones):
            feat = F_RandomProj(bb_name, **backbone_kwargs)
            disc = MultiScaleD(
                channels=feat.CHANNELS,
                resolutions=feat.RESOLUTIONS,
                **backbone_kwargs,
            )
            discriminator_feature_extractor.append([bb_name, feat])
            discriminator.append([bb_name, disc])
            if args.gan_low_res_train:
                discriminator2.append([bb_name + '_low', disc])
        discriminator_feature_extractor = nn.ModuleDict(discriminator_feature_extractor)
        discriminator_feature_extractor = discriminator_feature_extractor.train(False).to(dev)
        discriminator_feature_extractor.requires_grad_(False)
        if args.gan_low_res_train:
            discriminator = discriminator + discriminator2
        discriminator = nn.ModuleDict(discriminator)
        discriminator.to(dev)
        discriminator.train()
        if args.use_d_fp16:
            discriminator.convert_to_fp16()
    else:
        discriminator, discriminator_feature_extractor = None, None
    return discriminator, discriminator_feature_extractor

### GAN part ends here

INITIAL_LOG_LOSS_SCALE = 20.0

def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def get_param_groups_and_shapes(named_model_params):
    named_model_params = list(named_model_params)
    scalar = []
    matrix = []
    for n, p in named_model_params:
        if p.ndim <= 1 and p.requires_grad:
            scalar.append((n, p))
        if p.ndim > 1 and p.requires_grad:
            matrix.append((n, p))
    scalar_vector_named_params = (
        scalar,
        (-1),
    )
    matrix_named_params = (
        matrix,
        (1, -1),
    )

    return [scalar_vector_named_params, matrix_named_params]

def make_master_params(param_groups_and_shapes):
    """
    Copy model parameters into a (differently-shaped) list of full-precision
    parameters.
    """
    master_params = []
    for param_group, shape in param_groups_and_shapes:
        #a = []
        #for _, param in param_group:
        #    if param.requires_grad == True:
        #        a.append([param.detach().float()])
        #master_param = nn.Parameter(
        #    _flatten_dense_tensors(a).view(shape)
        #)
        master_param = nn.Parameter(
            _flatten_dense_tensors(
                [param.detach().float() for (_, param) in param_group]
            ).view(shape)
        )
        master_param.requires_grad = True
        master_params.append(master_param)
    return master_params

def zero_master_grads(master_params):
    for param in master_params:
        param.grad = None

def zero_grad(model_params):
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

def model_grads_to_master_grads(param_groups_and_shapes, master_params):
    """
    Copy the gradients from the model parameters into the master parameters
    from make_master_params().
    """
    for master_param, (param_group, shape) in zip(
        master_params, param_groups_and_shapes
    ):
        master_param.grad = _flatten_dense_tensors(
            [param_grad_or_zeros(param) for (_, param) in param_group]
        ).view(shape)

def param_grad_or_zeros(param):
    if param.grad is not None:
        return param.grad.data.detach()
    else:
        return th.zeros_like(param)

def master_params_to_model_params(param_groups_and_shapes, master_params):
    """
    Copy the master parameter data back into the model parameters.
    """
    # Without copying to a list, if a generator is passed, this will
    # silently not copy any parameters.
    for master_param, (param_group, _) in zip(master_params, param_groups_and_shapes):
        for (_, param), unflat_master_param in zip(
            param_group, unflatten_master_params(param_group, master_param.view(-1))
        ):
            param.detach().copy_(unflat_master_param)

def unflatten_master_params(param_group, master_param):
    return _unflatten_dense_tensors(master_param, [param for (_, param) in param_group])

def master_params_to_state_dict(
    model, param_groups_and_shapes, master_params, use_fp16
):
    if use_fp16:
        state_dict = model.state_dict()
        for master_param, (param_group, _) in zip(
            master_params, param_groups_and_shapes
        ):
            for (name, _), unflat_master_param in zip(
                param_group, unflatten_master_params(param_group, master_param.view(-1))
            ):
                assert name in state_dict
                state_dict[name] = unflat_master_param
    else:
        state_dict = model.state_dict()
        for i, (name, _value) in enumerate(model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
    return state_dict

def state_dict_to_master_params(model, state_dict, use_fp16):
    if use_fp16:
        named_model_params = [
            (name, state_dict[name].to(dev)) for name, _ in model.named_parameters()
        ]
        param_groups_and_shapes = get_param_groups_and_shapes(named_model_params)
        master_params = make_master_params(param_groups_and_shapes)
    else:
        master_params = [state_dict[name] for name, _ in model.named_parameters()]
    return master_params

def check_overflow(value):
    return (value == float("inf")) or (value == -float("inf")) or (value != value)

class MixedPrecisionTrainer:
    def __init__(
        self,
        *,
        model,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,
    ):
        self.model = model
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.param_groups_and_shapes = None
        self.lg_loss_scale = initial_lg_loss_scale

        #for name, param in self.model.named_parameters():
        #    print(name, param.requires_grad)

        if self.use_fp16:
            self.param_groups_and_shapes = get_param_groups_and_shapes(
                self.model.named_parameters()
            )
            self.master_params = make_master_params(self.param_groups_and_shapes)
            self.model.convert_to_fp16()

    def zero_grad(self):
        zero_grad(self.model_params)

    def backward(self, loss: th.Tensor):
        if self.use_fp16:
            loss_scale = 2**self.lg_loss_scale
            #print("loss value: ", (loss*loss_scale).item())
            (loss * loss_scale).backward()
        else:
            loss.backward()

    def optimize(self, opt: th.optim.Optimizer):
        if self.use_fp16:
            return self._optimize_fp16(opt)
        else:
            return self._optimize_normal(opt)

    def _optimize_fp16(self, opt: th.optim.Optimizer):
        logkv_mean("lg_loss_scale", self.lg_loss_scale)
        model_grads_to_master_grads(self.param_groups_and_shapes, self.master_params)
        grad_norm, param_norm = self._compute_norms(grad_scale=2**self.lg_loss_scale)
        if check_overflow(grad_norm):
            self.lg_loss_scale -= 1
            log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            zero_master_grads(self.master_params)
            return False

        logkv_mean("grad_norm", grad_norm)
        logkv_mean("param_norm", param_norm)

        for p in self.master_params:
            p.grad.mul_(1.0 / (2**self.lg_loss_scale))

        opt.step()
        zero_master_grads(self.master_params)
        master_params_to_model_params(self.param_groups_and_shapes, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth
        return True

    def _optimize_normal(self, opt: th.optim.Optimizer):
        # model_grads_to_master_grads(self.param_groups_and_shapes, self.master_params)
        grad_norm, param_norm = self._compute_norms()
        logkv_mean("grad_norm", grad_norm)
        logkv_mean("param_norm", param_norm)
        opt.step()
        return True

    def _compute_norms(self, grad_scale=1.0):
        grad_norm = 0.0
        param_norm = 0.0
        for p in self.master_params:
            with th.no_grad():
                param_norm += th.norm(p, p=2, dtype=th.float32).item() ** 2
                if p.grad is not None:
                    grad_norm += th.norm(p.grad, p=2, dtype=th.float32).item() ** 2
        return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)

    def master_params_to_state_dict(self, master_params):
        return master_params_to_state_dict(
            self.model, self.param_groups_and_shapes, master_params, self.use_fp16
        )

    def state_dict_to_master_params(self, state_dict):
        return state_dict_to_master_params(self.model, state_dict, self.use_fp16)


class DummyGenerator:
    def randn(self, *args, **kwargs):
        return th.randn(*args, **kwargs)

    def randint(self, *args, **kwargs):
        return th.randint(*args, **kwargs)

    def randn_like(self, *args, **kwargs):
        return th.randn_like(*args, **kwargs)

class DeterministicGenerator:
    """
    RNG to deterministically sample num_samples samples that does not depend on batch_size or mpi_machines
    Uses a single rng and samples num_samples sized randomness and subsamples the current indices
    """

    def __init__(self, num_samples, seed=0):

        self.num_samples = num_samples
        self.done_samples = 0
        self.seed = seed
        self.rng_cpu = th.Generator()
        if th.cuda.is_available():
            self.rng_cuda = th.Generator(dev)
        self.set_seed(seed)

    def get_global_size_and_indices(self, size):
        global_size = (self.num_samples, *size[1:])
        indices = th.arange(self.done_samples, self.done_samples + size[0])
        indices = th.clamp(indices, 0, self.num_samples - 1)
        return global_size, indices

    def get_generator(self, device):
        return self.rng_cpu if th.device(device).type == "cpu" else self.rng_cuda

    def randn(self, *size, dtype=th.float, device="cpu"):
        global_size, indices = self.get_global_size_and_indices(size)
        generator = self.get_generator(device)
        return th.randn(*global_size, generator=generator, dtype=dtype, device=device)[indices]

    def randint(self, low, high, size, dtype=th.long, device="cpu"):
        global_size, indices = self.get_global_size_and_indices(size)
        generator = self.get_generator(device)
        return th.randint(low, high, generator=generator, size=global_size, dtype=dtype, device=device)[indices]

    def randn_like(self, tensor):
        size, dtype, device = tensor.size(), tensor.dtype, tensor.device
        return self.randn(*size, dtype=dtype, device=device)

    def set_done_samples(self, done_samples):
        self.done_samples = done_samples
        self.set_seed(self.seed)

    def get_seed(self):
        return self.seed

    def set_seed(self, seed):
        self.rng_cpu.manual_seed(seed)
        if th.cuda.is_available():
            self.rng_cuda.manual_seed(seed)


def get_generator(generator, num_samples=0, seed=0):
    if generator == "dummy":
        return DummyGenerator()
    elif generator == "determ":
        return DeterministicGenerator(num_samples, seed)
    else:
        print("check if the necessary module has been removed. check in cm.random_util.py")
        raise NotImplementedError

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None

def get_target_param_groups_and_shapes(named_model_params, source_named_model_params):
    named_model_params = list(named_model_params)
    source_named_model_params = {n: p for (n,p) in list(source_named_model_params)}
    scalar = []
    matrix = []
    for n, p in named_model_params:
        if p.ndim <= 1 and source_named_model_params[n].requires_grad:
            scalar.append((n, p))
        if p.ndim > 1 and source_named_model_params[n].requires_grad:
            matrix.append((n, p))
    scalar_vector_named_params = (
        scalar,
        (-1),
    )
    matrix_named_params = (
        matrix,
        (1, -1),
    )
    return [scalar_vector_named_params, matrix_named_params]

def karras_sample(
    diffusion,
    model,
    shape,
    steps,
    clip_denoised=True,
    progress=False,
    callback=None,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80,  # higher for highres?
    rho=7.0,
    sampler="heun",
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    generator=None,
    ts=None,
    x_T=None,
    ctm=False,
    teacher=False,
    clip_output=True,
    train=False,
    ind_1=0,
    ind_2=0,
    gamma=0.5,
):
    if generator is None:
        generator = get_generator("dummy")

    if sampler in ["progdist", 'euler', 'exact', 'cm_multistep', 'gamma_multistep']:
        sigmas = get_sigmas_karras(steps + 1, sigma_min, sigma_max, rho, device=device)
    else:
        sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)

    if x_T == None:
        x_T = generator.randn(*shape, device=device) * sigma_max

    sample_fn = {
        "heun": sample_heun,
        # "dpm": sample_dpm,
        # "ancestral": sample_euler_ancestral,
        # "onestep": sample_onestep,
        "exact": sample_exact,
        # "gamma": sample_gamma,
        # "gamma_multistep": sample_gamma_multistep,
        # "progdist": sample_progdist,
        # "euler": sample_euler,
        # "multistep": stochastic_iterative_sampler,
        # "cm_multistep": sample_multistep,
    }[sampler]
    # print(sampler)
    if sampler in ["heun", "dpm"]:
        sampler_args = dict(
            s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise
        )
    elif sampler in ["multistep", "exact", "cm_multistep"]:
        sampler_args = dict(
            ts=ts, t_min=sigma_min, t_max=sigma_max, rho=rho, steps=steps
        )
    elif sampler in ["gamma"]:
        sampler_args = dict(ind_1=ind_1, ind_2=ind_2)
    elif sampler in ["gamma_multistep"]:
        sampler_args = dict(
            ts=ts, t_min=sigma_min, t_max=sigma_max, rho=rho, steps=steps, gamma=gamma,
        )
    else:
        sampler_args = {}
    if sampler in ['heun']:
        sampler_args['teacher'] = False if train else teacher
        sampler_args['ctm'] = ctm
    #print("clip_denoised, clip_output: ", clip_denoised, clip_output)
    def denoiser(x_t, t, s=th.ones(x_T.shape[0], device=device)):
        denoised, G_theta = diffusion.get_denoised_and_G(model, x_t, t, s, ctm, teacher, **model_kwargs)
        if sampler in ['exact', 'cm_multistep', 'onestep', 'gamma', 'gamma_multistep']:
            denoised = G_theta
        if clip_denoised:
            #print("clip denoised!!!")
            denoised = denoised.clamp(-1, 1)
        return denoised

    x_0 = sample_fn(
        denoiser,
        x_T,
        sigmas,
        generator,
        progress=progress,
        callback=callback,
        **sampler_args,
    )
    if clip_output:
        #print("clip output")
        return x_0.clamp(-1, 1)
    return x_0

def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)

@th.no_grad()
def sample_heun(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    teacher=False,
    ctm=False,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        print("sigmas: ", sigmas[i], ctm, teacher)
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        if ctm:
            denoised = denoiser(x, sigma_hat * s_in, s=sigma_hat * s_in)
        else:
            #if teacher:
            denoised = denoiser(x, sigma_hat * s_in, s=None)
            #else:
            #    denoised = denoiser(x, sigma_hat * s_in, s=sigma_hat * s_in)
        #print("denoised: ", denoised[0][0][0][:3])
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
            #print("last")
        else:
            #print("no last")
            # Heun's method
            x_2 = x + d * dt
            if ctm:
                denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, s=sigmas[i + 1] * s_in)
            else:
                #if teacher:
                denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, s=None)
                #else:
                #    denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, s=sigmas[i + 1] * s_in)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x

@th.no_grad()
def sample_exact(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    ts=[],
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    if ts != [] and ts != None:
        sigmas = []
        t_max_rho = t_max ** (1 / rho)
        t_min_rho = t_min ** (1 / rho)
        s_in = x.new_ones([x.shape[0]])

        for i in range(len(ts)):
            sigmas.append((t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho)
        sigmas = th.tensor(sigmas)
        sigmas = append_zero(sigmas).to(x.device)
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)


    for i in indices[:-1]:
        sigma = sigmas[i]
        print(sigma, sigmas[i+1])
        if sigmas[i+1] != 0:
            denoised = denoiser(x, sigma * s_in, s=sigmas[i + 1] * s_in)
            x = denoised
        else:
            denoised = denoiser(x, sigma * s_in, s=sigma * s_in)
            d = to_d(x, sigma, denoised)
            dt = sigmas[i + 1] - sigma
            x = x + d * dt
        #else:
        #    denoised = denoiser(x, sigma * s_in)

        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised": denoised,
                }
            )
        #x = denoised
    return x

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = th.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)

def append_zero(x):
    return th.cat([x, x.new_zeros([1])])


class TrainLoop:
    def __init__(
        self,
        #*,
        model,
        discriminator,
        diffusion,
        data,
        batch_size,
        args=None,
    ):
        self.args = args
        self.model = model
        if self.args.sanity_check:
            for name, param in self.model.named_parameters():
                log("check and understand how consistency-type models override model parameters")
                log("model parameter before overriding: ", param.data.cpu().detach().reshape(-1)[:3])
                break
        self.discriminator = discriminator
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = args.microbatch if args.microbatch > 0 else batch_size
        self.lr = args.lr
        self.ema_rate = (
            [args.ema_rate]
            if isinstance(args.ema_rate, float)
            else [float(x) for x in args.ema_rate.split(",")]
        )
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size #* dist.get_world_size()
        self.fids = []
        self.generator = get_generator('determ', self.args.eval_num_samples, self.args.eval_seed)
        self.x_T = self.generator.randn(*(self.args.sampling_batch, self.args.in_channels, self.args.image_size, self.args.image_size),
                                        device='cpu') * self.args.sigma_max #.to(dist_util.dev())
        if self.args.class_cond:
            self.classes = self.generator.randint(0, self.args.num_classes, (self.args.sampling_batch,), device='cpu')
            if self.args.data_name.lower() == 'cifar10':
                self.classes.sort()

        self.sync_cuda = th.cuda.is_available()
        self._load_and_sync_parameters()
        if self.args.sanity_check:
            for name, param in self.model.named_parameters():
                log("model parameter after overriding: ", param.data.cpu().detach().reshape(-1)[:3])
                break
        if self.discriminator != None:
            if self.args.d_opt_load:
                if self.args.sanity_check:
                    for name, param in self.discriminator.named_parameters():
                        log("discriminator parameter before overriding: ", param.data.cpu().detach().reshape(-1)[:3])
                        break
                self._load_and_sync_discriminator_parameters()
                if self.args.sanity_check:
                    for name, param in self.discriminator.named_parameters():
                        log("discriminator parameter after overriding: ", param.data.cpu().detach().reshape(-1)[:3])
                        break
        print("loaded discriminator")
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
        )
        print("loaded mixed precision trainer")
        if self.args.sanity_check:
            log("mp trainer master parameter (should same to the model parameter if no linear_probing): ", self.mp_trainer.master_params[1].reshape(-1)[:3])

        self.opt = RAdam(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.args.weight_decay
        )
        #if self.args.sanity_check:
        #    print("opt state dict before overriding: ", self.opt.state_dict())

        if self.discriminator != None:
            self.d_mp_trainer = MixedPrecisionTrainer(
                model=self.discriminator,
                use_fp16=args.use_d_fp16,
                fp16_scale_growth=args.fp16_scale_growth,
            )
            self.d_opt = RAdam(
                self.d_mp_trainer.master_params, lr=args.d_lr, weight_decay=self.args.weight_decay, betas=(0.5, 0.9)
            )
        print('going to resume step')
        if self.resume_step:
            self._load_optimizer_state()
            if self.discriminator != None:
                try:
                    #if self.args.sanity_check:
                    #    print("discriminator opt state dict before overriding: ", self.d_opt.state_dict())
                    self._load_d_optimizer_state()
                    #if self.args.sanity_check:
                    #    print("discriminator opt state dict after overriding: ", self.d_opt.state_dict())
                except:
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!! warning !!!!!!!!!!!!!!!!!!!!!!!!!!!! discriminator optimizer not loaded successfully")
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = self.model #DDP(
            #     self.model,
            #     device_ids=[dist_util.dev()],
            #     output_device=dist_util.dev(),
            #     broadcast_buffers=False,
            #     bucket_cap_mb=128,
            #     find_unused_parameters=False,
            # )
            self.ddp_discriminator = None
            if self.args.gan_training:
                self.ddp_discriminator = self.discriminator #DDP(
                #     self.discriminator,
                #     device_ids=[dist_util.dev()],
                #     output_device=dist_util.dev(),
                #     broadcast_buffers=False,
                #     bucket_cap_mb=128,
                #     find_unused_parameters=False,
                # )

        else:
            # if dist.get_world_size() > 1:
            #     logger.warn(
            #         "Distributed training requires CUDA. "
            #         "Gradients will not be synchronized properly!"
            #     )
            self.use_ddp = False
            self.ddp_model = self.model

        self.step = self.resume_step

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.args.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            log(f"loading pretrained model from checkpoint: {resume_checkpoint}...")
            state_dict = th.load(resume_checkpoint, map_location=dev)#"cpu")
            self.model.load_state_dict(state_dict, strict=False)
            log(f"end loading pretrained model from checkpoint: {resume_checkpoint}...")

        # dist_util.sync_params(self.model.parameters())
        # dist_util.sync_params(self.model.buffers())
        log(f"end synchronizing pretrained model from GPU0 to all GPUs")

    def _load_and_sync_discriminator_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.args.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            resume_checkpoint = bf.join(bf.dirname(resume_checkpoint), f"d_model{self.resume_step:06}.pt")
            # if dist.get_rank() == 0:
            if os.path.exists(resume_checkpoint):
                log(f"loading discriminator model from checkpoint: {resume_checkpoint}...")
                #try:
                #if self.args.map_location == 'cuda':
                #    state_dict = dist_util.load_state_dict(
                #        resume_checkpoint, map_location=dist_util.dev()
                #    )
                #else:
                state_dict = th.load(resume_checkpoint, map_location="cpu")
                self.discriminator.load_state_dict(state_dict)
                log(f"end loading discriminator model from checkpoint: {resume_checkpoint}...")

        # dist_util.sync_params(self.discriminator.parameters())
        # dist_util.sync_params(self.discriminator.buffers())
        # log(f"end synchronizing discriminator from GPU0 to all GPUs")

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)
        if self.args.sanity_check:
            log(f"{rate} ema param before overriding: ", ema_params[1].reshape(-1)[:3])
        main_checkpoint = find_resume_checkpoint() or self.args.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            # if dist.get_rank() == 0:
            log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            if self.args.map_location == 'cuda':
                state_dict = th.load(ema_checkpoint, map_location= dev)#"cpu")
            else:
                state_dict = load_state_dict(
                    ema_checkpoint, map_location='cpu'
                )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)
            log(f"end loading EMA from checkpoint: {ema_checkpoint}...")

        # dist_util.sync_params(ema_params)
        log(f"end synchronizing EMA from GPU0 to all GPUs")
        if self.args.sanity_check:
            log(f"{rate} ema param after overriding: ", ema_params[1].reshape(-1)[:3])

        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.args.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            #if self.args.map_location == 'cuda':
            #    state_dict = dist_util.load_state_dict(
            #        opt_checkpoint, map_location=dist_util.dev()
            #    )
            #else:
            state_dict = th.load(opt_checkpoint, map_location="cpu")
            self.opt.load_state_dict(state_dict)
            log(f"end loading optimizer state from checkpoint: {opt_checkpoint}")

        if self.args.sanity_check:
            print("opt state dict after overriding: ", self.opt.state_dict()['state'])

    def _load_d_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.args.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"d_opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            log(f"loading d_optimizer state from checkpoint: {opt_checkpoint}")
            if os.path.exists(opt_checkpoint):
                if self.args.map_location == 'cuda':
                    state_dict = load_state_dict(
                        opt_checkpoint, map_location=dev
                    )
                else:
                    state_dict = th.load(opt_checkpoint, map_location="cpu")
                self.d_opt.load_state_dict(state_dict)
            log(f"end loading d_optimizer state from checkpoint: {opt_checkpoint}")

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.args.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.args.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logkv("step", self.step + self.resume_step)
        logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def sampling(self, model, sampler, ctm=None, teacher=False, step=-1, num_samples=-1, batch_size=-1, rate=0.999,
                 png=False, resize=True, generator=None, class_generator=None, sample_dir=''):
        if not teacher:
            model.eval()
        if step == -1:
            step = self.args.sampling_steps
        if batch_size == -1:
            batch_size = self.args.sampling_batch

        number = 0
        while num_samples > number:

            with th.no_grad():
                model_kwargs = {}
                if self.args.class_cond:
                    if self.args.train_classes >= 0:
                        classes = th.ones(size=(batch_size,), device=dev, dtype=int) * self.args.train_classes
                        model_kwargs["y"] = classes
                    elif self.args.train_classes == -2:
                        classes = [0, 1, 9, 11, 29, 31, 33, 55, 76, 89, 90, 130, 207, 250, 279, 281, 291, 323, 386, 387,
                                   388, 417, 562, 614, 759, 789, 800, 812, 848, 933, 973, 980]
                        assert batch_size % len(classes) == 0
                        model_kwargs["y"] = th.tensor([x for x in classes for _ in range(batch_size // len(classes))], device=dev)
                    else:
                        if class_generator != None:
                            model_kwargs["y"] = class_generator.randint(0, self.args.num_classes, (batch_size,), device=dev)
                        else:
                            if num_samples == -1:
                                model_kwargs["y"] = self.classes.to(dev)
                            else:
                                model_kwargs["y"] = th.randint(0, self.args.num_classes, size=(batch_size, ), device=dev)
                if generator != None:
                    x_T = generator.randn(*(batch_size, self.args.in_channels, self.args.image_size, self.args.image_size),
                                device=dev) * self.args.sigma_max
                    if self.args.large_log:
                        print("x_T: ", x_T[0][0][0][:3])
                else:
                    x_T = None

                sample = karras_sample(
                    diffusion=self.diffusion,
                    model=model,
                    shape=(batch_size, self.args.in_channels, self.args.image_size, self.args.image_size),
                    steps=step,
                    model_kwargs=model_kwargs,
                    device=dev,
                    clip_denoised=True if teacher else self.args.clip_denoised,
                    sampler=sampler,
                    generator=None,
                    teacher=teacher,
                    ctm=ctm if ctm != None else True if self.args.training_mode.lower() == 'ctm' else False,
                    x_T=x_T if generator != None else self.x_T.to(dev) if num_samples == -1 else None,
                    clip_output=self.args.clip_output,
                    sigma_min=self.args.sigma_min,
                    sigma_max=self.args.sigma_max,
                    train=False,
                )
                if resize:
                    sample = F.interpolate(sample, size=224, mode="bilinear")

                sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                sample = sample.permute(0, 2, 3, 1)
                sample = sample.contiguous()
                # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                # dist.all_gather(gathered_samples, sample)
                # all_images = [sample.cpu().numpy() for sample in gathered_samples]
                # arr = np.concatenate(all_images, axis=0)
                arr = sample.cpu().numpy()
            # if dist.get_rank() == 0:
                os.makedirs(bf.join(get_blob_logdir(), f"{sample_dir}"), exist_ok=True)
                if self.args.large_log:
                    print(f"saving to {bf.join(get_blob_logdir(), sample_dir)}")
                nrow = int(np.sqrt(arr.shape[0]))
                image_grid = make_grid(th.tensor(arr).permute(0, 3, 1, 2) / 255., nrow, padding=2)
                if num_samples == -1:
                    print("1")
                    with bf.BlobFile(bf.join(get_blob_logdir(), f"{'teacher_' if teacher else ''}sample_{sampler}_sampling_step_{step}_step_{self.step}.png"), "wb") as fout:
                        save_image(image_grid, fout)
                else:
                    print("2")
                    if generator != None:
                        print("3")
                        os.makedirs(bf.join(get_blob_logdir(), sample_dir),
                                    exist_ok=True)
                        np.savez(bf.join(get_blob_logdir(), f"{sample_dir}/sample_{number // arr.shape[0]}.npz"),
                                    arr)
                        if png and number <= 3000:
                            print("4")
                            with bf.BlobFile(bf.join(get_blob_logdir(),
                                                        f"{sample_dir}/sample_{number // arr.shape[0]}.png"), "wb") as fout:
                                save_image(image_grid, fout)
                    else:
                        print("5")
                        r = np.random.randint(100000000)
                        # if self.args.large_log:
                        #     log(f'{dist.get_rank()} number {number}')
                        log(bf.join(get_blob_logdir(), f"{sample_dir}"))
                        os.makedirs(bf.join(get_blob_logdir(), f"{sample_dir}"),
                                    exist_ok=True)
                        np.savez(bf.join(get_blob_logdir(), f"{sample_dir}/sample_{r}.npz"),
                                    arr)
                        if png and number <= 1000:
                            print("6")
                            with bf.BlobFile(bf.join(get_blob_logdir(),
                                                        f"{sample_dir}/sample_{r}.png"), "wb") as fout:
                                save_image(image_grid, fout)

                number += arr.shape[0]
                print(f"{number} number samples complete")
        if not teacher:
            model.train()

    def calculate_similarity_metrics(self, image_path, num_samples=50000, step=1, batch_size=100, rate=0.999, sampler='exact', log=True):
        files = glob.glob(os.path.join(image_path, 'sample*.npz'))
        files.sort()
        count = 0
        psnr = 0
        ssim = 0
        for i, file in enumerate(files):
            images = np.load(file)['arr_0']
            for k in range((images.shape[0] - 1) // batch_size + 1):
                #ref_img = self.ref_images[count + k * batch_size: count + (k + 1) * batch_size]
                if count + batch_size > num_samples:
                    remaining_num_samples = num_samples - count
                else:
                    remaining_num_samples = batch_size
                img = images[k * batch_size: k * batch_size + remaining_num_samples]
                ref_img = self.ref_images[count: count + remaining_num_samples]
                psnr += cv2.PSNR(img, ref_img) * remaining_num_samples
                ssim += SSIM_(img,ref_img,multichannel=True,channel_axis=3,data_range=255) * remaining_num_samples
                count = count + remaining_num_samples
                print(count)
                if count >= num_samples:
                    break
            if count >= num_samples:
                break
        assert count == num_samples
        print(count)
        psnr /= num_samples
        ssim /= num_samples
        assert num_samples % 1000 == 0
        if log:
            log(f"{self.step}-th step {sampler} sampler (NFE {step}) EMA {rate} PSNR-{num_samples // 1000}k: {psnr}, SSIM-{num_samples // 1000}k: {ssim}")
        else:
            return psnr, ssim

    def calculate_inception_stats(self, data_name, image_path, num_samples=50000, batch_size=100, device=th.device('cuda')):
        if data_name.lower() == 'cifar10':
            print(f'Loading images from "{image_path}"...')
            mu = th.zeros([self.feature_dim], dtype=th.float64, device=device)
            sigma = th.zeros([self.feature_dim, self.feature_dim], dtype=th.float64, device=device)
            files = glob.glob(os.path.join(image_path, 'sample*.npz'))
            count = 0
            for file in files:
                images = np.load(file)['arr_0']  # [0]#["samples"]
                for k in range((images.shape[0] - 1) // batch_size + 1):
                    mic_img = images[k * batch_size: (k + 1) * batch_size]
                    mic_img = th.tensor(mic_img).permute(0, 3, 1, 2).to(device)
                    features = self.detector_net(mic_img, **self.detector_kwargs).to(th.float64)
                    if count + mic_img.shape[0] > num_samples:
                        remaining_num_samples = num_samples - count
                    else:
                        remaining_num_samples = mic_img.shape[0]
                    mu += features[:remaining_num_samples].sum(0)
                    sigma += features[:remaining_num_samples].T @ features[:remaining_num_samples]
                    count = count + remaining_num_samples
                    print(count)
                    if count >= num_samples:
                        break
                if count >= num_samples:
                    break
            assert count == num_samples
            print(count)
            mu /= num_samples
            sigma -= mu.ger(mu) * num_samples
            sigma /= num_samples - 1
            mu = mu.cpu().numpy()
            sigma = sigma.cpu().numpy()
            return mu, sigma
        else:
            filenames = glob.glob(os.path.join(image_path, '*.npz'))
            imgs = []
            for file in filenames:
                try:
                    img = np.load(file)  # ['arr_0']
                    try:
                        img = img['data']
                    except:
                        img = img['arr_0']
                    imgs.append(img)
                except:
                    pass
            imgs = np.concatenate(imgs, axis=0)
            os.makedirs(os.path.join(image_path, 'single_npz'), exist_ok=True)
            np.savez(os.path.join(os.path.join(image_path, 'single_npz'), f'data'),
                     imgs)  # , labels)
            log("computing sample batch activations...")
            sample_acts = self.evaluator.read_activations(
                os.path.join(os.path.join(image_path, 'single_npz'), f'data.npz'))
            log("computing/reading sample batch statistics...")
            sample_stats, sample_stats_spatial = tuple(self.evaluator.compute_statistics(x) for x in sample_acts)
            with open(os.path.join(os.path.join(image_path, 'single_npz'), f'stats'), 'wb') as f:
                pickle.dump({'stats': sample_stats, 'stats_spatial': sample_stats_spatial}, f)
            with open(os.path.join(os.path.join(image_path, 'single_npz'), f'acts'), 'wb') as f:
                pickle.dump({'acts': sample_acts[0], 'acts_spatial': sample_acts[1]}, f)
            return sample_acts, sample_stats, sample_stats_spatial

    def compute_fid(self, mu, sigma, ref_mu=None, ref_sigma=None):
        if np.array(ref_mu == None).sum():
            ref_mu = self.mu_ref
            assert ref_sigma == None
            ref_sigma = self.sigma_ref
        m = np.square(mu - ref_mu).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma, ref_sigma), disp=False)
        fid = m + np.trace(sigma + ref_sigma - s * 2)
        fid = float(np.real(fid))
        return fid

    def calculate_inception_stats_npz(self, image_path, num_samples=50000, step=1, batch_size=100, device=th.device('cuda'),
                                      rate=0.999):
        print(f'Loading images from "{image_path}"...')
        mu = th.zeros([self.feature_dim], dtype=th.float64, device=device)
        sigma = th.zeros([self.feature_dim, self.feature_dim], dtype=th.float64, device=device)

        files = glob.glob(os.path.join(image_path, 'sample*.npz'))
        count = 0
        for file in files:
            images = np.load(file)['arr_0']  # [0]#["samples"]
            for k in range((images.shape[0] - 1) // batch_size + 1):
                mic_img = images[k * batch_size: (k + 1) * batch_size]
                mic_img = th.tensor(mic_img).permute(0, 3, 1, 2).to(device)
                features = self.detector_net(mic_img, **self.detector_kwargs).to(th.float64)
                if count + mic_img.shape[0] > num_samples:
                    remaining_num_samples = num_samples - count
                else:
                    remaining_num_samples = mic_img.shape[0]
                mu += features[:remaining_num_samples].sum(0)
                sigma += features[:remaining_num_samples].T @ features[:remaining_num_samples]
                count = count + remaining_num_samples
                log(count)
            if count >= num_samples:
                break
        assert count == num_samples
        print(count)
        mu /= num_samples
        sigma -= mu.ger(mu) * num_samples
        sigma /= num_samples - 1
        mu = mu.cpu().numpy()
        sigma = sigma.cpu().numpy()

        m = np.square(mu - self.mu_ref).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma, self.sigma_ref), disp=False)
        fid = m + np.trace(sigma + self.sigma_ref - s * 2)
        fid = float(np.real(fid))
        assert num_samples % 1000 == 0
        log(f"{self.step}-th step exact sampler (NFE {step}) EMA {rate} FID-{num_samples // 1000}k: {fid}")

def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

import nvidia_smi
import functools
import gc
import datetime


def log_loss_dict(losses):
    for key, values in losses.items():
        logkv_mean(f"{key} mean", values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        logkv_mean(f"{key} std", values.std().item())
        #for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
        #    quartile = int(4 * sub_t / diffusion.num_timesteps)
        #    logkv_mean(f"{key}_q{quartile}", sub_loss)

def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return get_dir()

class CTMTrainLoop(TrainLoop):
    def __init__(
        self,
        *,
        target_model,
        teacher_model,
        ema_scale_fn,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.training_mode = self.args.training_mode
        self.ema_scale_fn = ema_scale_fn
        self.target_model = target_model
        self.teacher_model = teacher_model
        self.total_training_steps = self.args.total_training_steps

        if target_model:
            if self.args.sanity_check:
                for name, param in self.target_model.named_parameters():
                    log("target model parameter before overriding: ", param.data.cpu().detach().reshape(-1)[:3])
                    break
            self._load_and_sync_ema_parameters_to_target_parameters()
            if self.args.sanity_check:
                # print("doing sanity check ---------------->>>>>>>>>")
                for name, param in self.target_model.named_parameters():
                    log("target model parameter after overriding: ", param.data.cpu().detach().reshape(-1)[:3])
                    break
            self.target_model.requires_grad_(False)
            self.target_model.train()

            if self.args.use_fp16:
                # print("usinig args.usefp16 ----------------->>>>>>>>>>>>>")
                self.target_model_param_groups_and_shapes = get_target_param_groups_and_shapes(
                    self.target_model.named_parameters(), self.model.named_parameters()
                )
                self.target_model_master_params = make_master_params(
                    self.target_model_param_groups_and_shapes
                )
            else:
                # print("In the else part of args.usefp16 ----------------->>>>")
                self.target_model_param_groups_and_shapes = list(self.target_model.named_parameters())
                self.target_model_master_params = list(target_model.parameters())
            for rate, params in zip(self.ema_rate, self.ema_params):
                if rate == 0.999:
                    log(f"loading target model from 0.999 ema...")
                    update_ema(
                        self.target_model_master_params,
                        params,
                        rate=0.0,
                    )
                    if self.args.use_fp16:
                        master_params_to_model_params(
                            self.target_model_param_groups_and_shapes,
                            self.target_model_master_params,
                        )
            if self.args.sanity_check:
                for name, param in self.target_model.named_parameters():
                    log("target model parameter after all: ", param.data.cpu().detach().reshape(-1)[:3])
                    break

        if teacher_model:
            #self._load_and_sync_teacher_parameters()
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()
        self.diffusion.teacher_model = teacher_model
        # print("printing the teacher model:::::::::: ", type(teacher_model))
        # print("printing the target_model model:::::::::: ", type(target_model))
        self.global_step = self.step
        self.initial_step = copy.deepcopy(self.step)
        if self.args.gpu_usage:
            nvidia_smi.nvmlInit()
            self.deviceCount = nvidia_smi.nvmlDeviceGetCount()
            self.print_gpu_usage('Before everything')
        # print('self.args.check_dm_performance', self.args.check_dm_performance)
        # exit()check_dm_performance
        # self.args.check_dm_performance = False # -------------------------->>>>>>>>> i have hardcoded this to false. but it was true for uncond
        if self.args.check_dm_performance and False: # ------------>>>>>> it is throwing error at the sampling part
            if not os.path.exists(self.args.dm_sample_path_seed_42):
                self.sampling(model=self.teacher_model, sampler='heun', teacher=True, step=18 if self.args.data_name.lower() == 'cifar10' else 40,
                              num_samples=self.args.eval_num_samples, batch_size=self.args.eval_batch,
                              rate=0.0, ctm=False, png=False, resize=False,
                              generator=get_generator('determ', self.args.eval_num_samples, self.args.eval_seed),
                              class_generator=get_generator('determ', self.args.eval_num_samples, 0),
                              sample_dir=self.args.dm_sample_path_seed_42)
        print(self.args.data_name.lower())
        if self.args.data_name.lower() == 'cifar10':
            print('Loading Inception-v3 model...')
            detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
            self.detector_kwargs = dict(return_features=True)
            self.feature_dim = 2048
            with open_url(detector_url, verbose=(0 == 0)) as f:
                self.detector_net = pickle.load(f).to(dev)
            with open_url(self.args.ref_path) as f:
                ref = dict(np.load(f))
            self.mu_ref = ref['mu']
            self.sigma_ref = ref['sigma']

    def print_gpu_usage(self, prefix=''):
        for i in range(self.deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            log(
                f"{prefix} |Device {i}| Mem Free: {mem.free / 1024 ** 2:5.2f}MB / {mem.total / 1024 ** 2:5.2f}MB | gpu-util: {util.gpu / 100.0:3.1%} | gpu-mem: {util.memory / 100.0:3.1%} |")


    def _load_and_sync_ema_parameters_to_target_parameters(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            if rate == self.args.start_ema: # 0.999
                log(f"loading target model from {self.args.start_ema} ema...")
                state_dict = self.mp_trainer.master_params_to_state_dict(params)
                self.target_model.load_state_dict(state_dict)
        log(f"end loading target model from {self.args.start_ema} ema...")

    def _load_and_sync_target_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.args.resume_checkpoint
        if resume_checkpoint:
            path, name = os.path.split(resume_checkpoint)
            target_name = name.replace("model", "target_model")
            resume_target_checkpoint = os.path.join(path, target_name)
            if bf.exists(resume_target_checkpoint) == 0:
                log(
                    f"loading target model from checkpoint: {resume_target_checkpoint}..."
                )

                state_dict = load_state_dict(
                    resume_target_checkpoint, map_location=dev
                )
                self.target_model.load_state_dict(state_dict, strict=False)



    def _load_and_sync_teacher_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.args.resume_checkpoint
        if resume_checkpoint:
            path, name = os.path.split(resume_checkpoint)
            teacher_name = name.replace("model", "teacher_model")
            resume_teacher_checkpoint = os.path.join(path, teacher_name)

            if bf.exists(resume_teacher_checkpoint)  == 0:
                log(
                    f"loading teacher model from checkpoint: {resume_teacher_checkpoint}..."
                )
                state_dict = load_state_dict(
                    resume_teacher_checkpoint, map_location=dev
                )
                self.teacher_model.load_state_dict(state_dict)#, strict=False)



    def run_loop(self):
        if self.args.gpu_usage:
            self.print_gpu_usage('Before training')
        saved = False
        while (
            self.step < self.args.lr_anneal_steps
            or self.global_step < self.total_training_steps
        ):
            batch, cond = next(self.data)
            if self.args.large_log:
                print("batch size: ", batch.shape)
                # print("rank: ", dist.get_rank())
            if self.args.intermediate_samples:
                #print("!!: ", self.step, dist.get_rank(), self.args.sample_interval, self.args.training_mode)
                #if dist.get_rank() == 0:
                if self.step == self.initial_step + 10 or (self.step % self.args.sample_interval == self.args.sample_interval - 1):
                    if self.args.training_mode.lower() == 'ctm':
                        if self.args.consistency_weight > 0.:
                            self.sampling(model=self.ddp_model, sampler='exact', num_samples=self.args.sampling_batch, png=True)
                            self.sampling(model=self.ddp_model, sampler='exact', step=2, num_samples=self.args.sampling_batch, png=True)
                            self.sampling(model=self.ddp_model, sampler='exact', step=1, num_samples=self.args.sampling_batch, png=True)
                        else:
                            self.sampling(model=self.ddp_model, sampler='heun', ctm=True, teacher=True, num_samples=self.args.sampling_batch, png=True)
                    elif self.args.training_mode.lower() == 'cd':
                        self.sampling(model=self.ddp_model, sampler='onestep', step=1, num_samples=self.args.sampling_batch, png=True)
                    elif self.args.training_mode.lower() == 'edm':
                        self.sampling(model=self.ddp_model, sampler='heun', step=40, num_samples=self.args.sampling_batch, png=True)
                if self.step == self.initial_step + 10 and self.teacher_model != None:
                    self.sampling(model=self.teacher_model, sampler='heun', ctm=False, teacher=True, num_samples=self.args.sampling_batch, png=True)
            self.run_step(batch, cond)
            if self.args.gpu_usage:
                self.print_gpu_usage('After one step training')
            if self.args.large_log:
                print("mp trainer master parameter after one step update: ", self.mp_trainer.master_params[1].reshape(-1)[:3])
                for name, param in self.model.named_parameters():
                    print("model parameter after one step update: ", param.data.cpu().detach().reshape(-1)[:3])
                    break
                for name, param in self.target_model.named_parameters():
                    print("target model parameter after one step update: ", param.data.cpu().detach().reshape(-1)[:3])
                    break
            if self.args.check_ctm_denoising_ability:
                self.eval(step=18, sampler='heun', teacher=True, ctm=True, rate=0.0)
            if (
                self.global_step
                and self.args.eval_interval != -1
                and self.global_step % self.args.eval_interval == self.args.eval_interval - 1
                #and self.step - self.initial_step > 10
                or self.step == self.args.lr_anneal_steps - 1
                or self.global_step == self.total_training_steps - 1
            ):
                if self.args.gpu_usage:
                    self.print_gpu_usage('Before emptying cache in evaluation 1')
                gc.collect()
                th.cuda.empty_cache()
                if self.args.gpu_usage:
                    self.print_gpu_usage('After emptying cache in evaluation 1')
                model_state_dict = self.model.state_dict()
                if self.args.linear_probing:
                    self.eval(step=18, sampler='heun', teacher=True, ctm=True, rate=0.0,
                              generator=get_generator('determ', self.args.eval_num_samples, self.args.eval_seed),
                              class_generator=get_generator('determ', self.args.eval_num_samples, 0),
                              delete=True)
                self.evaluation(0.0)
                log('Evaluation with model parameter end')
                for rate, params in zip(self.ema_rate, self.ema_params):
                    if not self.args.compute_ema_fids:
                        if rate != 0.999:
                            continue
                    state_dict = self.mp_trainer.master_params_to_state_dict(params)
                    self.model.load_state_dict(state_dict, strict=False)
                    self.evaluation(rate)
                    log(f'Evaluation with {rate}-EMA model parameter end')
                self.model.load_state_dict(model_state_dict, strict=True)
                del model_state_dict, state_dict
                if self.args.gpu_usage:
                    self.print_gpu_usage('Before emptying cache in evaluation 2')
                gc.collect()
                th.cuda.empty_cache()
                if self.args.gpu_usage:
                    self.print_gpu_usage('After emptying cache in evaluation 2')
            # dist.barrier()
            if (
                    self.global_step
                    and self.args.eval_interval != -1
                    and self.global_step % self.args.save_check_period == self.args.save_check_period - 1
                    #and self.step - self.initial_step > 10000
                    or self.step == self.args.lr_anneal_steps - 1
                    or self.global_step == self.total_training_steps - 1
            ):
                gc.collect()
                th.cuda.empty_cache()
                model_state_dict = self.model.state_dict()
                for rate, params in zip(self.ema_rate, self.ema_params):
                    if rate == 0.999:
                        state_dict = self.mp_trainer.master_params_to_state_dict(params)
                        self.model.load_state_dict(state_dict, strict=False)
                        fid = self.save_check(rate)
                    # if dist.get_rank() == 0:
                        assert fid != None
                        self.fids.append(fid)
                        save_ckpt = (self.fids[-1] == np.min(self.fids))
                        log("FID by iteration (NFE 1, EMA 0.999): ", self.fids)
                self.model.load_state_dict(model_state_dict, strict=True)
                del model_state_dict, state_dict
            # if dist.get_rank() == 0:
                if save_ckpt:
                    self.save(save_full=False)
                gc.collect()
                th.cuda.empty_cache()
            if self.args.large_log:
                print("mp trainer master parameter after sampling: ",
                      self.mp_trainer.master_params[1].reshape(-1)[:3])
                for name, param in self.model.named_parameters():
                    print("model parameter after sampling: ", param.data.cpu().detach().reshape(-1)[:3])
                    break
                for name, param in self.target_model.named_parameters():
                    print("target model parameter after sampling: ", param.data.cpu().detach().reshape(-1)[:3])
                    break

            saved = False
            if (
                self.global_step
                and self.args.save_interval != -1
                and self.global_step % self.args.save_interval == 0
            ):
                self.save()
                if self.discriminator != None:
                    self.d_save()
                saved = True
                gc.collect()
                th.cuda.empty_cache()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            if self.global_step % self.args.log_interval == 0:
                dumpkvs()
                log(datetime.datetime.now().strftime("SONY-%Y-%m-%d-%H-%M-%S"))
            if self.args.large_log:
                print("mp trainer master parameter after saving: ",
                      self.mp_trainer.master_params[1].reshape(-1)[:3])
                for name, param in self.model.named_parameters():
                    print("model parameter after saving: ", param.data.cpu().detach().reshape(-1)[:3])
                    break
                for name, param in self.target_model.named_parameters():
                    print("target model parameter after saving: ", param.data.cpu().detach().reshape(-1)[:3])
                    break
                print(f"0.999 ema param after overriding (should be same to the target parameter): ",
                      self.ema_params[0][1].reshape(-1)[:3])

        # Save the last checkpoint if it wasn't already saved.
        if not saved:
            self.save()
            if self.discriminator != None:
                self.d_save()

    def save_check(self, rate):
        if self.args.training_mode.lower() == 'ctm':
            assert rate == 0.999
            #fid = self.eval(step=1, rate=rate, ctm=True, delete=True, out=True)
            fid = self.eval(step=1, rate=rate, ctm=True, generator=get_generator('determ', self.args.eval_num_samples, self.args.eval_seed),
                                  class_generator=get_generator('determ', self.args.eval_num_samples, 0),
                          metric='similarity', delete=True, out=True)
            return fid

    def evaluation(self, rate):
        if self.args.training_mode.lower() == 'ctm':
            if self.args.eval_fid:
                self.eval(step=1, rate=rate, ctm=True, delete=True)
            if self.args.eval_similarity:
                self.eval(step=1, rate=rate, ctm=True, generator=get_generator('determ', self.args.eval_num_samples, self.args.eval_seed),
                                  class_generator=get_generator('determ', self.args.eval_num_samples, 0),
                          metric='similarity', delete=True)
            if self.args.eval_fid:
                self.eval(step=2, rate=rate, ctm=True, delete=True)
            if self.args.eval_similarity:
                self.eval(step=2, rate=rate, ctm=True,
                          generator=get_generator('determ', self.args.eval_num_samples, self.args.eval_seed),
                          class_generator=get_generator('determ', self.args.eval_num_samples, 0),
                          metric='similarity', delete=True)
            if self.args.compute_ema_fids:
                if self.args.eval_fid:
                    self.eval(step=4, rate=rate, ctm=True, delete=True)
                if self.args.eval_similarity:
                    self.eval(step=4, rate=rate, ctm=True, generator=get_generator('determ', self.args.eval_num_samples, self.args.eval_seed),
                              class_generator=get_generator('determ', self.args.eval_num_samples, 0),
                              metric='similarity', delete=True)
            step = 18 if self.args.data_name.lower() == 'cifar10' else 40
            if self.args.eval_large_nfe:
                if self.args.eval_fid:
                    self.eval(step=step, rate=rate, ctm=True, delete=True)
                if self.args.eval_similarity:
                    self.eval(step=step, rate=rate, ctm=True, generator=get_generator('determ', self.args.eval_num_samples, self.args.eval_seed),
                              class_generator=get_generator('determ', self.args.eval_num_samples, 0),
                              metric='similarity', delete=True)

        elif self.args.training_mode.lower() == 'cm':
            if self.args.eval_fid:
                self.eval(step=1, sampler='onestep', rate=rate, ctm=False, delete=True)

    def run_step(self, batch, cond):
        if self.args.large_log:
            print("mp trainer master parameter before update: ", self.mp_trainer.master_params[1].reshape(-1)[:3])
            for name, param in self.model.named_parameters():
                print("model parameter before update: ", param.data.cpu().detach().reshape(-1)[:3])
                break
            for name, param in self.target_model.named_parameters():
                print("target model parameter before update: ", param.data.cpu().detach().reshape(-1)[:3])
                break
        self.forward_backward(batch, cond)
        if self.discriminator == None:
            took_step = self.mp_trainer.optimize(self.opt)
            if took_step:
                self._update_ema()
                if self.target_model:
                    self._update_target_ema()
                self.step += 1
                self.global_step += 1
        else:
            if self.step % self.args.g_learning_period == 0:
                took_step = self.mp_trainer.optimize(self.opt)
            else:
                took_step = self.d_mp_trainer.optimize(self.d_opt)
            # print(self.step, took_step)
            if took_step:
                if self.step % self.args.g_learning_period == 0:
                    self._update_ema()
                    if self.target_model:
                        self._update_target_ema()
                self.step += 1
                self.global_step += 1
        self._anneal_lr()
        self.log_step()

    def _update_target_ema(self):
        target_ema, scales = self.ema_scale_fn(self.global_step)
        with th.no_grad():
            update_ema(
                self.target_model_master_params,
                self.mp_trainer.master_params,
                rate=target_ema,
            )
            if self.args.use_fp16:
                master_params_to_model_params(
                    self.target_model_param_groups_and_shapes,
                    self.target_model_master_params,
                )

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        if self.discriminator != None:
            self.d_mp_trainer.zero_grad()
        num_heun_step = [self.diffusion.get_num_heun_step(num_heun_step=self.args.num_heun_step)]
        if self.args.gan_specific_time:
            gan_num_heun_step = [self.diffusion.get_num_heun_step(num_heun_step=self.args.gan_num_heun_step,
                                                                  heun_step_strategy=self.args.gan_heun_step_strategy)]
        diffusion_training_ = [np.random.rand() < self.args.diffusion_training_frequency]
        gan_training_ = [np.random.rand() < self.args.gan_training_frequency]
        # dist.broadcast_object_list(num_heun_step, 0)
        # if self.args.gan_specific_time:
            # dist.broadcast_object_list(gan_num_heun_step, 0)
        # dist.broadcast_object_list(diffusion_training_, 0)
        # dist.broadcast_object_list(gan_training_, 0)
        num_heun_step = num_heun_step[0]
        if self.args.gan_specific_time:
            gan_num_heun_step = gan_num_heun_step[0]
        else:
            gan_num_heun_step = -1
        diffusion_training_ = diffusion_training_[0]
        gan_training_ = gan_training_[0]

        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dev)
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            compute_losses = functools.partial(
                self.diffusion.ctm_losses,
                step=self.step,
                model=self.ddp_model,
                x_start=micro,
                model_kwargs=micro_cond,
                target_model=self.target_model,
                discriminator=self.ddp_discriminator,
                init_step=self.initial_step,
                ctm=True if self.training_mode.lower() == 'ctm' else False,
                num_heun_step=num_heun_step,
                gan_num_heun_step=gan_num_heun_step,
                diffusion_training_=diffusion_training_,
                gan_training_=gan_training_,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                if self.step % self.args.g_learning_period == 0:
                    # with self.ddp_model.no_sync():
                    losses = compute_losses()
                else:
                    # with self.ddp_discriminator.no_sync():
                    losses = compute_losses()

            if 'consistency_loss' in list(losses.keys()):
                # print("Consistency learning")
                loss = self.args.consistency_weight * losses["consistency_loss"].mean()

                if 'd_loss' in list(losses.keys()):
                    if self.args.large_log:
                        print("GAN learning, ", self.args.discriminator_weight, losses['d_loss'].mean())
                    loss = loss + self.args.discriminator_weight * losses['d_loss'].mean()
                if 'denoising_loss' in list(losses.keys()):
                    loss = loss + self.args.denoising_weight * losses['denoising_loss'].mean()
                log_loss_dict({k: v.view(-1) for k, v in losses.items()})
                if self.args.sanity_check:
                    # print("rank: ", dist.get_rank())
                    for name, param in self.model.named_parameters():
                        print("model parameter gradient for current microbatch: ",
                              th.autograd.grad(outputs=(2**self.mp_trainer.lg_loss_scale) * loss, inputs=param, retain_graph=True)[0].reshape(-1)[:3])
                        break
                self.mp_trainer.backward(loss)

            elif 'd_loss' in list(losses.keys()):
                assert self.step % self.args.g_learning_period != 0
                loss = (losses["d_loss"]).mean()
                self.d_mp_trainer.backward(loss)
                if self.args.large_log:
                    for param in self.discriminator.parameters():
                        try:
                            print("discriminator param data, grad: ", param.grad.reshape(-1)[:3])
                        except:
                            print("discriminator param grad: ", param.grad)
                        break

            elif 'denoising_loss' in list(losses.keys()):
                loss = losses['denoising_loss'].mean()
                log_loss_dict({k: v.view(-1) for k, v in losses.items()})
                self.mp_trainer.backward(loss)
            if self.args.sanity_check:
                # print("rank: ", dist.get_rank())
                for name, param in self.model.named_parameters():
                    print("model parameter gradient across all microbatch: ", param.grad.cpu().detach().reshape(-1)[:3])
                    break

    @th.no_grad()
    def eval(self, step=1, sampler='exact', teacher=False, ctm=False, rate=0.999, generator=None, class_generator=None, metric='fid', delete=False, out=False):
        model = self.model
        sample_dir = f"{self.step}_{sampler}_{step}_{rate}"
        if generator != None:
            sample_dir = sample_dir + "_seed_42"
        self.sampling(model=model, sampler=sampler, teacher=teacher, step=step,
                      num_samples=self.args.eval_num_samples, batch_size=self.args.eval_batch,
                      rate=rate, ctm=ctm, png=True, resize=False, generator=generator,
                      class_generator=class_generator, sample_dir=sample_dir)
        gc.collect()
        th.cuda.empty_cache()
    # if dist.get_rank() == 0:
        if self.args.data_name.lower() == 'cifar10':
            if metric == 'fid':
                mu, sigma = self.calculate_inception_stats(self.args.data_name,
                                                            os.path.join(get_blob_logdir(), sample_dir),
                                                            num_samples=self.args.eval_num_samples)
                log(f"{self.step}-th step {sampler} sampler (NFE {step}) EMA {rate}"
                            f" FID-{self.args.eval_num_samples // 1000}k: {self.compute_fid(mu, sigma)}")
            if metric == 'similarity':
                mu, sigma = self.calculate_inception_stats(self.args.data_name,
                                                            os.path.join(get_blob_logdir(), sample_dir),
                                                            num_samples=self.args.eval_num_samples)
                log(f"{self.step}-th step {sampler} sampler (NFE {step}) seed 42 EMA {rate}"
                            f" FID-{self.args.eval_num_samples // 1000}k: {self.compute_fid(mu, sigma)}")
            if self.args.check_dm_performance and False:
                log(f"{self.step}-th step {sampler} sampler (NFE {step}) EMA {rate}"
                            f" FID-{self.args.eval_num_samples // 1000}k compared with DM: {self.compute_fid(mu, sigma, self.dm_mu, self.dm_sigma)}")
                self.calculate_similarity_metrics(os.path.join(get_blob_logdir(), sample_dir),
                                                    num_samples=self.args.eval_num_samples, step=step, rate=rate)
            if delete:
                shutil.rmtree(os.path.join(get_blob_logdir(), sample_dir))
            if out:
                return self.compute_fid(mu, sigma)
        else:
            sample_acts, sample_stats, sample_stats_spatial = self.calculate_inception_stats(self.args.data_name,
                                                                            bf.join(get_blob_logdir(), sample_dir),
                                                                            num_samples=self.args.eval_num_samples)
            log(f"Inception Score-{self.args.eval_num_samples // 1000}k:", self.evaluator.compute_inception_score(sample_acts[0]))
            log(f"FID-{self.args.eval_num_samples // 1000}k:", sample_stats.frechet_distance(self.ref_stats))
            log(f"sFID-{self.args.eval_num_samples // 1000}k:", sample_stats_spatial.frechet_distance(self.ref_stats_spatial))
            prec, recall = self.evaluator.compute_prec_recall(self.ref_acts[0], sample_acts[0])
            log("Precision:", prec)
            log("Recall:", recall)
            #self.evaluator.sess.close()
            #tf.reset_default_graph()

    def save(self, save_full=True):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
        # if dist.get_rank() == 0:
            log(f"saving model {rate}...")
            if not rate:
                filename = f"model{self.global_step:06d}.pt"
            else:
                filename = f"ema_{rate}_{self.global_step:06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        for rate, params in zip(self.ema_rate, self.ema_params):
            if not save_full:
                if rate == 0.999:
                    save_checkpoint(rate, params)
            else:
                save_checkpoint(rate, params)
        if save_full:
            log("saving optimizer state...")
        # if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{self.global_step:06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        # if dist.get_rank() == 0:
            if self.target_model:
                log("saving target model state")
                filename = f"target_model{self.global_step:06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(self.target_model.state_dict(), f)
            if self.teacher_model and self.training_mode == "progdist":
                log("saving teacher model state")
                filename = f"teacher_model{self.global_step:06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(self.teacher_model.state_dict(), f)

            # Save model parameters last to prevent race conditions where a restart
            # loads model at step N, but opt/ema state isn't saved for step N.
            save_checkpoint(0, self.mp_trainer.master_params)
            # dist.barrier()

    def d_save(self):
        log("saving d_optimizer state...")
    # if dist.get_rank() == 0:
        with bf.BlobFile(
            bf.join(get_blob_logdir(), f"d_opt{self.global_step:06d}.pt"),
            "wb",
        ) as f:
            th.save(self.d_opt.state_dict(), f)
        with bf.BlobFile(bf.join(get_blob_logdir(), f"d_model{self.global_step:06d}.pt"), "wb") as f:
            th.save(self.d_mp_trainer.master_params_to_state_dict(self.d_mp_trainer.master_params), f)

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but opt/ema state isn't saved for step N.
        # dist.barrier()

    def log_step(self):
        step = self.global_step
        logkv("step", step)
        logkv("samples", (step + 1) * self.global_batch)

