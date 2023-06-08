import torch
from torch import Tensor
import torch.nn.functional as F
import math


def add_timing_signal_nd(x:Tensor, min_timescale=5.0, max_timescale=1.0e4, device='cpu'):
            """Adds a bunch of sinusoids of different frequencies to a Tensor.
            Each channel of the input Tensor is incremented by a sinusoid of a different
            frequency and phase in one of the positional dimensions.
            This allows attention to learn to use absolute and relative positions.
            Timing signals should be added to some precursors of both the query and the
            memory inputs to attention.
            The use of relative position is possible because sin(a+b) and cos(a+b) can be
            experessed in terms of b, sin(a) and cos(a).
            x is a Tensor with n "positional" dimensions, e.g. one dimension for a
            sequence or two dimensions for an image
            We use a geometric sequence of timescales starting with
            min_timescale and ending with max_timescale.  The number of different
            timescales is equal to channels // (n * 2). For each timescale, we
            generate the two sinusoidal signals sin(timestep/timescale) and
            cos(timestep/timescale).  All of these sinusoids are concatenated in
            the channels dimension.
            Args:
                x: a Tensor with shape [batch, d1 ... dn, channels]
                min_timescale: a float
                max_timescale: a float
            Returns:
                a Tensor the same shape as x.
            """
            num_dims = 2
            channels = x.shape[-1]
            num_timescales = channels // (num_dims * 2)
            log_timescale_increment = (
                    math.log(float(max_timescale) / float(min_timescale)) /
                    (torch.Tensor([num_timescales]).type(torch.float32) - 1))
            inv_timescales = min_timescale * torch.exp(
                    torch.Tensor(range(0, num_timescales)).type(torch.float32) * -log_timescale_increment)
            for dim in range(num_dims):
                length = x.shape[dim + 1]
                position = torch.Tensor(range(0, length)).type(torch.float32) 
                scaled_time = torch.unsqueeze(position, 1) * torch.unsqueeze(
                        inv_timescales, 0)
                signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=1)
                prepad = dim * 2 * num_timescales
                postpad = channels - (dim + 1) * 2 * num_timescales
                signal = F.pad(signal, (prepad, postpad, 0, 0))
                for _ in range(1 + dim):
                    signal = torch.unsqueeze(signal, 0)
                for _ in range(num_dims - 1 - dim):
                    signal = torch.unsqueeze(signal, -2)
                x = x + signal.to(device)
            return x