import torch
from torch import nn
from models.torch.add_timing_signal import add_timing_signal_nd
from models.torch.convnext import ConvNextEncoder
from models.torch.resnet import ResBlock, ResNet34


class Encoder(nn.Module):
        def __init__(self, emb_dim, device):
            super().__init__()
            self.device = device
            self.resnet = ResNet34(3, ResBlock)
            # self.convnext = ConvNextEncoder(3, stem_features=64, depths=[3,4,6,4], widths=[256, 512, 1024, 2048])
            self.fc = nn.LazyLinear(out_features=emb_dim)

        def forward(self, x):
            x = self.resnet(x)
            # x = self.convnext(x)
            x = x.permute(0, 2, 3, 1)
            x = add_timing_signal_nd(x, min_timescale=10.0, device=self.device)
            x = torch.reshape(x, (x.shape[0], -1, x.shape[3]))
            x = torch.relu(self.fc(x))
            
            return x