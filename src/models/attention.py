"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""
import torch

# attention layer for knowledge distillation for different tasks
class SelfAttention(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.attention = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

        with torch.no_grad():
            self.attention.weight.copy_(torch.zeros_like(self.attention.weight))

    def forward(self, x):
        features = self.conv(x)
        attention_mask = torch.sigmoid(self.attention(x))
        return features * attention_mask
