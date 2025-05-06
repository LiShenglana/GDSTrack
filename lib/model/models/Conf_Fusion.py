import torch.nn as nn
import torch
class Conf_Fusion(nn.Module):
    """
    Fusion N_mem memory features with confidence-value paradigm
    """

    def __init__(self, in_channels=256, out_channels=256):
        super(Conf_Fusion, self).__init__()

        self.conf_gen = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.value_gen = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        batch, mem_size, channel, h, w = x.shape
        x = x.view(-1, channel, h, w)

        # Calc confidence on each position
        confidence = self.conf_gen(x)
        confidence = torch.clamp(confidence, max=4, min=-6)
        # Softmax each confidence map across all confidence maps
        confidence = torch.exp(confidence)
        confidence = confidence.view(batch, mem_size, channel, h, w)
        confidence_sum = confidence.sum(dim=1).view(batch, 1, channel, h, w).repeat(1, mem_size, 1, 1, 1)
        confidence_norm = confidence / confidence_sum

        # The raw value for output (not weighted yet)
        value = self.value_gen(x)
        value = value.view(batch, mem_size, channel, h, w)

        # Weighted sum of the value maps, with confidence maps as element-wise weights
        out = confidence_norm * value
        out = value
        out = out.sum(dim=1)

        return out
