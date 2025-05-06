from torchprofile import profile_macs
import torch
inputs = torch.randn(1,3,224,224)
macs = profile_macs(model, inputs)
flops = 2 * macs
print(f"FLOPS: {flops:,}")