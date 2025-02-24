import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oneformer3d import ptv2_encoder, ptv3_encoder
from utils import count_parameters
from oneformer3d.ptv2_encoder import PTV2Encoder
encoder = PTV2Encoder()
# encoder = PTV2Encoder()
sample = torch.randn(100, 6)
print(encoder)
print('parameters:', count_parameters(encoder))  # 24064
print(encoder(sample).shape)
