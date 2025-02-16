import sys

sys.path.append('../')
from model import MyTransformerEncoderLayer

import torch
import torch.nn as nn

if __name__ == '__main__':
    src_len = 5
    batch_size = 2
    d_model = 32
    tgt_len = 6
    num_head = 8
    src = torch.rand((src_len, batch_size, d_model))  # shape: [src_len, batch_size, embed_dim]
    src_key_padding_mask = torch.tensor([[False, False, False, True, True],
                                         [False, False, False, False, True]])  # shape: [batch_size, src_len]

    tgt = torch.rand((tgt_len, batch_size, d_model))  # shape: [tgt_len, batch_size, embed_dim]

    encoder_layer = MyTransformerEncoderLayer(d_model, num_head, dim_feedforward=64)
    result = encoder_layer(src, src_key_padding_mask=src_key_padding_mask)
    print(result.shape)