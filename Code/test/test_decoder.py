import sys

sys.path.append('../')
from model import MyTransformerEncoderLayer
from model import MyTransformerEncoder
from model import MyTransformerDecoder
from model import MyTransformerDecoderLayer

import torch
import torch.nn as nn

if __name__ == '__main__':
    src_len = 5
    batch_size = 2
    d_model = 32
    tgt_len = 6
    num_head = 8
    num_layers = 2
    src = torch.rand((src_len, batch_size, d_model))  # shape: [src_len, batch_size, embed_dim]
    src_key_padding_mask = torch.tensor([[False, False, False, True, True],
                                         [False, False, False, False, True]])  # shape: [batch_size, src_len]

    tgt = torch.rand((tgt_len, batch_size, d_model))  # shape: [tgt_len, batch_size, embed_dim]
    tgt_key_padding_mask = torch.tensor([[False, False, False, False, True, True],
                                         [False, False, False, True, True, True]])  # shape: [batch_size, tgt_len]

    encoder_layer = MyTransformerEncoderLayer(d_model, num_head, dim_feedforward=64)
    encoder = MyTransformerEncoder(encoder_layer, num_layers)
    memory = encoder(src, src_key_padding_mask=src_key_padding_mask)

    decoder_layer = MyTransformerDecoderLayer(d_model, num_head, dim_feedforward=64)
    decoder = MyTransformerDecoder(decoder_layer, num_layers)
    result = decoder(tgt, memory, tgt_key_padding_mask=tgt_key_padding_mask,
                     memory_key_padding_mask=src_key_padding_mask)

    print(result.shape)
