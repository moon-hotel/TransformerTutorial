import sys

sys.path.append('../')

from model import TranslationModel

import torch
import torch.nn as nn

if __name__ == '__main__':
    src_len = 6
    batch_size = 2
    d_model = 32
    tgt_len = 5
    num_head = 8
    num_layers = 2
    src = torch.tensor([[4, 5, 2, 7, 5, 3],
                        [6, 7, 4, 3, 2, 1]]).transpose(0, 1)  # shape: [src_len, batch_size]
    src_key_padding_mask = torch.tensor([[False, False, False, False, True, True],
                                         [False, False, False, False, False, True]])  # shape: [batch_size, src_len]

    tgt = torch.tensor([[4, 2, 2, 7, 5],
                        [6, 0, 1, 3, 2]]).transpose(0, 1)  # shape: [tgt_len, batch_size]
    tgt_key_padding_mask = torch.tensor([[False, False, False, True, True],
                                         [False, False, True, True, True]])  # shape: [batch_size, tgt_len]

    trans_model = TranslationModel(src_vocab_size=10, tgt_vocab_size=8,
                                   d_model=d_model, nhead=num_head, num_decoder_layers=5, num_encoder_layers=2,
                                   dim_feedforward=10, dropout=0.1)
    tgt_mask = trans_model.my_transformer.generate_square_subsequent_mask(tgt_len)
    logits = trans_model(src, tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                         src_key_padding_mask=src_key_padding_mask,
                         memory_key_padding_mask=src_key_padding_mask)
    print(logits.shape)