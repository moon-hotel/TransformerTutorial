import sys

sys.path.append('../')
from model import MyMultiheadAttention

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
    # ============ 测试 MyMultiheadAttention ============
    my_mh = MyMultiheadAttention(embed_dim=d_model, num_heads=num_head)
    r = my_mh(src, src, src, key_padding_mask=src_key_padding_mask)
    print(r[0].shape)  # [src_len, batch_size, embed_dim]

    my_mh = MyMultiheadAttention(embed_dim=d_model, num_heads=num_head)
    r = my_mh(tgt, src, src, key_padding_mask=src_key_padding_mask)
    print(r[0].shape)

    mh = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_head)
    r = mh(tgt, src, src, key_padding_mask=src_key_padding_mask)
    print(r[0].shape)
