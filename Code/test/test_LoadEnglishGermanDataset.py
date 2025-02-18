import sys

sys.path.append('../')

from utils import LoadEnglishGermanDataset
from utils import my_tokenizer
from config import Config

if __name__ == '__main__':
    config = Config()
    data_loader = LoadEnglishGermanDataset(config.train_corpus_file_paths,
                                           batch_size=config.batch_size,
                                           tokenizer=my_tokenizer,
                                           min_freq=config.min_freq)
    train_iter, valid_iter, test_iter = data_loader.load_train_val_test_data(config.train_corpus_file_paths,
                                                                             config.val_corpus_file_paths,
                                                                             config.test_corpus_file_paths)
    print(data_loader.PAD_IDX)
    print('-------------------------')
    for src, tgt in train_iter:
        tgt_input = tgt[:-1, :]
        tgt_out = tgt[1:, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = data_loader.create_mask(src, tgt_input)
        print("src shape：", src.shape)  # [de_tensor_len,batch_size]
        print(src.transpose(0, 1)[:3])
        print("tgt shape:", tgt.shape)  # [de_tensor_len,batch_size]
        print("src input shape:", src.shape)
        print("src_padding_mask shape (batch_size, src_len): ", src_padding_mask.shape)
        print("tgt input shape:", tgt_input.shape)  # [tgt_len,batch_size]
        print(tgt_input.transpose(0, 1)[:3])
        print("tgt_padding_mask shape: (batch_size, tgt_len) ", tgt_padding_mask.shape)
        print("tgt output shape:", tgt_out.shape)  # [tgt_len,batch_size]
        print(tgt_out.transpose(0, 1)[:3])
        print("tgt_mask shape (tgt_len,tgt_len): ", tgt_mask.shape)
        break

# src shape： torch.Size([15, 5])
# tensor([[10, 33,  7, 34, 35, 36, 37, 38, 39, 40, 11, 41,  8, 42,  4],
#         [10, 27, 28, 29,  7,  8, 30, 31, 32,  4,  1,  1,  1,  1,  1],
#         [ 9,  6, 43, 44, 45, 11, 46, 47, 48,  4,  1,  1,  1,  1,  1]])
# tgt shape: torch.Size([17, 5])
# src input shape: torch.Size([15, 5])
# src_padding_mask shape (batch_size, src_len):  torch.Size([5, 15])
# tgt input shape: torch.Size([16, 5])
# tensor([[ 2, 11, 33, 10,  5, 34, 35, 36, 37, 38,  5, 39, 40,  5, 41,  4],
#         [ 2, 11, 27, 28, 29, 30,  5, 31, 32,  4,  3,  1,  1,  1,  1,  1],
#         [ 2,  8,  9,  7, 42, 43, 44, 45, 46,  4,  3,  1,  1,  1,  1,  1]])
# tgt_padding_mask shape: (batch_size, tgt_len)  torch.Size([5, 16])
# tgt output shape: torch.Size([16, 5])
# tensor([[11, 33, 10,  5, 34, 35, 36, 37, 38,  5, 39, 40,  5, 41,  4,  3],
#         [11, 27, 28, 29, 30,  5, 31, 32,  4,  3,  1,  1,  1,  1,  1,  1],
#         [ 8,  9,  7, 42, 43, 44, 45, 46,  4,  3,  1,  1,  1,  1,  1,  1]])
# tgt_mask shape (tgt_len,tgt_len):  torch.Size([15, 15])