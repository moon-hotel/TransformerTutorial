import sys

sys.path.append('../')
from copy import deepcopy
from config import Config
from model import TranslationModel
from model import CustomSchedule
from utils import LoadEnglishGermanDataset
from utils import my_tokenizer
import torch
import time
import os
import logging


def accuracy(logits, y_true, PAD_IDX):
    """
    :param logits:  [tgt_len,batch_size,tgt_vocab_size]
    :param y_true:  [tgt_len,batch_size]
    :param PAD_IDX:
    :return:
    """
    y_pred = logits.transpose(0, 1).argmax(axis=2).reshape(-1)
    # 将 [tgt_len,batch_size,tgt_vocab_size] 转成 [batch_size, tgt_len,tgt_vocab_size]
    y_true = y_true.transpose(0, 1).reshape(-1)
    # 将 [tgt_len,batch_size] 转成 [batch_size， tgt_len]
    acc = y_pred.eq(y_true)  # 计算预测值与正确值比较的情况
    mask = torch.logical_not(y_true.eq(PAD_IDX))  # 找到真实标签中，mask位置的信息。 mask位置为FALSE，非mask位置为TRUE
    acc = acc.logical_and(mask)  # 去掉acc中mask的部分
    correct = acc.sum().item()
    total = mask.sum().item()
    return float(correct) / total, correct, total


def train_model(config):
    logging.info("############载入数据集############")
    data_loader = LoadEnglishGermanDataset(config.train_corpus_file_paths,
                                           batch_size=config.batch_size,
                                           tokenizer=my_tokenizer,
                                           min_freq=config.min_freq)
    logging.info("############划分数据集############")
    train_iter, valid_iter, test_iter = \
        data_loader.load_train_val_test_data(config.train_corpus_file_paths,
                                             config.val_corpus_file_paths,
                                             config.test_corpus_file_paths)
    logging.info("############初始化模型############")
    translation_model = TranslationModel(src_vocab_size=len(data_loader.de_vocab),
                                         tgt_vocab_size=len(data_loader.en_vocab),
                                         d_model=config.d_model,
                                         nhead=config.num_head,
                                         num_encoder_layers=config.num_encoder_layers,
                                         num_decoder_layers=config.num_decoder_layers,
                                         dim_feedforward=config.dim_feedforward,
                                         dropout=config.dropout)
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        translation_model.load_state_dict(loaded_paras)
        logging.info("#### 成功载入已有模型，进行追加训练...")
    translation_model = translation_model.to(config.device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=data_loader.PAD_IDX)

    optimizer = torch.optim.Adam(translation_model.parameters(),
                                 lr=0.,
                                 betas=(config.beta1, config.beta2), eps=config.epsilon)
    lr_scheduler = CustomSchedule(config.d_model, optimizer=optimizer)
    translation_model.train()
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (src, tgt) in enumerate(train_iter):
            src = src.to(config.device)  # [src_len, batch_size]
            tgt = tgt.to(config.device)
            tgt_input = tgt[:-1, :]  # 解码部分的输入, [tgt_len,batch_size]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask \
                = data_loader.create_mask(src, tgt_input, config.device)
            logits = translation_model(
                src=src,  # Encoder的token序列输入，[src_len,batch_size]
                tgt=tgt_input,  # Decoder的token序列输入,[tgt_len,batch_size]
                src_mask=src_mask,  # Encoder的注意力Mask输入，这部分其实对于Encoder来说是没有用的
                tgt_mask=tgt_mask,
                # Decoder的注意力Mask输入，用于掩盖当前position之后的position [tgt_len,tgt_len]
                src_key_padding_mask=src_padding_mask,  # 用于mask掉Encoder的Token序列中的padding部分
                tgt_key_padding_mask=tgt_padding_mask,  # 用于mask掉Decoder的Token序列中的padding部分
                memory_key_padding_mask=src_padding_mask)  # 用于mask掉Encoder的Token序列中的padding部分
            # logits 输出shape为[tgt_len,batch_size,tgt_vocab_size]
            optimizer.zero_grad()
            tgt_out = tgt[1:, :]  # 解码部分的真实值  shape: [tgt_len,batch_size]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            # [tgt_len*batch_size, tgt_vocab_size] with [tgt_len*batch_size, ]
            loss.backward()
            lr_scheduler.step()
            optimizer.step()
            losses += loss.item()
            acc, _, _ = accuracy(logits, tgt_out, data_loader.PAD_IDX)
            msg = f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], Train loss :{loss.item():.3f}, Train acc: {acc}"
            logging.info(msg)
        end_time = time.time()
        train_loss = losses / len(train_iter)
        msg = f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s"
        logging.info(msg)
        if epoch % 2 == 0:
            acc = evaluate(config, valid_iter, translation_model, data_loader)
            logging.info(f"Accuracy on validation{acc:.3f}")
            state_dict = deepcopy(translation_model.state_dict())
            torch.save(state_dict, model_save_path)


def evaluate(config, valid_iter, model, data_loader):
    model.eval()
    correct, totals = 0, 0
    with torch.no_grad():
        for idx, (src, tgt) in enumerate(valid_iter):
            src = src.to(config.device)
            tgt = tgt.to(config.device)
            tgt_input = tgt[:-1, :]  # 解码部分的输入

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = \
                data_loader.create_mask(src, tgt_input, device=config.device)

            logits = model(src=src,  # Encoder的token序列输入，
                           tgt=tgt_input,  # Decoder的token序列输入
                           src_mask=src_mask,  # Encoder的注意力Mask输入，这部分其实对于Encoder来说是没有用的
                           tgt_mask=tgt_mask,  # Decoder的注意力Mask输入，用于掩盖当前position之后的position
                           src_key_padding_mask=src_padding_mask,  # 用于mask掉Encoder的Token序列中的padding部分
                           tgt_key_padding_mask=tgt_padding_mask,  # 用于mask掉Decoder的Token序列中的padding部分
                           memory_key_padding_mask=src_padding_mask)  # 用于mask掉Encoder的Token序列中的padding部分
            tgt_out = tgt[1:, :]  # 解码部分的真实值  shape: [tgt_len,batch_size]
            _, c, t = accuracy(logits, tgt_out, data_loader.PAD_IDX)
            correct += c
            totals += t
    model.train()
    return float(correct) / totals


if __name__ == '__main__':
    config = Config()
    train_model(config)

# [2025-02-20 22:19:35] - INFO: ############初始化模型############
# [2025-02-20 22:19:36] - INFO: Epoch: 0, Batch[0/8], Train loss :5.954, Train acc: 0.0010863661053775121
# [2025-02-20 22:19:36] - INFO: Epoch: 0, Batch[1/8], Train loss :5.954, Train acc: 0.004441976679622432
# [2025-02-20 22:19:36] - INFO: Epoch: 0, Batch[2/8], Train loss :5.955, Train acc: 0.004514672686230248
# [2025-02-20 22:19:36] - INFO: Epoch: 0, Batch[3/8], Train loss :5.958, Train acc: 0.002763957987838585
# [2025-02-20 22:19:36] - INFO: Epoch: 0, Batch[4/8], Train loss :5.940, Train acc: 0.0033821871476888386
# [2025-02-20 22:19:36] - INFO: Epoch: 0, Batch[5/8], Train loss :5.940, Train acc: 0.0016675931072818232
# [2025-02-20 22:19:36] - INFO: Epoch: 0, Batch[6/8], Train loss :5.964, Train acc: 0.003125
# [2025-02-20 22:19:37] - INFO: Epoch: 0, Batch[7/8], Train loss :5.954, Train acc: 0.002320185614849188
# [2025-02-20 22:19:37] - INFO: Epoch: 0, Train loss: 5.952, Epoch time = 0.676s
# [2025-02-20 22:19:37] - INFO: Accuracy on validation0.001
# [2025-02-20 22:19:37] - INFO: Epoch: 1, Batch[0/8], Train loss :5.961, Train acc: 0.001664816870144284
# [2025-02-20 22:19:37] - INFO: Epoch: 1, Batch[1/8], Train loss :5.947, Train acc: 0.002788622420524261
# [2025-02-20 22:19:37] - INFO: Epoch: 1, Batch[2/8], Train loss :5.939, Train acc: 0.004491858506457047
# [2025-02-20 22:19:37] - INFO: Epoch: 1, Batch[3/8], Train loss :5.940, Train acc: 0.002658160552897395
# [2025-02-20 22:19:37] - INFO: Epoch: 1, Batch[4/8], Train loss :5.931, Train acc: 0.004489337822671156
# [2025-02-20 22:19:37] - INFO: Epoch: 1, Batch[5/8], Train loss :5.938, Train acc: 0.0016844469399213925
# [2025-02-20 22:19:37] - INFO: Epoch: 1, Batch[6/8], Train loss :5.960, Train acc: 0.0021344717182497333
# [2025-02-20 22:19:37] - INFO: Epoch: 1, Batch[7/8], Train loss :5.952, Train acc: 0.001718213058419244
# [2025-02-20 22:19:37] - INFO: Epoch: 1, Train loss: 5.946, Epoch time = 0.232s
# [2025-02-20 22:19:37] - INFO: Epoch: 2, Batch[0/8], Train loss :5.926, Train acc: 0.0021321961620469083
# [2025-02-20 22:19:37] - INFO: Epoch: 2, Batch[1/8], Train loss :5.944, Train acc: 0.002186987424822307
# [2025-02-20 22:19:37] - INFO: Epoch: 2, Batch[2/8], Train loss :5.944, Train acc: 0.003206841261357563
# [2025-02-20 22:19:37] - INFO: Epoch: 2, Batch[3/8], Train loss :5.931, Train acc: 0.0027808676307007787
# [2025-02-20 22:19:37] - INFO: Epoch: 2, Batch[4/8], Train loss :5.930, Train acc: 0.005906674542232723
# [2025-02-20 22:19:37] - INFO: Epoch: 2, Batch[5/8], Train loss :5.941, Train acc: 0.0022271714922048997
# [2025-02-20 22:19:37] - INFO: Epoch: 2, Batch[6/8], Train loss :5.927, Train acc: 0.004296455424274973
# [2025-02-20 22:19:37] - INFO: Epoch: 2, Batch[7/8], Train loss :5.927, Train acc: 0.0029154518950437317
# [2025-02-20 22:19:37] - INFO: Epoch: 2, Train loss: 5.934, Epoch time = 0.231s
# [2025-02-20 22:19:37] - INFO: Accuracy on validation0.001
# [2025-02-20 22:19:37] - INFO: Epoch: 3, Batch[0/8], Train loss :5.932, Train acc: 0.0005254860746190226
# [2025-02-20 22:19:37] - INFO: Epoch: 3, Batch[1/8], Train loss :5.916, Train acc: 0.002799552071668533
# [2025-02-20 22:19:37] - INFO: Epoch: 3, Batch[2/8], Train loss :5.915, Train acc: 0.0022197558268590455
# [2025-02-20 22:19:37] - INFO: Epoch: 3, Batch[3/8], Train loss :5.896, Train acc: 0.004378762999452655
# [2025-02-20 22:19:37] - INFO: Epoch: 3, Batch[4/8], Train loss :5.917, Train acc: 0.002186987424822307
# [2025-02-20 22:19:37] - INFO: Epoch: 3, Batch[5/8], Train loss :5.934, Train acc: 0.0038022813688212928
# [2025-02-20 22:19:37] - INFO: Epoch: 3, Batch[6/8], Train loss :5.907, Train acc: 0.003952569169960474
# [2025-02-20 22:19:37] - INFO: Epoch: 3, Batch[7/8], Train loss :5.904, Train acc: 0.005353955978584176
# [2025-02-20 22:19:37] - INFO: Epoch: 3, Train loss: 5.915, Epoch time = 0.239s
# [2025-02-20 22:19:37] - INFO: Epoch: 4, Batch[0/8], Train loss :5.910, Train acc: 0.007103825136612022
# [2025-02-20 22:19:37] - INFO: Epoch: 4, Batch[1/8], Train loss :5.892, Train acc: 0.007198228128460687
# [2025-02-20 22:19:37] - INFO: Epoch: 4, Batch[2/8], Train loss :5.915, Train acc: 0.0065111231687466084
# [2025-02-20 22:19:37] - INFO: Epoch: 4, Batch[3/8], Train loss :5.886, Train acc: 0.003316749585406302
# [2025-02-20 22:19:38] - INFO: Epoch: 4, Batch[4/8], Train loss :5.883, Train acc: 0.0066518847006651885
# [2025-02-20 22:19:38] - INFO: Epoch: 4, Batch[5/8], Train loss :5.886, Train acc: 0.005461496450027308
# [2025-02-20 22:19:38] - INFO: Epoch: 4, Batch[6/8], Train loss :5.877, Train acc: 0.0011013215859030838
# [2025-02-20 22:19:38] - INFO: Epoch: 4, Batch[7/8], Train loss :5.876, Train acc: 0.005291005291005291
# [2025-02-20 22:19:38] - INFO: Epoch: 4, Train loss: 5.891, Epoch time = 0.232s
# [2025-02-20 22:19:38] - INFO: Accuracy on validation0.002
# [2025-02-20 22:19:38] - INFO: Epoch: 5, Batch[0/8], Train loss :5.894, Train acc: 0.0038997214484679664
# [2025-02-20 22:19:38] - INFO: Epoch: 5, Batch[1/8], Train loss :5.853, Train acc: 0.0038461538461538464
# [2025-02-20 22:19:38] - INFO: Epoch: 5, Batch[2/8], Train loss :5.895, Train acc: 0.006073992269464384
# [2025-02-20 22:19:38] - INFO: Epoch: 5, Batch[3/8], Train loss :5.858, Train acc: 0.005789473684210527
# [2025-02-20 22:19:38] - INFO: Epoch: 5, Batch[4/8], Train loss :5.840, Train acc: 0.003968253968253968
# [2025-02-20 22:19:38] - INFO: Epoch: 5, Batch[5/8], Train loss :5.864, Train acc: 0.007526881720430108
# [2025-02-20 22:19:38] - INFO: Epoch: 5, Batch[6/8], Train loss :5.870, Train acc: 0.002720348204570185
# [2025-02-20 22:19:38] - INFO: Epoch: 5, Batch[7/8], Train loss :5.843, Train acc: 0.006053268765133172
# [2025-02-20 22:19:38] - INFO: Epoch: 5, Train loss: 5.865, Epoch time = 0.232s
# [2025-02-20 22:19:38] - INFO: Epoch: 6, Batch[0/8], Train loss :5.838, Train acc: 0.006063947078280044
# [2025-02-20 22:19:38] - INFO: Epoch: 6, Batch[1/8], Train loss :5.854, Train acc: 0.005577244841048522
# [2025-02-20 22:19:38] - INFO: Epoch: 6, Batch[2/8], Train loss :5.843, Train acc: 0.005866666666666667
# [2025-02-20 22:19:38] - INFO: Epoch: 6, Batch[3/8], Train loss :5.821, Train acc: 0.004947773501924134
# [2025-02-20 22:19:38] - INFO: Epoch: 6, Batch[4/8], Train loss :5.835, Train acc: 0.0053475935828877
# [2025-02-20 22:19:38] - INFO: Epoch: 6, Batch[5/8], Train loss :5.819, Train acc: 0.004904632152588556
# [2025-02-20 22:19:38] - INFO: Epoch: 6, Batch[6/8], Train loss :5.807, Train acc: 0.005005561735261402
# [2025-02-20 22:19:38] - INFO: Epoch: 6, Batch[7/8], Train loss :5.801, Train acc: 0.004889975550122249
# [2025-02-20 22:19:38] - INFO: Epoch: 6, Train loss: 5.827, Epoch time = 0.231s
# [2025-02-20 22:19:38] - INFO: Accuracy on validation0.004
# [2025-02-20 22:19:38] - INFO: Epoch: 7, Batch[0/8], Train loss :5.799, Train acc: 0.00482573726541555
# [2025-02-20 22:19:38] - INFO: Epoch: 7, Batch[1/8], Train loss :5.808, Train acc: 0.006087437742114001
# [2025-02-20 22:19:38] - INFO: Epoch: 7, Batch[2/8], Train loss :5.784, Train acc: 0.007974481658692184
# [2025-02-20 22:19:38] - INFO: Epoch: 7, Batch[3/8], Train loss :5.790, Train acc: 0.006310958118187034
# [2025-02-20 22:19:38] - INFO: Epoch: 7, Batch[4/8], Train loss :5.784, Train acc: 0.011758417958311064
# [2025-02-20 22:19:38] - INFO: Epoch: 7, Batch[5/8], Train loss :5.804, Train acc: 0.007332205301748449
# [2025-02-20 22:19:38] - INFO: Epoch: 7, Batch[6/8], Train loss :5.792, Train acc: 0.007146783947223749
# [2025-02-20 22:19:38] - INFO: Epoch: 7, Batch[7/8], Train loss :5.771, Train acc: 0.00594883997620464
# [2025-02-20 22:19:38] - INFO: Epoch: 7, Train loss: 5.791, Epoch time = 0.230s
# [2025-02-20 22:19:38] - INFO: Epoch: 8, Batch[0/8], Train loss :5.791, Train acc: 0.009468700683850605
# [2025-02-20 22:19:39] - INFO: Epoch: 8, Batch[1/8], Train loss :5.775, Train acc: 0.009085746734809767
# [2025-02-20 22:19:39] - INFO: Epoch: 8, Batch[2/8], Train loss :5.759, Train acc: 0.01627384960718294
# [2025-02-20 22:19:39] - INFO: Epoch: 8, Batch[3/8], Train loss :5.739, Train acc: 0.014199890770071
# [2025-02-20 22:19:39] - INFO: Epoch: 8, Batch[4/8], Train loss :5.749, Train acc: 0.014665942422596416
# [2025-02-20 22:19:39] - INFO: Epoch: 8, Batch[5/8], Train loss :5.748, Train acc: 0.012279765082754938
# [2025-02-20 22:19:39] - INFO: Epoch: 8, Batch[6/8], Train loss :5.728, Train acc: 0.015013404825737266
# [2025-02-20 22:19:39] - INFO: Epoch: 8, Batch[7/8], Train loss :5.736, Train acc: 0.017654476670870115
# [2025-02-20 22:19:39] - INFO: Epoch: 8, Train loss: 5.753, Epoch time = 0.231s
# [2025-02-20 22:19:39] - INFO: Accuracy on validation0.021
# [2025-02-20 22:19:39] - INFO: Epoch: 9, Batch[0/8], Train loss :5.733, Train acc: 0.024828314844162706
# [2025-02-20 22:19:39] - INFO: Epoch: 9, Batch[1/8], Train loss :5.731, Train acc: 0.02491506228765572
# [2025-02-20 22:19:39] - INFO: Epoch: 9, Batch[2/8], Train loss :5.709, Train acc: 0.028556034482758622
# [2025-02-20 22:19:39] - INFO: Epoch: 9, Batch[3/8], Train loss :5.708, Train acc: 0.03986892408519935
# [2025-02-20 22:19:39] - INFO: Epoch: 9, Batch[4/8], Train loss :5.712, Train acc: 0.03874227961819203
# [2025-02-20 22:19:39] - INFO: Epoch: 9, Batch[5/8], Train loss :5.708, Train acc: 0.039226519337016576
# [2025-02-20 22:19:39] - INFO: Epoch: 9, Batch[6/8], Train loss :5.704, Train acc: 0.04538087520259319
# [2025-02-20 22:19:39] - INFO: Epoch: 9, Batch[7/8], Train loss :5.702, Train acc: 0.06476997578692494
# [2025-02-20 22:19:39] - INFO: Epoch: 9, Train loss: 5.713, Epoch time = 0.231s
