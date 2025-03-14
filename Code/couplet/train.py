import sys

sys.path.append('../')
from copy import deepcopy
from config import CoupleConfig
from model import CoupletModel
from utils import LoadCoupletDataset
import torch
import time
from torch.optim.lr_scheduler import LambdaLR
import os
import logging


def get_customized_schedule_with_warmup(optimizer, num_warmup_steps=4000,
                                        d_model=512, last_epoch=-1):
    def lr_lambda(current_step):
        current_step += 1
        arg1 = current_step ** -0.5
        arg2 = current_step * (num_warmup_steps ** -1.5)
        return (d_model ** -0.5) * min(arg1, arg2)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


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
    data_loader = LoadCoupletDataset(config.train_corpus_file_paths,
                                     batch_size=config.batch_size,
                                     min_freq=config.min_freq,
                                     max_len=config.max_len)
    logging.info("############划分数据集############")
    train_iter, test_iter = \
        data_loader.load_train_val_test_data(config.train_corpus_file_paths,
                                             config.test_corpus_file_paths)
    logging.info("############初始化模型############")
    couplet_model = CoupletModel(vocab_size=len(data_loader.vocab),
                                 d_model=config.d_model,
                                 nhead=config.num_head,
                                 num_encoder_layers=config.num_encoder_layers,
                                 num_decoder_layers=config.num_decoder_layers,
                                 dim_feedforward=config.dim_feedforward,
                                 dropout=config.dropout)
    if os.path.exists(config.model_save_path):
        loaded_paras = torch.load(config.model_save_path)
        couplet_model.load_state_dict(loaded_paras)
        logging.info("#### 成功载入已有模型，进行追加训练...")
    couplet_model = couplet_model.to(config.device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=data_loader.PAD_IDX)
    optimizer = torch.optim.Adam(couplet_model.parameters(), lr=1.0, betas=(config.beta1, config.beta2),
                                 eps=config.epsilon)
    scheduler = get_customized_schedule_with_warmup(optimizer, num_warmup_steps=config.num_warmup_steps,
                                                    d_model=config.d_model)
    couplet_model.train()
    max_test_acc = 0
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (src, tgt) in enumerate(train_iter):
            src = src.to(config.device)  # [src_len, batch_size]
            tgt = tgt.to(config.device)
            tgt_input = tgt[:-1, :]  # 解码部分的输入, [tgt_len,batch_size]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask \
                = data_loader.create_mask(src, tgt_input, config.device)
            logits = couplet_model(
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
            tgt_output = tgt[1:, :]  # 解码部分的真实值  shape: [tgt_len,batch_size]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1)) / src.shape[1]
            # [tgt_len*batch_size, tgt_vocab_size] with [tgt_len*batch_size, ]
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses += loss.item()
            acc, _, _ = accuracy(logits, tgt_output, data_loader.PAD_IDX)
            if (idx + 1) % config.train_info_per_batch == 0:
                msg = f"Epoch: {epoch}, Batch[{idx + 1}/{len(train_iter)}], Train loss :{loss.item():.3f}, Train acc: {acc:.3f}"
                logging.info(msg)
        end_time = time.time()
        train_loss = losses / len(train_iter)
        msg = f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s"
        logging.info(msg)
        if (epoch + 1) % config.model_save_per_epoch == 0:
            acc = evaluate(config, test_iter, couplet_model, data_loader)
            if acc > max_test_acc:
                max_test_acc = acc
                state_dict = deepcopy(couplet_model.state_dict())
                torch.save(state_dict, config.model_save_path)
            logging.info(f"Accuracy on test {acc:.3f}, max_acc {max_test_acc:.3f}")



def evaluate(config, test_iter, model, data_loader):
    model.eval()
    correct, totals = 0, 0
    for idx, (src, tgt) in enumerate(test_iter):
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
    config = CoupleConfig()
    train_model(config)
