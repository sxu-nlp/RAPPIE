# -*- encoding:utf-8 -*-
import os
import numpy as np
import torch
import math
import random
import argparse
import torch.nn as nn
import pickle
from einops import rearrange
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report
from optimizers import BertAdam
import pandas as pd

MODEL_PATH = "the path of [chatglm-6b]"

# 定义分类器
class ChatglmClassifier(nn.Module):
    # 定义构造函数
    def __init__(self, args):  # 定义类的初始化函数，用户传入的参数
        super(ChatglmClassifier, self).__init__()  # 调用父类nn.module的初始化方法，初始化必要的变量和参数
        emb_size = 2048
        dropout = args.dropout
        lstm_hidden_size = 200  # lstm_hidden_size的规格
        self.multi_head_attention1 = torch.nn.MultiheadAttention(emb_size, 8, dropout=dropout)
        self.multi_head_attention2 = torch.nn.MultiheadAttention(emb_size, 8, dropout=dropout)
        self.sigmoid = nn.Sigmoid()
        self.batchsize=args.batch_size
        self.labels_num = 7
        self.linear_adapter_i = torch.nn.Linear(4096, 2048)
        self.linear_adapter_r = torch.nn.Linear(4096, 2048)
        self.linear_adapter_u = torch.nn.Linear(4096, 2048)
        self.linear_adapter_sum1_1 = nn.Linear(2, 1)
        self.linear_adapter_sum1_2 = nn.Linear(4, 2)
        self.linear_adapter_sum1_3 = nn.Linear(6, 3)
        self.linear_adapter_sum1_4 = nn.Linear(8, 4)
        self.linear_adapter_sum1_5 = nn.Linear(10, 5)
        self.linear_adapter_sum1_6 = nn.Linear(12, 6)
        self.linear_adapter_sum1_7 = nn.Linear(14, 7)
        self.linear_adapter_sum1_8 = nn.Linear(16, 8)
        self.linear_1 = nn.Linear(in_features=emb_size, out_features=emb_size)
        self.linear_2 = nn.Linear(in_features=emb_size, out_features=emb_size)
        self.W_gs = torch.nn.Linear(emb_size, lstm_hidden_size)
        self.W_gr = torch.nn.Linear(emb_size, lstm_hidden_size)
        self.W_sum1 = torch.nn.Linear(emb_size, lstm_hidden_size)
        self.W_sum2 = torch.nn.Linear(emb_size, lstm_hidden_size)
        self.output_layer_1 = nn.Linear(emb_size, lstm_hidden_size)
        self.output_layer_2 = nn.Linear(lstm_hidden_size, self.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)  # softmax维度
        self.criterion = nn.NLLLoss()  # 损失函数（NLLLoss 函数输入 input 之前，需要对 input 进行 log_softmax 处理）

    def forward(self, inputs, label, user_r, role):
        inputs = self.linear_adapter_i(inputs)
        user_r = self.linear_adapter_u(user_r)
        role = self.linear_adapter_r(role)
        inputs = rearrange(inputs, 'B S E->S B E')
        user_r = user_r.transpose(0, 1)
        role = role.transpose(0, 1)
        att_user_inputs = self.multi_head_attention1(user_r, inputs, inputs)[0]
        att_user_role = self.multi_head_attention2(user_r, role, role)[0]
        W_att_user_inputs = self.sigmoid(self.linear_1(att_user_inputs))
        W_att_user_role = self.sigmoid(self.linear_2(att_user_role))
        g_s = W_att_user_inputs * att_user_inputs
        g_r = W_att_user_role * att_user_role
        sum1 = torch.cat((g_s, g_r), dim=1)
        sum2 = torch.mul(g_s, g_r)
        inputs = torch.mean(inputs.transpose(0, 1), dim=1)
        g_s = torch.mean(g_s.transpose(0, 1), dim=1)
        g_r = torch.mean(g_r.transpose(0, 1), dim=1)
        sum1  = torch.mean(sum1 .transpose(0, 1), dim=1)
        sum2 = torch.mean(sum2.transpose(0, 1), dim=1)
        sum1 = sum1.transpose(0, 1)
        if sum2.shape[0]==1:
            sum1 = self.linear_adapter_sum1_1(sum1)
        elif sum2.shape[0]==2:
            sum1 = self.linear_adapter_sum1_2(sum1)
        elif sum2.shape[0]==3:
            sum1 = self.linear_adapter_sum1_3(sum1)
        elif sum2.shape[0]==4:
            sum1 = self.linear_adapter_sum1_4(sum1)
        elif sum2.shape[0]==5:
            sum1 = self.linear_adapter_sum1_5(sum1)
        elif sum2.shape[0]==6:
            sum1 = self.linear_adapter_sum1_6(sum1)
        elif sum2.shape[0]==7:
            sum1 = self.linear_adapter_sum1_7(sum1)
        elif sum2.shape[0]==8:
            sum1 = self.linear_adapter_sum1_8(sum1)
        else:
            print("chucuo")
        sum1 = sum1.transpose(0, 1)
        mpoa_input = self.W_gs(g_s) + self.W_gr(g_r) + self.W_sum1(sum1) + self.W_sum2(sum2)  # 全部/去关注/去转评

        logits = self.output_layer_2(mpoa_input)
        loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))  # 损失函数
        return loss, logits

# 定义主函数
def main():
    def set_seed(seed=7):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def read_dataset(path):
        dataset = []
        with open(path, mode="r", encoding="utf-8") as f:  # 打开文件
            for line_id, line in enumerate(f):
                line = line.strip('\n').split('\t')  # 按 Tab键分割文本
                dataset.append(line)
        print("dataset:", len(dataset))
        return dataset

    def batch_loader(data, data_type):
        def get_glm_emb(text, max_len):
            tokens = tokenizer([text], return_tensors="pt")["input_ids"].tolist()[0]
            if len(tokens) > max_len:
                tokens = tokens[:max_len - 2] + tokens[-2:]
            while len(tokens) < max_len:
                tokens.append(3)
            emb = Chatglm.transformer(**{"input_ids": torch.tensor(tokens).unsqueeze(0).to(device)},
                                      output_hidden_states=True).last_hidden_state
            return emb.cpu().detach().squeeze()
        # 没有则获取glm表示
        if not os.path.exists(args.dataset + "_y/" + data_type + "/batch_0.pkl"): # 全部关系
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            Chatglm = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).half().to(device)
            Chatglm = Chatglm.eval()
            dataset = []
            random.shuffle(data)
            batch_id = 0
            for line_id, line in enumerate(data):
                if len(dataset) == batch_size or line_id == len(data)-1:
                    batch_file = args.dataset + "_y/" + data_type + "/batch_" + str(batch_id) + '.pkl' # 全部关系
                    batch_inputs = torch.tensor([example[0].numpy() for example in dataset])
                    batch_label = torch.LongTensor([example[1] for example in dataset])
                    batch_user_r = torch.tensor([example[2].numpy() for example in dataset])
                    batch_role = torch.tensor([example[3].numpy() for example in dataset])
                    with open(batch_file, "wb") as f:
                        pickle.dump([batch_inputs, batch_label, batch_user_r, batch_role], f)
                    dataset=[]
                    batch_id += 1
                label = int(line[columns["label"]])  # 赋值label
                text = line[columns["text"]]  # 40左右
                user_id = int(line[columns["user_id"]])
                # 用户关系
                user_r1 = user_r1_emb[user_id]  # 关注
                user_r2 = user_r2_emb[user_id]  # 转发
                user_r3 = user_r3_emb[user_id]  # 评论
                user_r = torch.stack([user_r1, user_r2, user_r3])  # 全部融合/去传播角色
                role = role_emb
                inputs = get_glm_emb(text, 128)
                dataset.append((inputs, label, user_r, role))
                if (line_id == len(data)-1 and data_type == 'test') or (line_id == len(data)-1 and data_type == "evaluate" ):
                    batch_file = args.dataset + "_y/" + data_type + "/batch_" + str(batch_id) + '.pkl' # 全部关系
                    batch_inputs = torch.tensor([example[0].numpy() for example in dataset])
                    batch_label = torch.LongTensor([example[1] for example in dataset])
                    batch_user_r = torch.tensor([example[2].numpy() for example in dataset])
                    batch_role = torch.tensor([example[3].numpy() for example in dataset])
                    with open(batch_file, "wb") as f:
                        pickle.dump([batch_inputs, batch_label, batch_user_r, batch_role], f)
                    dataset = []
                    batch_id += 1
        for i in range(math.ceil(len(data)/batch_size)):
            with open(args.dataset + "_y/" + data_type + "/batch_" + str(i) + ".pkl", "rb") as f:# 全部关系
                inputs, label, user_r, role = pickle.load(f)
                inputs = inputs.type(torch.float32)
                user_r = user_r.type(torch.float32)
                role = role.type(torch.float32)
            yield inputs.to(device), label.to(device), user_r.to(device), role.to(device)

    def evaluate(args, is_test):
        if is_test:
            dataset = read_dataset(args.dataset + "/test_y.txt")[1:]
            data_type = "test"
        else:
            dataset = read_dataset(args.dataset + "/valid_y.txt")[1:]
            data_type = "evaluate"

        instances_num = len(dataset)
        if is_test:
            logger("The number of evaluation instances: ", instances_num)

        correct = 0
        model.eval()
        pred_all = []
        gold_all = []
        for i, (inputs_batch, label_batch, user_r_batch, batch_role)in enumerate(batch_loader(dataset, data_type)):  # 循环
            with torch.no_grad():  # 进行计算图的构建
                loss, logits = model(inputs_batch, label_batch, user_r_batch, batch_role)
            logits = nn.Softmax(dim=1)(logits)
            pred = torch.argmax(logits, dim=1)
            gold = label_batch
            pred_all.extend(pred.cpu().numpy().tolist())
            gold_all.extend(gold.cpu().numpy().tolist())
            correct += torch.sum(pred == gold).item()
        logger(classification_report(gold_all, pred_all, digits=3))

    def save_model(model, model_path):
        if hasattr(model, "module"):
            torch.save(model.module.state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)

    def load_model(model, model_path):
        if hasattr(model, "module"):
            model.module.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        else:
            model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        return model

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # 命令行参数解析包
    parser.add_argument("--output_model_path", default="../other/classifier_model_y.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--dataset", default="../data", type=str,
                        help="Path of the trainset.")
    parser.add_argument("--emo_vec_path", type=str, default="other/emo_vector.json")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--high_encoder", choices=["bi-lstm", "lstm", "none"], default="bi-lstm")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last", "attention", "multi-head"],
                        default="mean", help="Pooling type.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--weight_decay_rate", type=float, default=1e-2,
                        help="zhengzahua.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=15,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to logger prompt.")
    parser.add_argument("--seed", type=int, default=6,
                        help="Random seed.")
    args = parser.parse_args()

    root = '../logger/'  # 日志文件
    file_name = root + 'logger.txt'
    log_file = open(file_name, 'a', encoding='utf-8')

    def logger(*args):
        str_list = " ".join([str(arg) for arg in args])
        print(str_list)
        log_file.write(str_list + '\n')
        log_file.flush()
    logger(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    model = ChatglmClassifier(args)
    model = model.to(device)

    # Training phase.
    logger("Start training.")
    path1 = "dataset/weibo_glm/user_embedding_follow_r.pkl"
    path2 = "dataset/weibo_glm/user_embedding_retweet_r.pkl"
    path3 = "dataset/weibo_glm/user_embedding_comment_r.pkl"
    file1 = open(path1, "rb")
    file2 = open(path2, "rb")
    file3 = open(path3, "rb")
    role_file = open("dataset/weibo_glm/role_features.pkl", "rb")
    user_r1_emb = pickle.load(file1).cpu().detach()
    user_r2_emb = pickle.load(file2).cpu().detach()
    user_r3_emb = pickle.load(file3).cpu().detach()
    role_emb = torch.from_numpy(pickle.load(role_file)).cpu().detach()
    file1.close()
    file2.close()
    file3.close()
    role_file.close()
    dataset = read_dataset(args.dataset + "/train_y_en.txt")
    columns = dict(zip(dataset[0], range(len(dataset[0]))))
    instances_num = len(dataset)-1
    batch_size = args.batch_size

    logger("Batch size: ", batch_size)
    logger("The number of training instances:", instances_num)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    train_steps = int(instances_num * args.epochs_num / batch_size) + 1
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)
    total_loss = 0.
    for epoch in range(1, args.epochs_num + 1):
        model.train()
        data_type = "train"
        for i, (inputs_batch, label_batch, user_r_batch, batch_role) in enumerate(batch_loader(dataset[1:], data_type)):  # 循环
            model.zero_grad()
            loss, logits = model(inputs_batch, label_batch, user_r_batch, batch_role)
            # print("loss",loss)
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                logger("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1,total_loss / args.report_steps))
                total_loss = 0.

            loss.backward()
            optimizer.step()
        evaluate(args, False, epoch)
        save_model(model, args.output_model_path)
    # Evaluation phase.
    logger("Test set evaluation.")
    model = load_model(model, args.output_model_path)
    evaluate(args, True, 15)
    log_file.close()

if __name__ == "__main__":
    main()
