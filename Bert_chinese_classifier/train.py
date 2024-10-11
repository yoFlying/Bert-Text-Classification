import os
from CONFIG import save_path
from bert_model import BertClassifier
import torch
import pandas as pd
from MyDataset import MyDataset
import numpy as np
import random
from torch import nn
import CONFIG
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
def save_model(save_name,model):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, save_name))
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
if __name__ == '__main__':
    # 本地数据地址
    print("--------------预处理数据----------------------")
    train_data_path = r"F:\OWNDATASETS\THUCNews\train.txt"
    dev_data_path = r"F:\OWNDATASETS\THUCNews\dev.txt"
    test_data_path = r"F:\OWNDATASETS\THUCNews\test.txt"
    label_path = r"F:\OWNDATASETS\THUCNews\class.txt"
    # 读取数据
    train_df = pd.read_csv(train_data_path, sep='\t', header=None)
    dev_df = pd.read_csv(dev_data_path, sep='\t', header=None)
    test_df = pd.read_csv(test_data_path, sep='\t', header=None)

    # 更改列名
    new_columns = ['text', 'label']
    train_df = train_df.rename(columns=dict(zip(train_df.columns, new_columns)))
    dev_df = dev_df.rename(columns=dict(zip(dev_df.columns, new_columns)))
    test_df = test_df.rename(columns=dict(zip(test_df.columns, new_columns)))

    # 读取标签
    real_labels = []
    with open("F:\OWNDATASETS\THUCNews\class.txt", 'r') as f:
        for row in f.readlines():
            real_labels.append(row.strip())
    # 因为要进行分词，此段运行较久，约40s
    train_dataset = MyDataset(train_df)
    dev_dataset = MyDataset(dev_df)
    test_dataset = MyDataset(test_df)
    print("--------------Dataset完成加载----------------------")

    # 定义模型
    model = BertClassifier()
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = Adam(model.parameters(), lr=CONFIG.lr)
    model = model.to(CONFIG.device)

    #构件数据加载器
    train_loader = DataLoader(dataset=dev_dataset,batch_size=CONFIG.batch_size,shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=CONFIG.batch_size)


    # 训练
    best_dev_acc = 0
    for epoch_num in range(CONFIG.epoch):
        total_acc_train = 0
        total_loss_train = 0
        for inputs, labels in tqdm(train_loader):
            input_ids = inputs['input_ids'].squeeze(1).to(CONFIG.device)
            masks = inputs['attention_mask'].to(CONFIG.device)
            labels = labels.to(CONFIG.device)
            output = model(input_ids, masks)

            batch_loss = criterion(output, labels)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            acc = (output.argmax(dim=1) == labels).sum().item()
            total_acc_train += acc
            total_loss_train += batch_loss.item()
            print(f' Loss: {batch_loss.item():.4f}, Accuracy: {acc / labels.size(0):.4f}')
        # ----------- 验证模型 -----------
        model.eval()
        total_acc_val = 0
        total_loss_val = 0
        # 不需要计算梯度
        with torch.no_grad():
            # 循环获取数据集，并用训练好的模型进行验证
            for inputs, labels in dev_loader:
                input_ids = inputs['input_ids'].squeeze(1).to(CONFIG.device)  # torch.Size([32, 35])
                masks = inputs['attention_mask'].to(CONFIG.device)  # torch.Size([32, 1, 35])
                labels = labels.to(CONFIG.device)
                output = model(input_ids, masks)

                batch_loss = criterion(output, labels)
                acc = (output.argmax(dim=1) == labels).sum().item()
                total_acc_val += acc
                total_loss_val += batch_loss.item()
            print(f'''Epochs: {epoch_num + 1}
              | Train Loss: {total_loss_train / len(train_dataset): .3f}
              | Train Accuracy: {total_acc_train / len(train_dataset): .3f}
              | Val Loss: {total_loss_val / len(dev_dataset): .3f}
              | Val Accuracy: {total_acc_val / len(dev_dataset): .3f}''')

            # 保存最优的模型
            if total_acc_val / len(dev_dataset) > best_dev_acc:
                best_dev_acc = total_acc_val / len(dev_dataset)
                save_model('best.pt')
        model.train()
# 保存最后的模型，以便继续训练
save_model('last.pt')