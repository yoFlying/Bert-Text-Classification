import pandas as pd

# 本地数据地址
train_data_path = r"F:\OWNDATASETS\THUCNews\train.txt"
dev_data_path = r"F:\OWNDATASETS\THUCNews\dev.txt"
test_data_path = r"F:\OWNDATASETS\THUCNews\test.txt"
label_path = r"F:\OWNDATASETS\THUCNews\class.txt"

# 读取数据
train_df = pd.read_csv(train_data_path, sep='\t', header=None)
dev_df = pd.read_csv(dev_data_path, sep='\t', header=None)
test_df = pd.read_csv(test_data_path, sep='\t', header=None)

# 更改列名
print("*******更改列名**********")
new_columns = ['text', 'label']
train_df = train_df.rename(columns=dict(zip(train_df.columns, new_columns)))
dev_df = dev_df.rename(columns=dict(zip(dev_df.columns, new_columns)))
test_df = test_df.rename(columns=dict(zip(test_df.columns, new_columns)))

# 读取标签
real_labels = []
with open("F:\OWNDATASETS\THUCNews\class.txt", 'r') as f:
    for row in f.readlines():
        real_labels.append(row.strip())
print(real_labels)

# 下载的预训练文件路径
BERT_PATH = r"F:\modelsDownloads\bert-base-chinese"

from transformers import BertTokenizer
# 加载分词器
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

example_text = '我爱北京天安门。'
bert_input = tokenizer(example_text,padding='max_length',
                       max_length = 10,
                       truncation=True,
                       return_tensors="pt") # pt表示返回tensor
print(bert_input)
# {'input_ids': tensor([[ 101, 2769, 4263, 1266,  776, 1921, 2128, 7305,  511,  102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
example_text = tokenizer.decode(bert_input.input_ids[0])
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, df):
        # tokenizer分词后可以被自动汇聚
        self.texts = [tokenizer(text,
                                padding='max_length',  # 填充到最大长度
                                max_length = 35, 	# 经过数据分析，最大长度为35
                                truncation=True,
                                return_tensors="pt")
                      for text in df['text']]
        # Dataset会自动返回Tensor
        self.labels = [label for label in df['label']]

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

# 因为要进行分词，此段运行较久，约40s
train_dataset = MyDataset(train_df)
dev_dataset = MyDataset(dev_df)
test_dataset = MyDataset(test_df)

from torch import nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, 10)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer
# print("*************************开始训练******************")
import torch
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import random
import os
#
# # 训练超参数
# epoch = 5
# batch_size = 64
# lr = 1e-5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# random_seed = 1999
save_path = './bert_checkpoint'
#
#
# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#
#
# setup_seed(random_seed)
#
#
def save_model(save_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, save_name))
#

# 定义模型
model = BertClassifier()
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)
model = model.to(device)
criterion = criterion.to(device)

# 构建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

# 训练
best_dev_acc = 0
for epoch_num in range(epoch):
    total_acc_train = 0
    total_loss_train = 0
    for inputs, labels in tqdm(train_loader):
        input_ids = inputs['input_ids'].squeeze(1).to(device)  # torch.Size([32, 35])
        masks = inputs['attention_mask'].to(device)  # torch.Size([32, 1, 35])
        labels = labels.to(device)
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
            input_ids = inputs['input_ids'].squeeze(1).to(device)  # torch.Size([32, 35])
            masks = inputs['attention_mask'].to(device)  # torch.Size([32, 1, 35])
            labels = labels.to(device)
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


# 加载模型
model = BertClassifier()
model.load_state_dict(torch.load(os.path.join(save_path, 'best.pt')))
model = model.to(device)
# model.eval()


# def evaluate(model, dataset):
#     model.eval()
#     test_loader = DataLoader(dataset, batch_size=128)
#     total_acc_test = 0
#     with torch.no_grad():
#         for test_input, test_label in test_loader:
#             input_id = test_input['input_ids'].squeeze(1).to(device)
#             mask = test_input['attention_mask'].to(device)
#             test_label = test_label.to(device)
#             output = model(input_id, mask)
#             acc = (output.argmax(dim=1) == test_label).sum().item()
#             total_acc_test += acc
#     print(f'Test Accuracy: {total_acc_test / len(dataset): .3f}')
#
#
# evaluate(model, test_dataset)
#
# print("评估模型")
# evaluate(model, test_dataset)
while True:
    text = input('新闻标题：')
    bert_input = tokenizer(text, padding='max_length',
                            max_length = 35,
                            truncation=True,
                            return_tensors="pt")
    input_ids = bert_input['input_ids'].to(device)
    masks = bert_input['attention_mask'].unsqueeze(1).to(device)
    output = model(input_ids, masks)
    pred = output.argmax(dim=1)
    print(real_labels[pred])









