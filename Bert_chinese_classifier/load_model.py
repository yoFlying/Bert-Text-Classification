
from test1111 import test_dataset
import torch
import os
from torch.utils.data import Dataset, DataLoader


save_path = './bert_checkpoint'
# 加载模型
model = BertClassifier()
model.load_state_dict(torch.load(os.path.join(save_path, 'best.pt')))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()



