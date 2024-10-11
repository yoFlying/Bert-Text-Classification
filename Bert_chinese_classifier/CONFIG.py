from transformers import BertTokenizer
import torch
# 加载分词器
# 下载的预训练文件路径
BERT_PATH = r"F:\modelsDownloads\bert-base-chinese"
save_path = './bert_checkpoint'
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
epoch = 5
batch_size = 64
lr = 1e-5
random_seed = 1999
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")