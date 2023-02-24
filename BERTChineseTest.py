import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd

# 資料前處理
# pandas 讀取需要的欄位
usecols = ['input text', '處理的字串', 'nlu 解析結果']
df_train = pd.read_csv("E:/機器學習/專題生資料/sample.csv", usecols=usecols)

# 去除空值
delete_list = []
for i in range(len(df_train)):
    if df_train['input text'].at[i] == " ":
        # print("null discover!")
        delete_list.append(i)
df_train = df_train.drop(df_train.index[delete_list])
# print(df_train)
# print(df_train.index)

# 檢查太長的輸入
for i in df_train.index:
    if len(str(df_train['input text'].at[i])) > 512:
        print(i + ' is too long!')

# 提取 intension
for i in df_train.index:
    temp = df_train['nlu 解析結果'].at[i]
    df_train['nlu 解析結果'].at[i] = temp[15:temp.index('"',17)]
    # print(df_train['nlu 解析結果'].at[i])

# 去除不必要的欄位並重新命名兩標題的欄位名
df_train = df_train.reset_index()
df_train = df_train.loc[:, ['input text', '處理的字串', 'nlu 解析結果']]
df_train.columns = ['text_a', 'text_b', 'label']

# idempotence, 將處理結果另存成 tsv 供 PyTorch 使用
df_train.to_csv("train.tsv", sep="\t", index=False)

# test 前處理
df_test = pd.read_csv("E:/機器學習/專題生資料/test.csv", usecols=usecols)
delete_list = []
for i in range(len(df_test)):
    if df_test['input text'].at[i] == " ":
        # print("null discover!")
        delete_list.append(i)
    elif df_test['nlu 解析結果'].at[i] == "[]":
        delete_list.append(i)
df_test = df_test.drop(df_test.index[delete_list])
for i in df_test.index:
    temp = df_test['nlu 解析結果'].at[i]
    df_test['nlu 解析結果'].at[i] = temp[15:temp.index('"',16)]
df_test = df_test.loc[:, ['input text', '處理的字串', 'nlu 解析結果']]
df_test.columns = ['text_a', 'text_b', 'label']
df_test.to_csv("test.tsv", sep="\t", index=False)


# 模型
# 指定使用模型
PRETRAINED_MODEL_NAME = "bert-base-chinese"

# 取得此預訓練模型所使用的 tokenizer
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

class BertDataset(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, mode, tokenizer):
        assert mode in ["train", "test"]  # 一般訓練你會需要 dev set
        self.mode = mode
        # 大數據你會需要用 iterator=True
        self.df = pd.read_csv(mode + ".tsv", sep="\t").fillna("")
        self.len = len(self.df)
        self.label_map = {'play_song': 0, 'play_album': 1, 'play_artist': 2, 'play_playlist': 3, 'play_artist_song': 4, 'play_artist_trending': 5, 'play_theme': 6, 'play_category': 7, 'play_audiobook': 8, 'play_general_music':9, 'play_language': 10, 'play_new_song': 11, 'play_audiobook_category': 12, 'play_chart': 13, 'play_trending':14, 'play_language_new_song': 15, 'play_artist_album': 16, 'play_language_chart': 17, 'play_general_audiobook': 18, 'play_category_chart': 19, 'play_collected_songs': 20, 'play_language_trending': 21, 'cmd_player_next_song': 22, 'play_artist_new_song': 23, 'play_category_trending': 24, 'cmd_query_song_info': 25, 'play_language_category': 26, 'continue_play_audiobook': 27, 'play_daily_discovery': 28, 'cmd_player_resume': 29, 'play_user_playlist': 30, 'cmd_player_previous_song': 31, 'play_general_user_playlist': 32, 'play_category_new_song': 33, 'cmd_player_stop': 34, 'cmd_player_pause': 35, 'cmd_add_to_song_collection': 36, 'play_new_user_playlist': 37}
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer

    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        if self.mode == "test":
            text_a, text_b = self.df.iloc[idx, :2].values
            label_tensor = None
        else:
            text_a, text_b, label = self.df.iloc[idx, :].values
            # 將 label 文字也轉換成索引方便轉換成 tensor
            label_id = self.label_map[label]
            label_tensor = torch.tensor(label_id)

    # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
        word_pieces = ["[CLS]"]
        tokens_a = self.tokenizer.tokenize(text_a)
        word_pieces += tokens_a + ["[SEP]"]
        len_a = len(word_pieces)
        
        # 第二個句子的 BERT tokens
        tokens_b = self.tokenizer.tokenize(text_b)
        word_pieces += tokens_b + ["[SEP]"]
        len_b = len(word_pieces) - len_a
        
        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a + [1] * len_b, 
                                        dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len

# 初始化一個專門讀取訓練樣本的 Dataset，使用中文 BERT 斷詞
trainset = BertDataset("train", tokenizer=tokenizer)

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    
    # 測試集有 labels
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None
    
    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, 
                                    batch_first=True)
    
    # attention masks，將 tokens_tensors 裡頭不為 zero padding 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids


# 初始化一個每次回傳 64 個訓練樣本的 DataLoader
# 利用 `collate_fn` 將 list of samples 合併成一個 mini-batch 是關鍵
BATCH_SIZE = 64
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch)

data = next(iter(trainloader))

tokens_tensors, segments_tensors, \
    masks_tensors, label_ids = data

# print(f"""
# tokens_tensors.shape   = {tokens_tensors.shape} 
# {tokens_tensors}
# ------------------------
# segments_tensors.shape = {segments_tensors.shape}
# {segments_tensors}
# ------------------------
# masks_tensors.shape    = {masks_tensors.shape}
# {masks_tensors}
# ------------------------
# label_ids.shape        = {label_ids.shape}
# {label_ids}
# """)

# 載入一個可以做中文多分類任務的模型，n_class = 3
from transformers import BertForSequenceClassification

PRETRAINED_MODEL_NAME = "bert-base-chinese"
NUM_LABELS = 38

model = BertForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)

# # high-level 顯示此模型裡的 modules
# print("""
# name            module
# ----------------------""")
# for name, module in model.named_children():
#     if name == "bert":
#         for n, _ in module.named_children():
#             print(f"{name}:{n}")
#     else:
#         print("{:15} {}".format(name, module))

def get_predictions(model, dataloader, compute_acc=False):
    predictions = None
    correct = 0
    total = 0
      
    with torch.no_grad():
        # 遍巡整個資料集
        for data in dataloader:
            # 將所有 tensors 移到 GPU 上
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            
            
            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)
            
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            
            # 用來計算訓練集的分類準確率
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                
            # 將當前 batch 記錄下來
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
    
    if compute_acc:
        acc = correct / total
        return predictions, acc
    return predictions
    
# 讓模型跑在 GPU 上並取得訓練集的分類準確率
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = model.to(device)
_, acc = get_predictions(model, trainloader, compute_acc=True)
print("classification acc:", acc)

def get_learnable_params(module):
    return [p for p in module.parameters() if p.requires_grad]
     
model_params = get_learnable_params(model)
clf_params = get_learnable_params(model.classifier)

print(f"""
整個分類模型的參數量：{sum(p.numel() for p in model_params)}
線性分類器的參數量：{sum(p.numel() for p in clf_params)}
""")
