from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas as pd, numpy as np, string
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import transformers
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#specify GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load datasets and create dataframe
train_df = pd.read_csv('cse-472-project-ii-ai-generated-text-detection/train.csv')
test_df = pd.read_csv('cse-472-project-ii-ai-generated-text-detection/test.csv')


#concatenating title, abstract, and introduction columns into one column with all text
train_df['text'] = 'Title: ' + train_df['title'] + ' Abstract: ' + train_df['abstract'] + ' Introduction: ' + train_df['introduction']
test_df['text'] = 'Title: ' + test_df['title'] + ' Abstract: ' + test_df['abstract'] + ' Introduction: ' + test_df['introduction']

#split training data into train and validation sets
train_text, val_text, train_label, val_label = train_test_split(train_df['text'], train_df['label'], test_size=0.2, random_state=42)
test_text = test_df['text']

#import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')

#load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

#tokenize and enocde text in training set
tokens_train = tokenizer.batch_encode_plus(train_text.tolist(), max_length=512, padding=True, truncation=False)

#tokenize and en dccode text in validation set
tokens_val = tokenizer.batch_encode_plus(val_text.tolist(), max_length=512, padding=True, truncation=False)

#tokenize and encode text in test set
tokens_test = tokenizer.batch_encode_plus(test_text.tolist(), max_length=512, padding=True, truncation=False)

#convert lists to tensors
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_label.tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_label.tolist())

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])

#wrap training tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

#sample data during training
train_sample = RandomSampler(train_data)

#dataloader for training dataset
train_dataloader = DataLoader(train_data, sampler=train_sample, batch_size=16)

#wrap validation tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

#sample validation data
val_sample = RandomSampler(val_data)

#dataloader for validation dataset
val_dataloader = DataLoader(val_data, sampler=val_sample, batch_size=16)

#set params of BERT model to not require gradients so only params of added layers trained
for param in bert.parameters():
    param.requires_grad = False

class BERTArchitecture(nn.Module):

    def __init__(self, bert):
        super(BERTArchitecture, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.dropout(cls_hs)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x 

#pass the pre-trained BERT to BERTArchitecture()
model = BERTArchitecture(bert)

#model to device indicated in beginning of code
model = model.to(device)

#AdamW optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

#compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_label), y=train_label)

print("Class weights: ", class_weights)

#converting class weights to tensors


