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
from transformers import DistilBertTokenizerFast, DistilBertModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.cuda.amp import autocast, GradScaler


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

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

#tokenize and enocde text in training set
tokens_train = tokenizer.batch_encode_plus(train_text.tolist(), max_length=256, padding=True, truncation=True)

#tokenize and en dccode text in validation set
tokens_val = tokenizer.batch_encode_plus(val_text.tolist(), max_length=256, padding=True, truncation=True)

#tokenize and encode text in test set
tokens_test = tokenizer.batch_encode_plus(test_text.tolist(), max_length=256, padding=True, truncation=True)

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
train_dataloader = DataLoader(train_data, sampler=train_sample, batch_size=10, num_workers=4)

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

    def forward(self, sent_id, mask):
        seq_output = self.bert(sent_id, attention_mask=mask)[0] 
        cls_hs = seq_output[:, 0, :] 
        x = self.dropout(cls_hs)
        x = self.fc(x)
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

positive_class_weight = torch.tensor([class_weights[1]], dtype=torch.float)
positive_class_weight = positive_class_weight.to(device)

# Initialize the loss function
cross_entropy = nn.BCEWithLogitsLoss(pos_weight=positive_class_weight)
epochs = 1

#training func
def train():
    
    model.train()
    total_loss, total_accuracy = 0, 0
    total_preds=[]
  
    for step,batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        model.zero_grad()        
        preds = model(sent_id, mask)
        labels = labels.unsqueeze(1).float()
        loss = cross_entropy(preds, labels)
        total_loss = total_loss + loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        preds = preds.detach().cpu().numpy()

        if device == torch.device("cuda"):
            torch.cuda.empty_cache()

    total_preds.append(preds)
    avg_loss = total_loss / len(train_dataloader)
    total_preds  = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

#eval func
def evaluate():
    
    print("\nEvaluating...")
    model.eval()
    total_loss, total_accuracy = 0, 0
    total_preds = []

    for step,batch in enumerate(val_dataloader):
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch
        with torch.no_grad():
            preds = model(sent_id, mask)
            labels = labels.unsqueeze(1).float()
            loss = cross_entropy(preds,labels)
            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)
    
    if device == torch.device("cuda"):
            torch.cuda.empty_cache()

    avg_loss = total_loss / len(val_dataloader) 
    total_preds  = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

#training the model
best_valid_loss = float('inf')
# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]

for epoch in range(epochs):
     
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    
    #train model
    train_loss, _ = train()
    #evaluate model
    valid_loss, _ = evaluate()
    
    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    
    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

# get predictions for test data
with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()

test_ids = test_df['ID']
# model's performance
preds = np.argmax(preds, axis = 1)
# Create a DataFrame for submission
submission_df = pd.DataFrame({
    'ID': test_ids,
    'label': preds
})

# Export the DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)
