from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas as pd, numpy as np, string
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast

# specify GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load datasets and create dataframe
train_df = pd.read_csv('cse-472-project-ii-ai-generated-text-detection/train.csv')
test_df = pd.read_csv('cse-472-project-ii-ai-generated-text-detection/test.csv')


#concatenating title, abstract, and introduction columns into one column with all text
train_df['text'] = 'Title: ' + train_df['title'] + ' Abstract: ' + train_df['abstract'] + ' Introduction: ' + train_df['introduction']
test_df['text'] = 'Title: ' + test_df['title'] + ' Abstract: ' + test_df['abstract'] + ' Introduction: ' + test_df['introduction']

#split training data into train and validation sets
train_x, val_x, train_y, val_y = train_test_split(train_df['text'], train_df['label'], test_size=0.2, random_state=42)
test_x = test_df['text']

#import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')

#load BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

#data prep

#create datasets and dataloaders

#model architecture

#training

#evaluation


