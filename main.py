from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas as pd, xgboost, numpy, textblob, string

#load datasets and create dataframe
train_df = pd.read_csv('cse-472-project-ii-ai-generated-text-detection/train.csv')
test_df = pd.read_csv('cse-472-project-ii-ai-generated-text-detection/test.csv')

#preprocessing 
train_x = train_df['ID']
train_y = train_df['label']