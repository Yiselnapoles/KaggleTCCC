# Author: Juan Antonio Cabeza Sousa
# Kaggle: Toxic Comment Classification Challenge
# Phase: Basic Analysis
#
#
# 2018-01-23
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('/home/juan/Documents/DataScience/KaggleTCCC/DataToUse/train.csv')
test = pd.read_csv('/home/juan/Documents/DataScience/KaggleTCCC/DataToUse/test.csv')
sub = pd.read_csv('/home/juan/Documents/DataScience/KaggleTCCC/DataToUse/sample_submission.csv')

# Dimmensions of the datasets
train.shape
test.shape
sub.shape

# train first 5 rows
train.head()
# Test first 5 rows
test.head()
# Example of upload file
sub.head()

# Train dataset basic analysis
#
#
fig, ax = plt.subplots()
# def basic_analysis_train():
res = {}
# Missing values
na_count = train.isnull().values.any()
# Split comment_text
train['comment_text'] = train['comment_text'].str.replace('\n', '')
train['comment_text_split']  = train['comment_text'].apply(lambda x: x.split())
# Maximum number of words in a column
max_train = train.comment_text_split.str.len().max()
# Minimum number of words in a column
min_train = train.comment_text_split.str.len().min()
# Check column has 1 or 0 values and get number of ones
n_toxic_values = train.toxic.unique()
n_toxic = sum(train['toxic'] == 1)

n_severe_toxic_values = train.severe_toxic.unique()
n_severe_toxic = sum(train['severe_toxic']==1)

n_obscene_values = train.obscene.unique()
n_obscene = sum(train['obscene']==1)

n_threat_values = train.threat.unique()
n_threat = sum(train['threat']==1)

n_insult_values = train.insult.unique()
n_insult = sum(train['insult']==1)

n_identity_hate_values = train.identity_hate.unique()
n_identity_hate = sum(train['identity_hate']==1)

res['na_count'] = na_count
res['sentence_max_length'] = max_train
res['sentence_min_length'] = max_train

# Count tags
train.iloc[:,[2,3,4,5,6,7]].sum()
# Check multitag
train.iloc[:,[2,3,4,5,6,7]].sum(axis=1)
train[train.iloc[:,[2,3,4,5,6,7]].sum(axis=1) > 1]
