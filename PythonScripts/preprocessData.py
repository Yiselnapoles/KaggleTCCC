# Author: Juan Antonio Cabeza Sousa
# Kaggle: Toxic Comment Classification Challenge
# Phase: Basic Analysis
#
#
# 2018-01-23
import numpy as np
import pandas as pd

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
def basic_analysis_train():
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

    res['toxic'] = {'Values':n_toxic_values,'Count': n_toxic}
    res['severe_toxic'] = {'Values':n_severe_toxic_values,'Count': n_severe_toxic}
    res['obscene'] = {'Values':n_obscene_values,'Count': n_obscene}
    res['threat'] = {'Values':n_threat_values,'Count': n_threat}
    res['insult'] = {'Values':n_insult_values,'Count': n_insult}
    res['identity_hate'] = {'Values':n_identity_hate_values,'Count': n_identity_hate}

    return res

basic_analysis_train()
