# Author: Juan Antonio Cabeza Sousa
# Kaggle: Toxic Comment Classification Challenge
# Phase: Basic Analysis
#
#
# 2018-01-28

# %% -- Load libraries
#
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import seaborn as sns
from collections import OrderedDict
from operator import itemgetter
# %% -- Load data
#
#

train = pd.read_csv('/home/juan/Documents/DataScience/KaggleTCCC/DataToUse/train.csv')
test = pd.read_csv('/home/juan/Documents/DataScience/KaggleTCCC/DataToUse/test.csv')
sub = pd.read_csv('/home/juan/Documents/DataScience/KaggleTCCC/DataToUse/sample_submission.csv')

# New Wikipedia Data

# Toxicity annotated comments
tac = pd.read_csv('/home/juan/Documents/DataScience/KaggleTCCC/DataToUse/WikipediaData/toxicity_annotated_comments.tsv',sep='\t')
# Toxicity annotations
tan = pd.read_csv('/home/juan/Documents/DataScience/KaggleTCCC/DataToUse/WikipediaData/toxicity_annotations.tsv',sep='\t')
# Toxicity worker demographic
twd = pd.read_csv('/home/juan/Documents/DataScience/KaggleTCCC/DataToUse/WikipediaData/toxicity_worker_demographics.tsv',sep='\t')


# %% -- Basic Exploratory Data Analysis
#
#
rows_train = train.shape[0]
rows_test  = test.shape[0]
allrows = rows_train + rows_test

print('TRAIN ROWS :', rows_train)
print('TEST ROWS :', rows_test)
print("Percentages: TRAIN = ",round(rows_train*100/allrows), "% ,TEST = ",round(rows_test*100/allrows),'%')

## Train Dataset

# Clean comments [ No tags ]
clean_comments = len(train[train.iloc[:,2:].sum(axis=1)==0])
clean_comments
train.iloc[:,2:].sum()

d_plot_tags = {}
for col in train.iloc[:,2:]:
    d_plot_tags[col] = train[col].sum()

d_plot_tags['clean_comment'] = clean_comments
sorted_plot_tags = sorted(d_plot_tags.items(), key=itemgetter(1))

plotdata_1 = pd.DataFrame.from_dict(sorted_plot_tags)
plotdata_1.columns = ['Tag','Count']

# Graph with tags
plt.figure(figsize=(8,4))

sns.barplot(plotdata_1.Tag, plotdata_1.Count)
plt.xticks(rotation=90)
plt.show()

# Graph with multi-tags
dmulti_tag=train.iloc[:,2:].sum(axis=1)
dmulti_tag_count=dmulti_tag.value_counts()

multitagpd = pd.DataFrame({'NTags': dmulti_tag_count.index,'Count': dmulti_tag_count.values})

plt.figure(figsize = (8,4))
ax = sns.barplot(multitagpd.NTags, multitagpd.Count)
plt.show()


# Correlation between Variables. No clean comments
toxic_comments = train[train.iloc[:,2:].sum(axis=1) >= 1]
toxcorr = toxic_comments.iloc[:,2:-1]
toxcorr.corr()
