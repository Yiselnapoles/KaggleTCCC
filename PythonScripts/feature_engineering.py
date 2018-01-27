# Author: Juan Antonio Cabeza Sousa
# Kaggle: Toxic Comment Classification Challenge
# Phase: Basic Analysis
#
#
# 2018-01-27
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('/home/juan/Documents/DataScience/KaggleTCCC/DataToUse/train.csv')
test = pd.read_csv('/home/juan/Documents/DataScience/KaggleTCCC/DataToUse/test.csv')
sub = pd.read_csv('/home/juan/Documents/DataScience/KaggleTCCC/DataToUse/sample_submission.csv')
