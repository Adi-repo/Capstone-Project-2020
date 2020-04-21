import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, Input, load_model
from keras.layers import Dense, Dropout
from keras.layers import Embedding, Activation, Flatten
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, BatchNormalization
from keras.utils import to_categorical
from keras import optimizers
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import time
from IPython.display import Image
from IPython.core.display import HTML
from keras.callbacks import ModelCheckpoint


def window(a, w = 512, o = 256, copy = False): #window sliding function
    #default for training, for testing data we will split each signal in four of 1024 and apply
    #a window size of 512 with a stride (o) of 256
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::o]
    if copy:
        return view.copy()
    else:
        return view


def folder_to_df(letter): #import the .txt files
    full_path ="data/tuh_eeg_dataset/Evaluation/"+ letter + "/*.*"
    files = glob.glob(full_path)
    df_list = []
    for file in files:
        df_list.append(pd.read_csv(file, header = None))
    big_df = pd.concat(df_list, ignore_index=True, axis= 1)
    return big_df.T



def norm(X): # zero mean and unit variance normalization
    X = X - np.mean(X)
    X = X / np.std(X)
    return X


def load_data_as_df():
    A = norm(folder_to_df('Abnormal'))
    B = norm(folder_to_df('Normal'))
    
    test_abnormal = A
    test_normal = B

    return test_abnormal, test_normal


test_abnormal, test_normal = load_data_as_df()

best_model = load_model('best_model.0.908.h5')

# %%
"""
Additional necessary funtions
"""

# %%
def split_vote(df):
    res = list()
    for i in range(len(df)):
        res += [window(df.iloc[i].values,w= 512, o = 256)]
    return np.asarray(res)

def count_votes(my_list): 
    freq = {} 
    for i in my_list: 
        if (i in freq): 
            freq[i] += 1
        else: 
            freq[i] = 1
    return freq

def reshape_signal(signal):
    signal = np.expand_dims(signal, axis=1)
    signal = np.expand_dims(signal, axis=0)
    return np.asarray(signal)

def evaluate_subsignals(subsignals,model):
    vote_list = np.array([])
    for i in range(len(subsignals)):
        mini_signal = reshape_signal(subsignals[i])
        ynew = model.predict_classes(mini_signal)
        vote_list = np.append(vote_list, ynew)
    decision = count_votes(vote_list)
    return decision_to_str(decision), vote_list

def decision_to_str(dec):
    res = list()
    for key,val in dec.items():
        if key == 0:
            res += ['normal: ' + str(val) + ' votes' + '\n']
        if key == 1:
            res += ['abnormal: ' + str(val) + ' votes' + '\n']
    return res

# %%
big_signal = split_vote(test_normal)
ctr = 0
ptr = 0
print(big_signal[0])
for i in range(len(list(big_signal))):
    subsignals = big_signal[i]
    decision, vote_list = evaluate_subsignals(subsignals,best_model)
    print(vote_list)
    vote = list(vote_list)
    n = len(vote)
    c = vote.count(0)
    p = n/100
    print("-------------------------------------------------------------------")
    print()
    res = c/p
    if(int(res)>50):
        ctr = ctr + 1
    ptr = ptr + 1
acc_abnormal = (ctr/ptr) * 100
#print('Accuracy: {}%'.format(round(acc_abnormal,2)))
big_signal = split_vote(test_abnormal)
ctr = 0
ptr = 0
print(big_signal[0])
for i in range(len(list(big_signal))):
    subsignals = big_signal[i]
    decision, vote_list = evaluate_subsignals(subsignals,best_model)
    print(vote_list)
    vote = list(vote_list)
    n = len(vote)
    c = vote.count(0)
    p = n/100
    print("-------------------------------------------------------------------")
    print()
    res = (n-c)/p
    if(int(res)>50):
        ctr = ctr + 1
    ptr = ptr + 1
acc_normal = (ctr/ptr) * 100
accuracy = (acc_abnormal + acc_normal)/2
print('Accuracy: {}%'.format(round(accuracy,2)))