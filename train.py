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

def folder_to_df(letter): #import the .txt files
    full_path ="data/tuh_eeg_dataset/Train/"+ letter + "/*.*"
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

def enrich_train(df): #enrich data by splicing the 4097-long signals 
    #into 512 long ones with a stride of 64
    labels = df.iloc[:,-1]
    data = df.iloc[:, :-1]
    res = list()
    for i in range(len(data)):
        res += [window(data.iloc[i].values)]
    return res

def reshape_x(arr): #shape the input data into the correct form (x1,x2,1)
    nrows = arr.shape[0]
    ncols = arr.shape[1]
    return arr.reshape(nrows, ncols, 1)

def load_data_as_df():
    A = norm(folder_to_df('Abnormal'))
    B = norm(folder_to_df('Normal'))
    
    abnormal = A
    normal = B

    return normal, abnormal

normal, abnormal = load_data_as_df()

normal_train = normal
abnormal_train = abnormal

def format_enrich_train(normal, abnormal):
    
    normal_train_enr = np.asarray(enrich_train(normal)).reshape(-1, np.asarray(enrich_train(normal)).shape[-1])
    abnormal_train_enr = np.asarray(enrich_train(abnormal)).reshape(-1, np.asarray(enrich_train(normal)).shape[-1])

    #change into a dataframe to add labels easily
    normal_train_enr_df = pd.DataFrame(normal_train_enr)
    abnormal_train_enr_df = pd.DataFrame(abnormal_train_enr)
    
    normal_train_enr_df['labels'] = 0 
    abnormal_train_enr_df['labels'] = 1

    #concat all
    data_labels = pd.concat([normal_train_enr_df,abnormal_train_enr_df], ignore_index = True)
    

    #separates data and labels into numpy arrays for keras
    data = data_labels.drop('labels', axis = 1).values
    labels = data_labels.labels.values
    
    #labels = np.expand_dims(labels, axis=1)
    
    return data, labels

def create_model():           
    model = Sequential()
    
    #Conv - 1
    model.add(Conv1D(32, 5,strides =  3, input_shape=(512,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #Conv - 2
    model.add(Conv1D(24, 5,strides =  3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #Conv - 3
    model.add(Conv1D(16, 3,strides =  2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #Conv - 4
    model.add(Conv1D(8, 3,strides =  2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #FC -1
    model.add(Flatten())
    model.add(Dense(20))
    model.add(Activation('relu'))
    #Dropout
    model.add(Dropout(0.5))
    #FC -2
    model.add(Dense(2,activation = 'softmax'))

    adam = optimizers.Adam(lr=0.00002, beta_1=0.9, beta_2=0.999, epsilon=0.00000001, decay=0.0, amsgrad=False)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model

def train_evaluate_model(model, xtrain, ytrain, xval, yval, fold):
    model_name = 'P-1D-CNN'
    checkpointer = ModelCheckpoint(filepath='checkpoints/''fold'+ str(fold)+'.'+model_name + '.{epoch:03d}-{accuracy:.3f}.h5',verbose=0,monitor ='accuracy', save_best_only=True)
    history = model.fit(xtrain, ytrain, batch_size=32, callbacks = [checkpointer],epochs=200, verbose = 1)
    print(history)
    score = model.evaluate(xval, yval, batch_size=32)
    print('\n')
    print(score)
    return score, history

n_folds = 10
X, y = format_enrich_train(normal, abnormal)
skf = StratifiedKFold(n_splits=10, shuffle=True)

#10 fold cross validation loop
for i, (train, test) in enumerate(skf.split(X,y)):
    print("Running Fold", i+1, "/", n_folds)
    start_time = time.time()
    X = reshape_x(X)
    xtrain, xval = X[train], X[test]
    ytrain, yval = y[train], y[test]
    ytrain = to_categorical(ytrain, num_classes=2, dtype='float32')
    yval = to_categorical(yval, num_classes=2, dtype='float32')


    model = None # Clearing the NN.
    model = create_model()
    score, history = train_evaluate_model(model, xtrain, ytrain, xval, yval, i+1)
    print("Ran ", i+1, "/", n_folds, "Fold in %s seconds ---" % (time.time() - start_time))