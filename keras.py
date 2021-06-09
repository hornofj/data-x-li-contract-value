import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import math
import statistics as st
import datetime
from statistics import mean

#Load the file containing variables [X_train, y_train, X_test, y_test]
import pickle
with open(r"../data-x-li-data/df_merged_train_test_05p.pickle", "rb") as input_file:
    X_train, y_train, X_test, y_test = pickle.load(input_file)

# define base model
model = Sequential()
model.add(Dense(11, input_dim=11,  activation='relu')) #input_dim - how many columns is in X_train
model.add(Dense(5,  activation='relu'))
model.add(Dense(2,  activation='relu'))
model.add(Dense(1, activation='linear')) #last layer shows 
model.compile(loss=tf.keras.losses.MeanAbsolutePercentageError(), optimizer='adam')

model.fit(X_train.values, y_train.values, epochs = 10, batch_size= 50)