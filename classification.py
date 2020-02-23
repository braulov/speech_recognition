from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras import metrics
from keras.layers import Flatten
from keras.layers import Conv2D, MaxPooling2D,Conv1D, MaxPooling1D
from frequency_for_classification import frequency_sepectrum
from frequency_for_classification import fft_sig
from frequency_for_classification import quick_search
from keras.models import Model
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras.optimizers import Adagrad
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
def bad_division(a):
    fft_x=a[:][0]
    freq=a[:][1]
    #max_el_freq=freq[0][0]
    #max_el_fft_x=fft_x[0][0]
    for i in range(len(fft_x)):
        max_el=max(fft_x[i])
        fft_x[i]=fft_x[i]/max_el
    for i in range(len(freq)):
        max_el=max(freq[i])
        freq[i]=freq[i]/max_el
    
    return a

def mfcc_loop(n_first,n_last,grade):
    a=[]
    for i in range(n_first, n_last):
        (rate,sig) = wav.read("C:\Work\speech_recognition\{}\sample_{}.wav".format(grade,i))
        a.append(mfcc(sig,rate))
    return a

def create_model():
    global shape
    # create model
    '''model = Sequential()
    model.add(Dense(num, input_dim=num, kernel_initializer='zeros', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='zeros', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model'''

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5,5),activation='relu',padding='same'))
    model.add(Conv2D(32, kernel_size=(5,5),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='normal'))
    model.add(Dense(2, activation='softmax',kernel_initializer='normal'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
seed = 7
np.random.seed(seed)
sit_train=[fft_sig('C:\Work\speech_recognition\sit\sample_{}.wav'.format(i)) for i in range(1,6)]
down_train=[fft_sig('C:\Work\speech_recognition\down\sample_{}.wav'.format(i)) for i in range(1,6)]
#sit_train=[fft_sig('C:\Work\speech_recognition\sit\sample_{}.wav'.format(i)) for i in range(1,6)]
#down_train=[fft_sig('C:\Work\speech_recognition\down\sample_{}.wav'.format(i)) for i in range(1,6)]
X_train=sit_train+down_train
#print(len(X_train))
y_train=[0]*len(sit_train)+[1]*len(down_train)
print(np.array(sit_train).shape)
#sit_test=[fft_sig('C:\Work\speech_recognition\sit\sample_{}.wav'.format(i)) for i in range(6,11)]
#down_test=[fft_sig('C:\Work\speech_recognition\down\sample_{}.wav'.format(i)) for i in range(6,11)]
X_test=sit_test+down_test
y_test=[0]*len(sit_test)+[1]*len(down_test)

shape=len(sit_train[0][0])
X_train = np.array(X_train[:][:][:]).astype('float32')
X_test = np.array(X_test[:][:][:]).astype('float32')
X_train = np.array(bad_division(X_train))
X_test = np.array(bad_division(X_test))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

y_train = np_utils.to_categorical(y_train,num_classes=2)
y_test = np_utils.to_categorical(y_test,num_classes=2)
#print(y_test[0])
num_classes = 2

model = create_model()
# Fit the model
model.fit(X_train,y_train, validation_data=(X_test,y_test), epochs=10, batch_size=1, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test,y_test, verbose=0)
print("Error: %.2f%%" % (100-scores[1]*100))
