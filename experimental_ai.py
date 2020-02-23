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
from keras.layers import Conv1D, MaxPooling1D
from frequency import frequency_sepectrum
from frequency import define_whistle
from frequency import fft_sig
from frequency import quick_search
from keras.models import Model
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras.optimizers import Adagrad
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

def create_model():
    # create model
    '''model = Sequential()
    model.add(Dense(num, input_dim=num, kernel_initializer='zeros', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='zeros', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model'''

    model = Sequential()
    model.add(Conv1D(32, kernel_size=(5),activation='relu',input_shape=(66048,1)))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='normal'))
    model.add(Dense(num_classes, activation='softmax',kernel_initializer='normal'))
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
seed = 7
np.random.seed(seed)

sit_train_1=[fft_sig('C:\Work\speech_recognition\sit\sample_{}.wav'.format(i))[0] for i in range(1,6)]
down_train_1=[fft_sig('C:\Work\speech_recognition\down\sample_{}.wav'.format(i))[0] for i in range(1,6)]
X_train_1=sit_train_1+down_train_1
#print(len(X_train))
y_train=[0]*len(sit_train_1)+[1]*len(down_train_1)
sit_train_2=[fft_sig('C:\Work\speech_recognition\sit\sample_{}.wav'.format(i))[1] for i in range(1,6)]
down_train_2=[fft_sig('C:\Work\speech_recognition\down\sample_{}.wav'.format(i))[1] for i in range(1,6)]
X_train_2=sit_train_2+down_train_2
#print(len(X_train))



sit_test_1=[fft_sig('C:\Work\speech_recognition\sit\sample_{}.wav'.format(i))[0] for i in range(6,11)]
down_test_1=[fft_sig('C:\Work\speech_recognition\down\sample_{}.wav'.format(i))[0] for i in range(6,11)]
X_test_1=sit_test_1+down_test_1
y_test=[0]*len(sit_test_1)+[1]*len(down_test_1)

sit_test_2=[fft_sig('C:\Work\speech_recognition\sit\sample_{}.wav'.format(i))[1] for i in range(6,11)]
down_test_2=[fft_sig('C:\Work\speech_recognition\down\sample_{}.wav'.format(i))[1] for i in range(6,11)]
X_test_2=sit_test_2+down_test_2


X_train_1 = np.array(bad_division(X_train_1))
X_train_2 = np.array(bad_division(X_train_2))
X_test_1 = np.array(bad_division(X_test_1))
X_test_2 = np.array(bad_division(X_test_2))


num = X_train_1.shape[1]

X_train_1 = X_train_1.reshape(X_train_1.shape[0], num).astype('float32')
X_train_2 = X_train_2.reshape(X_train_2.shape[0], num).astype('float32')
X_test_1 = X_test_1.reshape(X_test_1.shape[0], num).astype('float32')
X_test_2 = X_test_2.reshape(X_test_2.shape[0], num).astype('float32')

X_train_1 = np.expand_dims(X_train_1, axis=2)
X_train_2 = np.expand_dims(X_train_2, axis=2)
X_test_1 = np.expand_dims(X_test_1, axis=2)
X_test_2 = np.expand_dims(X_test_2, axis=2)

y_train = np_utils.to_categorical(y_train,num_classes=2)
y_test = np_utils.to_categorical(y_test,num_classes=2)
#print(y_test[0])
num_classes = 2

model = create_model()

first_input = Input(shape=(66048, 1))
second_input = Input(shape=(66048, 1))
first_dense = create_model()(first_input)

second_dense = create_model()(second_input)

merge_one = concatenate([first_dense, second_dense])



model = Model(inputs=[first_input,second_input], outputs=merge_one)
ada_grad = Adagrad(lr=0.1, epsilon=1e-08, decay=0.0)
model.compile(optimizer=ada_grad, loss='binary_crossentropy',
               metrics=['accuracy'])
print(type(y_train))
print(len(y_train))
# Fit the model
model.fit([X_train_1,X_train_2],np.reshape([y_train, y_train], (10,4)), validation_data=([X_test_1, X_test_2],np.reshape([y_test, y_test],(10,4))), epochs=10, batch_size=1, verbose=2)
# Final evaluation of the model
scores = model.evaluate([X_test_1, X_test_2],np.reshape([y_test, y_test],(10,4)), verbose=0)
print("Error: %.2f%%" % (100-scores[1]*100))
'''model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# training
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test),
          callbacks=[history])

# evaluating and printing results
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])'''
