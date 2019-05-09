#derived from: https://www.youtube.com/redirect?event=video_description&redir_token=a9jQkyqcTqBJNqzkrrA9zTRmgD58MTU0MzE3NTEyNkAxNTQzMDg4NzI2&q=https%3A%2F%2Fgithub.com%2FTannerGilbert%2FKeras-Tutorials%2Ftree%2Fmaster%2F4.%2520LSTM%2520Text%2520Generation&v=QtQt1CUEE3w

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import random
import sys
import io

tweet_keywords = [[0,1,2],[0,3,2],[5,4,1],[2,3,4],[0,0,1],[2,4,3]]
melding = [[0,1],[1,0],[1,0],[0,1],[1,1],[1,0]]

maxlen = 2
step = 1
sequences = []
next_melding = []
for i in range(0,len(tweet_keywords)-maxlen,step):
    sequences.append(tweet_keywords[i:i+maxlen])
    next_melding.append(melding[i+maxlen])
print('nb sequences:', len(sequences))

print(sequences[:3])
print(next_melding[:3])

x = np.array(sequences)
y = np.array(next_melding)

model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(tweet_keywords[0]))))
model.add(Dense(len(melding[0])))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.fit(x, y, batch_size=128, epochs=50)


def predict_melding():
    for i in range(0,len(sequences)-maxlen):
        x_pred = np.array(sequences[i:i+maxlen])
        pred = model.predict(x_pred)
        print()
        for j in range(len(pred)):
            print("time = " + str(i) + " predictions for time = "+str(i+j+1)+ ":")
            for k in range(len(pred[0])):
                print("indicent " + str(k) + ": " + str(round(pred[0][k]* 100,2)) + "%")

predict_melding()
