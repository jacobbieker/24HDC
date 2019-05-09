from __future__ import print_function

import keras
from keras.utils import Sequence
from sklearn.utils import shuffle
import re

from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import random
import sys
import io


class TweetSeqGenerator(Sequence):
  
    def __init__(self, data, labels, batch_size, normalize=False):
      self.data = data
      self.labels = labels
      if len(self.data) != len(self.labels):
        raise ValueError("Data and Labels have to be same size")
      self.batch_size = batch_size
      self.normalize = normalize
      
      # If normalize, then normalize each timeseries to sum to 1 on each keyword
      if self.normalize:
        for i in range(len(self.data)):
            # Normalizes the rows of the matrix to 1
            # Axis=0 should normalize the columns to 1
            self.data[i] = self.data[i]/np.max(self.data[i], axis=1)

    def __getitem__(self, index):
        """
        Go through each set of files and augment them as needed
        :param index:
        :return:
        """
        timeseries = self.data[index * self.batch_size:(index + 1) * self.batch_size]
        labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        
        print("timeseries:")
        print(timeseries)
        print(timeseries.shape)
        print("labels")
        print(labels)
        print(labels.shape)

        return timeseries, labels

    def __len__(self):
        """
        Returns the length of the list of paths, as the number of events is not known
        :return:
        """
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def on_epoch_end(self):
            self.data = shuffle(self.data, self.labels)
            

def generate_timeseries(category, length, limit=None):
  """
  Generates timeseries from the tweets, and gets the next event to occur in the future
  """
  return NotImplementedError


def clean_tweet(tweet):
  """
  Cleans a tweet to just English characters, removing URLs, etc.
  What is given is just the tweet itself, not the metadata
  """
  tweet = re.sub(r"http\S+", "", tweet) # removes URLs
  tweet = re.sub(r"[^a-zA-Z0-9]+", ' ', tweet) # Removes non-alphanumeric chars
  tweet = tweet.lower() # Lowercases it
  
  return tweet

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

print(len(x))
print(len(y))

model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(tweet_keywords[0]))))
model.add(Dense(len(melding[0])))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

train_gen = TweetSeqGenerator(data=x, labels=y, batch_size=1)
val_gen = TweetSeqGenerator(data=x, labels=y, batch_size=1)

model.fit_generator(generator=train_gen, validation_data=val_gen, epochs=50)


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
