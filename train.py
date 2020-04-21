import gensim, re
import numpy as np
from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm


import pandas as pd
import string

path_to_json = 'tech_soft_none.json'
df = pd.read_json(path_to_json,orient='split')


df['CleanData'] = df['text'].str.translate(str.maketrans('','',string.punctuation))

trainDF = pd.DataFrame()
trainDF['text'] = df['CleanData']
trainDF['label'] = df['label']

from keras.preprocessing.text import Tokenizer

# how many features should the tokenizer extract
features = 500
tokenizer = Tokenizer(num_words = features)
# fit the tokenizer on our text
tokenizer.fit_on_texts(trainDF['text'])

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)



import gensim, re
import numpy as np
from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding

text = [re.sub(r'([^\s\w]|_)+', '', sentence) for sentence in trainDF['text']]
text = [sentence.lower().split() for sentence in text]

word_model = gensim.models.Word2Vec(text, size=300, min_count=1, iter=10)

#Embedding Matrix
# save the vectors in a new matrix
embedding_matrix = np.zeros((len(word_model.wv.vocab) + 1, 300))
for i, vec in enumerate(word_model.wv.vectors):
  embedding_matrix[i] = vec
  
  
  
#Tokenising for LSTM

# how many features should the tokenizer extract
features = 500
tokenizer = Tokenizer(num_words = features)
# fit the tokenizer on our text
tokenizer.fit_on_texts(trainDF['text'])

# get all words that the tokenizer knows
word_index = tokenizer.word_index

# put the tokens in a matrix
X = tokenizer.texts_to_sequences(trainDF['text'])
X = pad_sequences(X)

# prepare the labels
y = pd.get_dummies(trainDF['label'])

# split in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

  
    
#LSTM MODEL 
from keras import backend as K
# init model
model = Sequential()
# emmbed word vectors
model.add(Embedding(len(word_model.wv.vocab)+1,300,input_length=X.shape[1],weights=[embedding_matrix],trainable=False))
# learn the correlations
model.add(LSTM(300,return_sequences=False))
model.add(Dense(y.shape[1],activation="softmax"))
# output model skeleton
# model.summary()

# evaluation functions
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
    

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['acc',f1_m,precision_m, recall_m])

# optimizer "adam" is an adaptive learning rate optimization algorithm that's been designed specifically for training deep neural networks.
# The algorithms leverages the power of adaptive learning rates methods to find individual learning rates for each parameter

history = model.fit(X_train, y_train, validation_split=0.25, epochs=1, batch_size=32, verbose=1)
        
model.save('lstm.h5')                        