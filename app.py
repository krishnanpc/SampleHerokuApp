
from flask import Flask
from flask import request

app = Flask(__name__)
#### Boilerplate code - 1 #####

@app.route('/hello')
def hello():
    return 'Hello, World!'
    
### Code to create the Data File
 
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


# get all words that the tokenizer knows
word_index = tokenizer.word_index

                            
                                                                                    
    
#### Code to load NLP Model and prepare function ####
from keras.preprocessing import sequence
from keras.models import load_model
from keras.preprocessing.text import text_to_word_sequence
from keras.datasets import imdb
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
import gensim, re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding

import tensorflow as tf
    

from keras import backend as K 


# maximum words in each sentence
maxlen = 121
# get the word index from imdb dataset
#word_index = imdb.get_word_index()
# load the Model from file


text = [re.sub(r'([^\s\w]|_)+', '', sentence) for sentence in trainDF['text']]
text = [sentence.lower().split() for sentence in text]

word_model = gensim.models.Word2Vec(text, size=300, min_count=1, iter=10)

#Embedding Matrix
# save the vectors in a new matrix
embedding_matrix = np.zeros((len(word_model.wv.vocab) + 1, 300))
for i, vec in enumerate(word_model.wv.vectors):
  embedding_matrix[i] = vec


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




#Function definitiosn
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
    





# init model
model = Sequential()
# emmbed word vectors
model.add(Embedding(len(word_model.wv.vocab)+1,300,input_length=X.shape[1],weights=[embedding_matrix],trainable=False))
# learn the correlations
model.add(LSTM(300,return_sequences=False))
model.add(Dense(y.shape[1],activation="softmax"))
# output model skeleton
# model.summary()







#model = load_model('lstm_md.h5',custom_objects={'f1_m': f1_m,'precision_m': precision_m,'recall_m': recall_m})

model.load_weights('lstm_model.h5') #CHANGE THIS TO PARTH MODEL


    
global graph
graph = tf.get_default_graph()

# method that does the prediction – we will call this later
def predict_sentiment(my_test):
    
    
    
    
      
    with graph.as_default():
        
        # tokenize the sentence
        X1 = tokenizer.texts_to_sequences([my_test])
        X1 = pad_sequences(X1, maxlen=121)
        result = model.predict_classes(X1,verbose = 0)
        # y_pred = nlp_model.predict(sent_test)
        # K.clear_session()
    # return a predicted sentiment real value between 0 and 1
    return result
#### Code to load NLP Model and prepare function ####   
    
# default HTML to show at first when no input is sent
htmlDefault = '<h4>Simple Python NLP demo</h4><b>Type some text to analyze its sentiment using Deep Learning</b><br><form><textarea rows=10 cols=100 name=\'text_input\'></textarea><br><input type=submit></form>'
# build a route or HTTP endpoint
# this route will read text parameter and analyze it
@app.route('/process')
def process():
             # define returning HTML
             retHTML = ''
             # get the HTTP parameter by name 'text_input'
             in_text = request.args.get('text_input')
             # if input is provided process else show default page
             if in_text is not None:
                 # first show what was typed
                 retHTML += 'TEXT: <b>%s</b>'%(in_text)
                 # run the deep learning Model
                 result = predict_sentiment(in_text)
                 # if positive sentiment
                 if result > 0.5:
                 # if negative sentiment
                     retHTML += 'Class: <b>%s</b>'%(result)
                     
                 else:
                     retHTML += 'CLass: <b>%s</b>'%(result)
# just show
                 return retHTML
             else:
                 return htmlDefault
         ##### New Code #####  
    
# main application run code
if __name__ == '__main__':
     app.run(debug=False,host='0.0.0.0',port=5000)
     
     
     
     
     
     