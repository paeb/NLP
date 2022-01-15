# Q & A Chatbot with Python

import pickle # special file format
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout
from keras.layers import add, dot, concatenate
from keras.layers import LSTM
from keras.models import load_model
import matplotlib.pyplot as plt

# VECTORIZING THE DATA

# for each row/tuple in the data, we have the story, the question, and the answer
# and we must convert the strings into numerical data
# the story is parsed into a list of individual words/tokens, same process for the question
# format of each dataset: a list of tuples

# training data
with open('nlp_course_notes/06-Deep-Learning/train_qa.txt','rb') as f:
    train_data = pickle.load(f)

# testing data
with open('nlp_course_notes/06-Deep-Learning/test_qa.txt','rb') as f:
    test_data = pickle.load(f)

# let's make a vocabulary of each word in our dataset (for tokenization later)

vocab = set() # set that holds the vocabulary (words used)

all_data = train_data + test_data

for story,question,answer in all_data:
    vocab = vocab.union(set(story)) # use set() to obtain a list of unique words in the story
    # union() combines the current vocab with the specific story's vocab
    vocab = vocab.union(set(question))

vocab.add('no') # include the two possible answers
vocab.add('yes')

vocab_len = len(vocab) + 1 # add 1 for padding later, because index 0 is reserved

# we need to pad the sequences so that every input is a vector of the same length

# all_story_lengths = [len(data[0]) for data in all_data]
max_story_len = max([len(data[0]) for data in all_data])
max_question_len = max([len(data[1]) for data in all_data])

# Reserve index 0 for pad_sequences
vocab_size = len(vocab) + 1

# we need to vectorize the dataset (convert into numerical vectors)

tokenizer = Tokenizer(filters=[]) # don't use filters in tokenization
tokenizer.fit_on_texts(vocab) # assign a number to each word in the vocab

train_story_text = []
train_question_text = []
train_answers = []

for story,question,answer in train_data:
    train_story_text.append(story)
    train_question_text.append(question)
    train_answers.append(answer)

train_story_seq = tokenizer.texts_to_sequences(train_story_text) # replace each word with its associated number

# use vectorization to transform the data (stories, questions, answers) into numerical format
def vectorize_stories(data,word_index=tokenizer.word_index,max_story_len=max_story_len,max_question_len=max_question_len):

    # STORIES
    X = []
    # QUESTIONS
    Xq = []
    # ANSWERS
    Y = []

    for story,query,answer in data:

        x = [word_index[word.lower()] for word in story] # convert each word in the story to a number
        xq = [word_index[word.lower()] for word in query] # convert each word in the query to a number

        y = np.zeros(len(word_index) + 1) # make the answer an array of 0's

        y[word_index[answer]] = 1 # set the yes/no index value as 1

        X.append(x)
        Xq.append(xq)
        Y.append(y)

    # pad the sequences so each vector is the same length
    # no need to pad the output sequences because they are the same length (either yes / no)
    return (pad_sequences(X,maxlen=max_story_len),pad_sequences(Xq,maxlen=max_question_len),np.array(Y))

inputs_train, queries_train, answers_train = vectorize_stories(train_data)
inputs_test, queries_test, answers_test =  vectorize_stories(test_data)

# BUILD THE ENCODERS (embedding layers)

# PLACEHOLDERS, for storing inputs later on
input_sequence = Input((max_story_len,)) # shape = (max_story_len,batch_size)
question = Input((max_question_len,))
vocab_size = len(vocab) + 1 # number of distinct words used (vocabulary)

# SENTENCES / STORIES

# Input Encoder M (input -> sequence of input vectors)
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size,output_dim=64))
input_encoder_m.add(Dropout(0.3)) # randomly "turns off" some of the neurons (ignoring some inputs and avoids overfitting)
# Output: (samples, story_maxlen, embedding_dim)

# Input Encoder C (input -> sequence of input vectors)
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size,output_dim=max_question_len))
input_encoder_c.add(Dropout(0.3)) # randomly "turns off" some of the neurons (ignoring some inputs and avoids overfitting)
# Output: (samples, story_maxlen, max_question_len)

# QUESTIONS / QUERIES

question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,output_dim=64,input_length=max_question_len))
question_encoder.add(Dropout(0.3)) # randomly "turns off" some of the neurons (ignoring some inputs and avoids overfitting)
# Output: (samples, question_maxlen, embedding_dim)

# USE THE ENCODERS TO ENCODE THE INPUTS (STORIES & QUESTIONS)

input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

# COMPUTE THE MATCH BETWEEN THE FIRST INPUT VECTOR SEQUENCE AND THE QUESTION (USING THE DOT PRODUCT)
match = dot([input_encoded_m,question_encoded],axes=(2,2))
match = Activation('softmax')(match)

# PRODUCE THE RESPONSE FROM THE CHATBOT
response = add([match,input_encoded_c]) # we take the match and then add it to the second input sequence
response = Permute((2,1))(response) # Change the order of the dimensions

answer = concatenate([response,question_encoded]) # combine the response vector with the question vector, this is a tensor

# MAKE THE LSTM MODEL
answer = LSTM(32)(answer) # we need to reduce the tensor into a response/output
# output shape: (samples, 32)

answer = Dropout(0.5)(answer) # make sure that we don't overfit

answer = Dense(vocab_size)(answer) # outputs vectors (bunch of zeros and we should only see values for YES/NO)

answer = Activation('softmax')(answer) # turn the vector into a 0 or 1

model = Model([input_sequence,question],answer) # by putting in the answer, we link the encodings to the model

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy']) # these are paramaters that affect
# how the network learns the weights

# TRAIN THE MODEL ON THE TEST DATA AND USE VALIDATION DATA
history = model.fit([inputs_train,queries_train],answers_train,batch_size=32,epochs=1,validation_data=([inputs_test,queries_test],answers_test))

# summarize history for accuracy (objective is to increase accuracy)
'''
plt.plot(history.history['accuracy']) # training accuracy
plt.plot(history.history['val_accuracy']) # validation/test accuracy
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''

# model.save('name.h5') -> how to save a custom model for later

filename = 'nlp_course_notes/06-Deep-Learning/chatbot_120_epochs.h5'

# filename = 'brandon_model.h5' # use pre-trained weights to save time
# filename = 'chatbot_10.h5'
# model.save(filename)

model.load_weights(filename)
# model = load_model(filename)

# CUSTOM-MADE USER INTERFACE

finished = False

print("Welcome to Brandon Pae's Q&A Chatbot! \n Here is the full vocabulary: ")
for element in vocab:
    print(element + "   |   ",end="")
print("")

while not finished:
    user_story = input("Please input a story (and add spaces around punctuation): ")
    user_question = input("Please input a question (and add spaces around punctuation): ")
    user_data = [(user_story.split(),user_question.split(),'yes')] # the answer doesn't matter here
    user_story, user_ques, user_ans = vectorize_stories(user_data)
    pred_results = model.predict(([user_story,user_ques])) # outputs the probabilities for every single vocabulary word

    val_max = np.argmax(pred_results[0])

    for key,val in tokenizer.word_index.items():
        if val == val_max:
            k = key
            print("Chatbot Response: " + str(k))

    finished = input("Would you like to continue? [Yes/No] ").lower()
    if finished == "yes":
        finished = False
    else:
        finished = True

print("Chatbot interaction complete.")