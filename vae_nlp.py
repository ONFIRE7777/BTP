#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import itertools
import numpy as np
from scipy import spatial
from scipy.stats import norm
import nltk.data
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import reuters
from nltk. corpus import gutenberg
from nltk.corpus import brown
from nltk.tokenize import sent_tokenize
from gensim.models import KeyedVectors
from keras.layers import Input, Dense, Lambda, Layer
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras import backend as K
from keras import metrics
from gensim import models
from sklearn.model_selection import train_test_split


# In[2]:


word2vec_path = 'GoogleNews-vectors-negative300.bin.gz'
w2v = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)


# In[3]:


def split_into_sent (text):
    strg = ''
    for word in text:
        strg += word
        strg += ' '
    strg_cleaned = strg.lower()
    for x in ['\xd5d','\n','"',"!", '#','$','%','&','(',')','*','+',',','-','/',':',';','<','=','>','?','@','[','^',']','_','`','{','|','}','~','\t']:
        strg_cleaned = strg_cleaned.replace(x, '')
    sentences = sent_tokenize(strg_cleaned)
    return sentences


# In[4]:


def vectorize_sentences(sentences):
    vectorized = []
    for sentence in sentences:
        byword = sentence.split()
        concat_vector = []
        for word in byword:
            try:
                concat_vector.append(w2v[word])
            except:
                pass
        vectorized.append(concat_vector)
    return vectorized


# In[5]:


import pandas as pd
#Data Input and Preprocessing
data = pd.read_csv('reddit.csv')
data_mod = pd.DataFrame([[]])

for row in range(len(data["id"])):
    #print(data["id"][row])
    data["id"][row]=(data["id"][row].splitlines())
    for num in range(len(data["id"][row])):
        strin=''.join(data["id"][row][num].split())
        newstr='';
        for i in range(len(strin)-2):
            newstr+=strin[i+2]
        data["id"][row][num]=newstr

for row in range(len(data["hate_speech_idx"])):
        if data["hate_speech_idx"][row]=='n/a':
            data["hate_speech_idx"][row]="[0]"

for row in range(len(data["hate_speech_idx"])):
        if data["hate_speech_idx"][row]=='n/a':
            data["hate_speech_idx"][row]="[0]"

for row in range(len(data["hate_speech_idx"])):
        if isinstance(data["hate_speech_idx"][row],float):
            data["hate_speech_idx"][row]="[0]"  

for row in range(len(data["response"])):
        if isinstance(data["response"][row],float):
            data["response"][row]='This is normal'
            
for row in range(len(data["text"])):
    #print(data["id"][row])
    data["text"][row]=(data["text"][row].splitlines())
    for num in range(len(data["text"][row])):
        strin=' '.join(data["text"][row][num].split())
        newstr='';
        for i in range(len(strin)-2):
            newstr+=strin[i+2]
        data["text"][row][num]=newstr
    
    data["hate_speech_idx"][row]=data["hate_speech_idx"][row].strip('][').split(', ');
    data["hate_speech_idx"][row]=list(map(int, data["hate_speech_idx"][row]))
    data["response"][row] = data["response"][row].replace("\"","\'")
    data["response"][row] = data["response"][row].replace("[\'","")
    data["response"][row] = data["response"][row].replace("\']","")
    data["response"][row] = data["response"][row].split('\', \'');
    
    for idx in data["hate_speech_idx"][row]:
        if idx!=0 and idx<=len(data["text"][row]):
            bad=idx-1;
#            print(bad)
            for resp in data["response"][row]:
                newrow=pd.Series([data["text"][row][bad],resp])
                rowdf=pd.DataFrame([newrow])
                data_mod=pd.concat([data_mod,rowdf],ignore_index=True)
                #print("done")

 




# In[6]:


df=data_mod
df = df.iloc[1:]
df.head()


# In[7]:


data_train, data_test = train_test_split(df, test_size=0.20, random_state=42)
train_x = pd.DataFrame(data_train[0])
train_y = pd.DataFrame(data_train[1])
test_x  = pd.DataFrame(data_test[0])
test_y  = pd.DataFrame(data_test[1])
type(test_y)


# In[8]:


import re
import os
import collections
import string
def remove_punct(text):
    text_nopunct = ''
    text_nopunct = re.sub('['+string.punctuation+']', '', text)
    return text_nopunct

train_x[0] = train_x[0].apply(lambda x: remove_punct(x))
train_y[1] = train_y[1].apply(lambda x: remove_punct(x))
test_x[0] = test_x[0].apply(lambda x: remove_punct(x))
test_y[1] = test_y[1].apply(lambda x: remove_punct(x))
test_y.head()


# In[9]:


from nltk import word_tokenize, WordNetLemmatizer
trainx_tokens = [word_tokenize(sen) for sen in train_x[0]]
trainy_tokens = [word_tokenize(sen) for sen in train_y[1]]
testx_tokens = [word_tokenize(sen) for sen in test_x[0]]
testy_tokens = [word_tokenize(sen) for sen in test_y[1]]


# In[10]:


def lower_token(tokens): 
    return [w.lower() for w in tokens]    
    
lower_tokensx = [lower_token(token) for token in trainx_tokens]
lower_tokensy = [lower_token(token) for token in trainy_tokens]

print((lower_tokensy[0]))


# In[11]:


from nltk.corpus import stopwords
stoplist = stopwords.words('english')


# In[12]:


def remove_stop_wordsy(tokens):
    return [word for word in trainy_tokens if word not in stoplist]


# In[13]:


def remove_stop_wordsx(tokens):
    return [word for word in trainx_tokens if word not in stoplist]


# In[14]:


filtered_wordsx = [remove_stop_wordsx(sen) for sen in lower_tokensx] 


# In[225]:


filtered_wordsy= [remove_stop_wordsy(sen) for sen in lower_tokensy] 


# In[226]:


data_train_x=[[]]
data_train_y=[[]]

for i in range(len(filtered_wordsx[0])):
    x=filtered_wordsx[0][i]
    data_train_x.append(x)
for i in range(len(filtered_wordsy[0])):
    x=filtered_wordsy[0][i]
    data_train_y.append(x)

data_train_y.remove([])
data_train_x.remove([])
print((data_train_y[0]))

resultx = [' '.join(sen) for sen in data_train_x]
resulty = [' '.join(sen) for sen in data_train_y]
#print(len(resultx))
print(resulty[0])
train_x = pd.DataFrame(resultx,columns=['Text'])
train_y = pd.DataFrame(resulty,columns=['Response'])
print(train_y.head())
print(train_x.head())


# # Preprocessing Text

# The preprocessing code is data specific.  
#   
# It is an example of how one can use a pre-trained word2vec to embed sentences into a vector space.

# In[2]:


all_training_words = [word for tokens in data_train_x for word in tokens]
training_sentence_lengths = [len(tokens) for tokens in data_train["tokens"]]
TRAINING_VOCAB = sorted(list(set(all_training_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_training_words), len(TRAINING_VOCAB)))
print("Max sentence length is %s" % max(training_sentence_lengths))


# In[228]:


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_comments, generate_missing=False):
    embeddings = clean_comments['tokens'].apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    return list(embeddings)


# In[ ]:





# In[ ]:


data_array = np.array(data_concat)
np.random.shuffle(data_array)

train = data_array[:8000]
test = data_array[8000:10000]


# # Variational Autoencoder

# In[ ]:


batch_size = 500
original_dim = 3000
latent_dim = 1000
intermediate_dim = 1200
epochs = 200
epsilon_std = 1.0

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# placeholder loss
def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)

# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # we don't use this output, but it has to have the correct shape:
        return K.ones_like(x)

loss_layer = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, [loss_layer])
vae.compile(optimizer='rmsprop', loss=[zero_loss])

#checkpoint
cp = [callbacks.ModelCheckpoint(filepath="/home/ubuntu/pynb/model.h5", verbose=1, save_best_only=True)]

#train
vae.fit(train, train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(test, test), callbacks=cp)

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# build a generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)


# # Generating Text From Latent Space

# In[ ]:


# some matrix magic
def sent_parse(sentence, mat_shape):
    data_concat = []
    word_vecs = vectorize_sentences(sentence)
    for x in word_vecs:
        data_concat.append(list(itertools.chain.from_iterable(x)))
    zero_matr = np.zeros(mat_shape)
    zero_matr[0] = np.array(data_concat)
    return zero_matr


# In[ ]:


# input: original dimension sentence vector
# output: text
def print_sentence_with_w2v(sent_vect):
    word_sent = ''
    tocut = sent_vect
    for i in range (int(len(sent_vect)/300)):
        word_sent += w2v.most_similar(positive=[tocut[:300]], topn=1)[0][0]
        word_sent += ' '
        tocut = tocut[300:]
    print(word_sent)


# In[ ]:


# input: encoded sentence vector
# output: encoded sentence vector in dataset with highest cosine similarity
def find_similar_encoding(sent_vect):
    all_cosine = []
    for sent in sent_encoded:
        result = 1 - spatial.distance.cosine(sent_vect, sent)
        all_cosine.append(result)
    data_array = np.array(all_cosine)
    maximum = data_array.argsort()[-3:][::-1][1]
    new_vec = sent_encoded[maximum]
    return new_vec


# In[ ]:


# input: two points, integer n
# output: n equidistant points on the line between the input points (inclusive)
def shortest_homology(point_one, point_two, num):
    dist_vec = point_two - point_one
    sample = np.linspace(0, 1, num, endpoint = True)
    hom_sample = []
    for s in sample:
        hom_sample.append(point_one + s * dist_vec)
    return hom_sample


# In[ ]:


# input: two written sentences, VAE batch-size, dimension of VAE input
# output: the function embeds the sentences in latent-space, and then prints their generated text representations
# along with the text representations of several points in between them
def sent_2_sent(sent1,sent2, batch, dim):
    a = sent_parse([sent1], (batch,dim))
    b = sent_parse([sent2], (batch,dim))
    encode_a = encoder.predict(a, batch_size = batch)
    encode_b = encoder.predict(b, batch_size = batch)
    test_hom = hom_shortest(encode_a[0], encode_b[0], 5)
    
    for point in test_hom:
        p = generator.predict(np.array([point]))[0]
        print_sentence(p)


# Printing sentences from the training set and comparing them with the original will test whether the custom print function works properly.

# In[ ]:


print_sentence_with_w2v(train[1])
print_sentence_with_w2v(train[2])


# The encoder takes the training set of sentence vectors (concatenanted word vectors) and embeds them into a lower dimensional vector space.

# In[ ]:


sent_encoded = encoder.predict(np.array(train), batch_size = 500)


# The decoder takes the list of latent dimensional encodings from above and turns them back into vectors of their original dimension.

# In[ ]:


sent_decoded = generator.predict(sent_encoded)


# The encoder trained above embeds sentences (concatenated word vetors) into a lower dimensional space. The code below takes two of these lower dimensional sentence representations and finds five points between them. It then uses the trained decoder to project these five points into the higher, original, dimensional space. Finally, it reveals the text represented by the five generated sentence vectors by taking each word vector concatenated inside and finding the text associated with it in the word2vec used during preprocessing.

# In[ ]:


test_hom = shortest_homology(sent_encoded[3], sent_encoded[10], 5)
for point in test_hom:
    p = generator.predict(np.array([point]))[0]
    print_sentence_with_w2v(p)


# The code below does the same thing, with one important difference. After sampling equidistant points in the latent space between two sentence embeddings, it finds the embeddings from our encoded dataset those points are most similar to. It then prints the text associated with those vectors.
#   
# This allows us to explore how the Variational Autoencoder clusters our dataset of sentences in latent space. It lets us investigate whether sentences with similar concepts or grammatical styles are represented in similar areas of the lower dimensional space.

# In[ ]:


test_hom = shortest_homology(sent_encoded[2], sent_encoded[1500], 20)
for point in test_hom:
    p = generator.predict(np.array([find_similar_encoding(point)]))[0]
    print_sentence_with_w2v(p)

