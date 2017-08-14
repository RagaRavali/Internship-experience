
# coding: utf-8

# # Getting data

# In[1]:

import requests
import json


apikey = 'bcc855fd71b859791b2202d8297da1e3'
crawl_name = 'GEN'
url = "https://api.diffbot.com/v3/crawl/data"
urlData = {"token": apikey, "name": crawl_name, "format": json}
	# Go get the data
r = requests.get(url,urlData)
data = json.loads(r.text)
dict = {}
df = []
for rec in data:
    #dict[rec['text']] = rec['title']
    df.append(rec['title'])
print(len(df))


# saved data in excel

# In[7]:

import pandas as pd
df = list(set(df))
save_df = pd.DataFrame(df)
path = "E:/intern/learn emojis from vector space models/data.csv"
save_df.to_excel(path)
writer = pd.ExcelWriter('data.xlsx', engine='xlsxwriter')
save_df.to_excel(writer, sheet_name='Sheet1')
writer.save()


# importing data from excel and converting to strings

# In[1]:

import pandas as pd

df = pd.read_excel("E:/intern/learn emojis from vector space models/data.xlsx")
df = df[0].astype(str)
df = set(df)
print(df)


# In[2]:

import re 
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plot

try:  
    # UCS-4
    e = re.compile(u'['
        u'\U0001F300-\U0001F64F'
        u'\U0001F680-\U0001F6FF'
        u'\u2600-\u26FF\u2700-\u27BF]', 
        re.UNICODE)
    #e = re.compile('U+2600-U+26FF')
except re.error:  
    # UCS-2
    e = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    e = re.compile(u'(('
        u'\ud83c[\udf00-\udfff]|'
        u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
        u'[\u2600-\u26FF\u2700-\u27BF]))', 
        re.UNICODE)
emojis = []  
for x in df:
        match = e.findall(x)
        emojis.append(match)

empty_lists = 0
for x in emojis:
    if x == []:
        empty_lists = empty_lists+1
        
print("There are ", empty_lists , "titles without using emojis out of ", len(df), " titles")

emojis = [x for x in emojis if x!=[]]

sum = 0
combined = []
for i in emojis:
    combined.extend(i)
    sum = sum+ len(i)
    

print("There are total ",sum, " emojis used in the data")
result = sum/len(emojis)
print("On an average, there are ", result , " in each title.")
combined = []
for i in emojis:
    combined.extend(i)
print("number of unique emojis used : ", len(set(combined)))
        
dfe =  pd.DataFrame(combined,columns=['text'])
s = pd.Series(' '.join(dfe['text']).split()).value_counts(ascending=True)
print(s)


# In[3]:

import re 
import pandas as pd

try:  
    # UCS-4
    e = re.compile(u'['
        u'\U0001F300-\U0001F64F'
        u'\U0001F680-\U0001F6FF'
        u'\u2600-\u26FF\u2700-\u27BF]+', 
        re.UNICODE)
    #e = re.compile('U+2600-U+26FF')
except re.error:  
    # UCS-2
    e = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    e = re.compile(u'(('
        u'\ud83c[\udf00-\udfff]|'
        u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
        u'[\u2600-\u26FF\u2700-\u27BF]+))', 
        re.UNICODE)

def convert_spaces(title):
    new_title = e.sub(r' \g<0> ',title)
    return new_title


df = [convert_spaces(x) for x in df]
df = list(set(df))
print(len(df))
print(df)



# In[3]:

####################This is for extracting all the emoticons only and not the words. Finding the most likely used emoticons together.
    
import re

exclude = {'/','"', '.',"'",'&', '-', '#', '(', ')', '$', '@', '%', '!','‼️‼️','‼️', '}', '[', ':', ']', '_', '|', ';', '{', '?', '=', '\\', '~', '*', ',', '^', '`','!'}

def preprocess(tweet):
    tweet = ''.join([i.lower() for i in tweet if i not in exclude])
    tweet = re.sub("[a-zA-Z0-9_]+", "",tweet)
    
    tweet = re.sub(" +"," ",tweet)

    return tweet

tweets = [preprocess(x) for x in df]
#print(tweets)


# In[4]:

import re 
import pandas as pd
from collections import Counter

try:  
    # UCS-4
    e = re.compile(u'['
        u'\U0001F300-\U0001F64F'
        u'\U0001F680-\U0001F6FF'
        u'\u2600-\u26FF\u2700-\u27BF]', 
        re.UNICODE)
    #e = re.compile('U+2600-U+26FF')
except re.error:  
    # UCS-2
    e = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    e = re.compile(u'(('
        u'\ud83c[\udf00-\udfff]|'
        u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
        u'[\u2600-\u26FF\u2700-\u27BF]))', 
        re.UNICODE)
    
emojis = [] 
#print(emojis)
for x in tweets:
        match = e.findall(x)
        emojis.append(match)
combined = []

for i in emojis:
    combined.extend(i)

#print(set(combined))
emoji_strings = []        
for i in emojis:
    string = ' '.join(str(e) for e in i)
    emoji_strings.append(string)
    
emoji_strings = list(filter(None, emoji_strings))
print(emoji_strings)
emo = emoji_strings


# In[5]:

import itertools

def remove_space(sent):
    sent = sent.replace(" ","")
    #sent = ''.join(ch for ch, _ in itertools.groupby(sent))
    sent = sent.replace(""," ")
    #sent = sent.replace("💋","")
    #sent = sent.replace("❤","")
    #sent = sent.replace("💦","")
    sent = sent.strip()
    return sent
    
emoji_strings = [remove_space(x) for x in emoji_strings]

print(emoji_strings)


# In[6]:

wt = [list(x.split()) for x in emoji_strings]

num_features = 30   # Word vector dimensionality  
min_word_count = 5   # Minimum word count  
num_workers = 10       # Number of threads to run in parallel  
context = 5          # Context window size  
downsampling = 1e-3   # Downsample setting for frequent words

from gensim.models import word2vec  
 
model = word2vec.Word2Vec(wt, workers=num_workers,   
            size=num_features, min_count = min_word_count, 
            window = context, sample = downsampling)

model.init_sims(replace=True)  


# In[8]:

model.most_similar('🙌',topn=15)


# In[7]:

print(model['💘'])


# In[10]:

vocab = list(model.wv.vocab.keys())
emo = list(set(combined))
final_missing = [x for x in emo if x not in vocab]

print(len(vocab))
print(len(final_missing))
print(final_missing)


# In[88]:

from sklearn.cluster import KMeans

#w = [list(x.split( )) for x in emoji_strings]
vectors = []

for i in set(vocab):
    vectors.append(model[i])
    
#print(len(vectors))
    
#print(word_vectors)
kmeans_clustering = KMeans( n_clusters = 10)
idx = kmeans_clustering.fit_predict( vectors )

#print(len(idx))

sum = 0
for k in range(0,10):
    j = [vectors[i] for i,val in enumerate(idx) if val==k]
    sum = sum + len(j)
print("Total number of emojis present : ",sum)

for k in range(0,10):
    cluster = [vocab[i] for i,val in enumerate(idx) if val == k]
    print(cluster)


# In[11]:

from sklearn.cluster import AffinityPropagation

vectors = []

for i in set(vocab):
    vectors.append(model[i])
    
ap_clustering = AffinityPropagation()
idx = ap_clustering.fit_predict( vectors )


for k in range(0,75):
    cluster = [vocab[i] for i,val in enumerate(idx) if val == k]
    print(cluster)


# In[15]:

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

#kmeans_clustering = KMeans( n_clusters = 10 )
#idx = kmeans_clustering.fit_predict( vectors )

X = model[vocab]

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

kmeans_clustering = KMeans( n_clusters = 10 )
idx = kmeans_clustering.fit_predict( X_tsne )
df_e = zip(X_tsne[:,0],X_tsne[:,1],vocab)
lis =[]
for i in df_e:
    lis.append(i)


df = pd.DataFrame(lis, columns = ["0", "1", "emo"])
#print(df)
path = "E:/intern/learn emojis from vector space models/result.xlsx"
df.to_excel(path)
writer = pd.ExcelWriter('result.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()


# In[43]:

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
import pandas as pd
import seaborn as sns

#kmeans_clustering = KMeans( n_clusters = 10 )
#idx = kmeans_clustering.fit_predict( vectors )

X = model[vocab]

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

ap_clustering = AffinityPropagation()
idx = ap_clustering.fit_predict( X_tsne )
df_e = zip(X_tsne[:,0],X_tsne[:,1],vocab)
lis =[]
for i in df_e:
    lis.append(i)


df = pd.DataFrame(lis, columns = ["0", "1", "emo"])

for k in range(0,21):
    cluster = [vocab[i] for i,val in enumerate(idx) if val == k]
    print(cluster)


# In[45]:

plt.scatter(df['0'],df['1'])

x = []
for i in df['0']:
    x.append(i)
y = []
for i in df['1']:
    y.append(i)
dict = {}
j = 0

for i in vocab:
    dict[j] = vocab
    j = j+1
print(len(dict))
plt.show()


# In[50]:

import numpy as np
print(max(idx))
rands = np.random.random_sample((10000,))
for k in range(0,max(idx)):
    cluster = [X_tsne[i] for i,val in enumerate(idx) if val == k]
    cluster = np.array(cluster)
    plt.scatter(cluster[:,0],cluster[:,1],color = (rands[k],rands[k],rands[k]))

plt.show()


# In[ ]:



