
# coding: utf-8

# In[4]:

import requests
import json


apikey = 'bcc855fd71b859791b2202d8297da1e3'
crawl_name = 'GEN'
url = "https://api.diffbot.com/v3/crawl/data"
urlData = {"token": apikey, "name": crawl_name, "format": json}
	# Go get the data
r = requests.get(url,urlData)
data = json.loads(r.text)
df = []
for rec in data:
    df.append(rec['title'])
    
print(len(df))


# In[2]:

df = set(df)
print(len(df))
#print(dict.values())
import re 
import pandas as pd

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

def convert_spaces(title):
    new_title = e.sub(r' \g<0> ',title)
    return new_title


df = [convert_spaces(x) for x in df]
df = list(set(df))
print(df)


# In[4]:

#df = dict.values()
###########This is pre processing of the text in the title.
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

exclude = {'/','"', '.','&', '-', '#', '(', ')', '$', '@', '%', '!','‼️‼️','‼️', '}', '[', ':', ']', '_', '|', ';', '{', '?', '=', '\\', '~', '*', ',', '^', '`','!'}
stop = stopwords.words('english')
stop.append("i'll") #appending all the commonly used abbrevations
stop.append("‼️‼️")
stop.append("‼️‼️you")

#print(stop)
lemmatizer = WordNetLemmatizer()
def preprocess(tweet):
    tweet = tweet.lower()
    tweet = ''.join([i.lower() for i in tweet if i not in exclude])
    tweet = ''.join([i for i in tweet if not i.isdigit()])
    #tweet = ' '.join( [w for w in tweet.split() if len(w)>1] )
    tweet =" ".join(x.lower() for x in tweet.split() if x not in stop)
    tweet =" ".join(lemmatizer.lemmatize(x,'v') for x in tweet.split())
    tweet =" ".join(lemmatizer.lemmatize(x) for x in tweet.split())
    tweet = re.sub('((www\.[^\s]+)|(https://[^\s]+))',' ',tweet)
    tweet = re.sub("http\S+", " ", tweet)
    tweet = re.sub("https\S+", " ", tweet)
    
    return tweet

tweets = [preprocess(x) for x in df]
print(tweets)


# In[2]:

foo = "🌆🌆🌆🌆🌆 🌆 🌆 🌆 🌆 🌆 🌆 🌆 🌆 🌆 🌆 🌆 🌆 🌆 new management amaze staff 🌆 🌆 🌆 🌆 🌆 🌆 🌆 🌆 🌆 🌆 🌆 🌆 🌆 🌆 🌆 🌆 🌆 🌆 🌆 🌆"
import itertools
''.join(ch for ch, _ in itertools.groupby(foo))


# In[2]:

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
for x in tweets:
        match = e.findall(x)
        emojis.append(match)
        

    
print("There are total ",sum, " emoticonns used in the data")
print("On an average, there are ", int(sum/len(emojis)), " in each title.")
combined = []
for i in emojis:
    combined.extend(i)
print(set(combined))
        
dfe =  pd.DataFrame(combined,columns=['text'])
s = pd.Series(' '.join(dfe['text']).split())
s.hist()


# In[1]:

wt = [list(x.split()) for x in tweets]

# Set values for various parameters
num_features = 2    # Word vector dimensionality  
min_word_count = 1   # Minimum word count  
num_workers = 4       # Number of threads to run in parallel  
context = 5          # Context window size  
downsampling = 1e-3   # Downsample setting for frequent words

from gensim.models import word2vec  
 
model = word2vec.Word2Vec(wt, workers=num_workers,   
            size=num_features, min_count = min_word_count, 
            window = context, sample = downsampling)

model.init_sims(replace=True)  


# In[27]:

model.most_similar('🙌',topn=15)


# In[25]:

model.most_similar('💸',topn=15)


# In[19]:

from sklearn.cluster import KMeans

vocab = list(model.wv.vocab.keys())
vectors = []
vocab_emoji = list(set(combined))
for i in set(combined):
    vectors.append(model[i])
    
print(len(vectors))
    
#print(word_vectors)
kmeans_clustering = KMeans( n_clusters = 10 )
idx = kmeans_clustering.fit_predict( vectors )


for k in range(0,10):
    cluster = [vocab_emoji[i] for i,val in enumerate(idx) if val == k]
    print(cluster)
    
    


# In[9]:

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

X = model[combined]

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
#plt.show()

vocabulary = model.wv.vocab
for label, x, y in zip(vocabulary, X_tsne[:, 0], X_tsne[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
plt.show()


# In[ ]:



