Introduction (Understand the problem)
Emojis contribute to the major part in daily conversations and online advertisements. Sometimes, only an emoji without any words in a sentence can explain its complete meaning. Emojis have their own meanings, in every platform they are used. Emojis can be stickers, gifs, small images etc, which can be used in place of words. Human trafficking is one of the special cases where emojis are used with different meanings corresponding to the advertisement. Emojis are used in different ways in trafficking data. Generally, trafficking advertisements have their titles displayed on websites, whose links refers to the complete body. So, titles play a major roles and every one tries to highlight the titles using emojis. Emojis are also used in replacing important words in the title. These can be understood by humans when they read them, but there are thousands of advertisements, which are impossible to be read and understood by humans, so machines should be given enough knowledge to read the titles between words. There is no literal format for the advertisement, knowledge is evolving day-by day.  The emoji used this year to replace a particular word, may not be same for the next year. This is too dynamic to find enough knowledge for machines to understand the literal meanings of emojis.  It is nearly impossible to get the exact meanings of the emojis, but then it is possible to get the similar emojis to be grouped together to see what they contribute to the meaning of sentence and not only see, what can be next replacements to the current emojis in the context. For example, as per ground truth, the rose emoji is used in place of currency but it may not be the only one corresponding to currency, moreover, it may not be the only meaning for the rose emoji. By grouping similar emojis together, we can see which are the emojis that are nearer to the meaning rose in the corpus. 

Machines are used to only numbers and not the words or emojis. So, the words and emojis are needed to be converted to vectors that are used for machine learning techniques. 

Related Work:
Data:
All the titles from backpage are extracted using diffbot created API. There are 56878  titles analysed in total. In backpage many titles are repeated or posted again & again to reach top of the list. So, by removing the redundant data there is very less data to develop very exact vectors.. So, there are only 28,300 unique titles. Emojis are extracted using regex. Values counts for each emojis are collected and sorted according to their values. The total number of emojis used are 186288. Out of which, number of unique emojis are only 905. It is not that all the titles must contain emojis, there are nearly 9271 titles without a single emoji.

Titles are collected in excel. Excel cannot save the emojis in the original format when we look at it. But when extracted, the emojis are extracted in the proper way. 

Data Discovery:
We need a lot of data to understand the semantic meaning. There are many emojis which are used just once and as a contrast some emojis like “??, ? , ??” are used a lot of times than remaining ones, In some titles, they are used as highlighters. The usage of emojis are quite different. Some are used along with words and some as a replacement to word. There are many ways that an emoji can be used in titles. Some of the examples are listed below, 
“??ExY”,  “Lad??3??”, “l??k” are some of the examples, where a word is not used with a normal spelling. 
Either, numbers, special characters or emojis are used in place of letters in a word, which makes it difficult for the machine to read the word. 
Some of the words are changed in spelling intentionally and used for example “garanteed” is used for “guaranteed” , “betiful” is used for “beautiful”.  
In some other words, like “S?t?U?n?N?i?N?g” emojis or symbols are used between each letter, which make them difficult to correct even with a spelling corrector. 
Some of the symbols are not removed in special characters, like “you!!!!” is different from “ you!!!!”. 
Some of the words are used frequently, which don’t have meaning like abbreviations.
“ßl??” is not identified as  “blow” and when we try to remove special characters. It becomes “O”, which cannot be sensed any.
Many numericals contribute to the word meaning. For example, “Dr3aM”, ”t3aM”. In these examples, numericals cannot be removed generally but in the sentences like “Hung masculine Daddy/Dom - 45”. So, removing numericals can lead to removal of important words.
“???????” is not identified as list of emoticons and also they are not considered as a word like “goddess”. 
Though all the normal alpha-numeric characters are present in a word, it is still not converted to lower case. Eg : “HeRe”

Reading emojis from titles is difficult even for humans with polymorphism of emojis, so making machines learn about them requires extensive proper feeding of data and a really lot of data to get the semantic meaning of the data. Semantic meaning of emojis is how humans get the meaning of emojis, the same way the computers are to be trained to learn them. From these observations, 

Model:
Data cleaning is the most important step in every NLP application, from the data discovery, we found many forms of emojis and text in the data. There are emojis that are used just once and there are emojis that are used nearly 12,000 times. First, the emojis are used attached to word, like “?? Amazing ASsets??Killer Curves??Gorgeous Face????  Sky - 22'”, It is difficult to separate the words. So, a technique to insert space between a word and emoji is used. It worked for the necessity but created a problem in situations like, “Lad??3??” is converted as “Lad??3 ??” to two separate. But as there are very less number of this situations. Basic preprocessing steps like, removing punctuation, digits, using only the lemma of words. Finally removing unnecessary words. Many times, there is a specific pattern in title, which has repeated set of emojis, which are used as highlighters, So, removing the repeated characters like “?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ??” to reduce to “?? ?? ?? ?? ??” to get more meaning about them and to analyse what do the emoticons occurring together contribute to.

Machines cannot read the input strings. So, we need to convert them to vectors. Converting words to vectors have a lot of methods like one hot encoded vector, where a dictionary of all the words in corpus is taken and vector of a particular word is formed by placing 1 in the word position and all 0s in the dictionary length vector. This gives a very length vectors, which are difficult to process. So word embedding techniques were evolved, such as word2vec, glovec, facebook fasttext and many. Every algorithm uses same techniques, skip-gram model and continuous bag of words models. These techniques are shallow neural network models, where the context in which two words occurred can be found. Starting with a set of random weights, each weight is updated using each of the title and finally we see which words can be occurred in similar context. This technique of getting the semantic vectors is the machine readable meaning of the word. Same technique can be applied to emojis too.

Forming word vectors
Method 1 :
Started with the technique to map words and emojis into the vector space.  This can show the meaning of emoji related to word. As we already know the meaning of words, if we see the emoji that are similar to that words, that shows what’s the meaning of emoji would be. For this, word2vec is used to develop word vectors and similarity measure was used to find the most highly related words. 

model.most_similar('??',topn=15)
('??', 0.9572778940200806),
 ('??', 0.9472787976264954),
 ('??', 0.9418334364891052),
 ('suck', 0.9349520802497864),
 ('??', 0.9347986578941345),
 ('??', 0.9286587238311768),
 ('??', 0.9279194474220276),
 ('fee', 0.9275314807891846),
 ('alot', 0.9259772300720215),
 ('??', 0.9248500466346741),
 ('avaliabletext', 0.9242998361587524),
 ('shes', 0.9237587451934814),
 ('??', 0.922288715839386),
 ('??', 0.9222823977470398),
 ('nkc', 0.9202950596809387)
This method is fine to find the similar words to the emoji. But in the corpus, words are not properly defined, There are many words with the same meaning but presented in different ways as stated in exceptions. So taking words into context and getting the meaning of emoji did not work well. For this, extensive data modification is needed. Techniques like trafficking customized spell corrector, word predictor for different types of words would come to use. There is another problem, in this case all the word and emojis are formed to a single group, if the similarity is considered, nearly all the similar words/emojis have the same percentage of similarity. This kind of vectors will not help in processing these vectors to machine learning algorithms.

Method 2 :
Considering only emojis for vector formation would be a good idea, to check which emojis appear in the similar context. For this,  only the emojis are extracted from corpus. This method solved the problem of the previous method where all the emojis are bundled together in single group in vector space. In this method, as we can see the similarity metric is meaningful to get the most similar emojis. In this method, emojis are converted to lists and a word2vec from gensim is used with it’s parameters and similar emoticons are extracted for each emojis, like 
model.most_similar('??',topn=15)
('??', 0.8189571499824524),
 ('??', 0.7896915674209595),
 ('?', 0.7673460245132446),
 ('?', 0.7438008189201355),
 ('??', 0.7414641380310059),
 ('??', 0.6909445524215698),
 ('?', 0.6819630861282349),
 ('?', 0.6795470118522644),
 ('??', 0.6777126789093018),
 ('??', 0.6762962341308594),
 ('??', 0.6716013550758362),
 ('??', 0.6629748344421387),
 ('??', 0.6623550057411194),
 ('??', 0.6570390462875366),
 ('??', 0.653336226940155)

This gave almost similar emojis as per ground truth, this method cannot give the exact meaning of the emoji but the emojis that are similar to the present emoji. This technique is majorly used to find which emojis are used in similar context, which emoji can replace another, in a dynamic environment like trafficking it is really important to know which emoji is the replacement in future. For example, in current situation rose is used in place of currency, this method can predict what can be next rose.

With filters : One other variation of method 2, by removing the highlighter emojis in the corpus like, the most used emojis like “?? ??” are removed before passing to word2vec, In this case, the all the emojis are formed as a single cloud, it became difficult to find the most similar ones. This may be because, the emojis ?? ?? may not be highlighters but contribute to some important meaning in the title.

Variations in word2vec:
Word2vec has certain parameters which contribute to the vectors formed. The parameters included are,

Increase in the amount of data, gives more accurate vectors, as algorithm will have many examples to learn from. It gives answers to, which emoticons can be used in similar context, what is the next replacement to the particular emoticon? And so on. 

Feature vector length : Word2vec deals with projecting a one hot encoded vector to a less dimensional vector. The number of dimensions to be chosen is actually very random provided it is not too less nor too big. The dimension of the vector  to be selected going by intuition, because the word2vec founder, google to selected the 300 dimension too randomly. 
Context window : This contributes to the vector formation. Here the titles are too short to select the high context window. I have chosen the smallest title length as context window not to lose any important features in smaller titles.
Minimum word count : This is important. There are many emojis that are just used for only once or twice. These kind of emojis act as outliers for determining actual vectors. The highest count of emoji occurrence is 13,000 out of 1 million emojis used. So considering this count, minimum word count is taken between 5 to 10.

Visualizing emoji vectors :
The emoji vectors found is of high-dimensional, which cannot be visualized. So to present them on a two-dimensional or three-dimensional vectors, there are two kind of algorithms followed,

PCA(Principal component analysis) : This is the linear conversion of high dimensional vector to a lower one. This is, taking as a linear equation of all the dimensions to form a lower dimensional one, which is two. (Used here)
T-SNE(T-distributed stochastic neighbour embedding) : The linear transformation would not work for keeping the related distance between any two points, so the nonlinear transformation is used(scikit-library). In this case, the probability of each point nearer to other point is given in the form of cost function and a gradient descent is used on it to reduce the error on cost function.
So, using T-SNE two-dimensional vectors are generated and  plotted in excel along with emojis as labels. This gives the intuitive difference between two emojis. This can be used to 
See which emojis are close together in the context
See which emojis are replacements of each other.
Used for visualizing the clusters.
Get the intuitive meanings of emojis by looking at the ground truth emojis like,


?
(69) Sexual position where both partners perform oral sex simultaneously
?
(airplane) "new in town" - Indicative of movement of trafficked victim
??
(cherry) Describing a younger/underage victim
??
(Kiss/lips) currency
??
(lollipop) Describing a younger/underage victim
??
(not 18) Describing a younger/underage victim
??
(rose) currency
??
100% real






Using emoji vectors :
The emoji vectors are a set of feature vectors, which can be used as input to any machine learning algorithms. In this human trafficking corpus, there is not much ground truth to classify the emojis, so the unsupervised clustering is used. 

K-means clustering : This is hard clustering, where we give the number of clusters initially and all the spherical clusters are formed. Sample clusters formed are,

['??', '?', '??', '??', '??', '??', '??', '??', '??', '?', '??', '??', '?', '??', '??', '??', '??', '??', '?', '??', '?', '?', '??', '??', '??', '??', '??', '??', '?', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '?', '?', '?', '??', '??', '?', '??', '??', '??', '??', '??', '??', '??', '??', '?', '??', '??', '??', '??', '??', '??', '??', '??', '?', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '?', '??', '?', '?', '?', '??', '??', '?', '??', '?', '??', '??', '??', '??', '??', '?', '?', '??', '?', '?', '??', '??', '?', '?', '??', '??', '??', '??', '??', '??', '??', '?', '??', '??', '??', '??', '??', '?', '??', '??', '?', '?', '?', '?', '??', '??', '?', '??', '??', '??', '??', '??', '??', '??', '??', '?', '??', '??', '??', '?', '??', '?', '?', '?', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '?', '??', '??', '?', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '?', '??', '??', '??', '?', '?', '??', '??', '?', '??', '??', '??', '?', '??', '??', '??', '?', '??', '?', '?', '?', '?', '??', '??', '??', '??', '??', '??', '??', '??', '??', '?', '?', '?', '??', '??', '??', '??', '??', '?', '??', '??', '??', '?', '??', '??', '??', '??', '??', '?', '??', '??', '??', '??', '??', '??', '?', '??', '?', '??', '??', '?', '??', '?', '?', '??', '??', '?', '?', '?', '?', '?', '??', '??', '?', '??', '??', '??', '??', '??', '?', '??', '??', '??', '?', '??', '?', '?', '??', '??', '??', '??', '??', '??', '?', '??', '??', '??', '?', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '?', '??', '??', '??', '??', '?', '??', '??']

['??', '??', '??', '??', '??', '??', '?', '??', '??', '??', '??', '??', '??', '??', '??', '?', '??', '??', '??', '??', '?', '??']

['??', '??', '?', '??', '??', '??', '??', '?', '??', '??', '??', '??', '??', '?', '??', '?', '??', '?', '?', '??', '??', '??', '??', '??', '??', '??', '??', '?', '?', '??']
['??', '?', '??', '??', '??', '?', '??', '?', '??', '??', '??', '??', '?', '?', '??', '?', '??', '?', '??', '??', '??', '??', '??', '??', '??', '??', '?', '?']

Which showed that, a single large cluster is formed. When we see a graph, we see that there is a single large group of clusters which necessarily need not be same. 
model.most_similar('??',topn=15)
('??', 0.8189571499824524),
 ('??', 0.7896915674209595),
 ('?', 0.7673460245132446),
 ('?', 0.7438008189201355),
 ('??', 0.7414641380310059),
 ('??', 0.6909445524215698),
 ('?', 0.6819630861282349),
 ('?', 0.6795470118522644),
 ('??', 0.6777126789093018),
 ('??', 0.6762962341308594),
 ('??', 0.6716013550758362),
 ('??', 0.6629748344421387),
 ('??', 0.6623550057411194),
 ('??', 0.6570390462875366),
 ('??', 0.653336226940155)
The above similarity measure says that '??’ has very less similar emojis such as '??','??','?','?','??'. But when in the clusters, '??’ is grouped into a large cluster, loosing it’s outlier’s nature. K-means cannot be used to identify outliers, it groups everything into one large cluster, it is not good in taking the importance of emoji vectors. 

Affinity Propagation:
This is another type of clustering algorithm, which is good at getting outliers, it is good at identifying how many clusters would the data need. It works by taking each data point as a centroid of the cluster and sees if it can form a cluster by passing message to other data points. 
It takes the input as only the data points and return the number of clusters and their data points. This algorithm is very useful when we do not know how many clusters to form. In k-means clustering, clusters are spherical. Each centroid can include all the points within radius and tries to cover all algorithms. Affinity propagation algorithm is good at identifying the outliers i..e, it is not obliged to include all the data into clusters, It excludes those points that cannot form to a cluster, it groups outliers, it perfectly gets the points that are not nearer.



