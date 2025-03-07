#!/usr/bin/env python
# coding: utf-8

# In[210]:


from PyPDF2 import PdfReader

paper = PdfReader('A_Conversation_with_Kanti_Mardia.pdf')

count = len(paper.pages)
text = " "
for i in range(count):
    page = paper.pages[i]
    text += page.extract_text()
    
#text


# In[211]:


import re
text = re.sub(r"[\d#]", "", text)
text = re.sub(r"[^\w\s]", '', text)
text = re.sub('\n', ' ', text)
text =  re.sub(r"\+", "", text)
text = re.sub(r"-", "", text)
text= re.sub(r"/", " ", text)
text= re.sub(r"@", " ", text)
text= re.sub(r"_", " ", text)
text=  re.sub(r"\\", " ", text)
text= re.sub(r"=", " ", text)
text= re.sub(r":", " ", text)
text= re.sub(r"\?", " ? ", text)
text= re.sub(r"!", " ! ", text)
text= re.sub(r"&", " ", text)
text= re.sub(r"\|", " ", text)
text= re.sub(r";", " ", text)
text= re.sub(r"\(", " ", text)
text= re.sub(r"\)", " ", text)

#print(text)


# In[212]:


from nltk.stem import PorterStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import MWETokenizer
import collections

mwe = MWETokenizer([('Omar', 'Khayyam')], separator='_')

#ps = PorterStemmer()
#words = word_tokenize(text)
words_new = mwe.tokenize(word_tokenize(text))

count_Omar_Khayyam = words_new.count("Omar_Khayyam")
count_Omar_Khayyam


# In[194]:


from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import collections
wordlist = []
ps = PorterStemmer()
words = word_tokenize(text)

for w in words:
    if w not in stop_words:
        wordlist=ps.stem(w)
        
    
dicwordcount = {}
for word in words:
    if word not in dicwordcount.keys():
        dicwordcount[word] = 1
    else:
        dicwordcount[word] +=1
        
#print(dicwordcount)
n_print = 5000#int(input("How many most common words to print: "))
print("\nOK. The {} most common words are as follows \n".format(n_print))
word_counter = collections.Counter(dicwordcount)
lst = word_counter.most_common(n_print)
df = pd.DataFrame(lst, columns = ['word', 'Count'])
df.to_csv('df.csv', index=False)


# In[215]:


import matplotlib.pyplot as plt 
from wordcloud import WordCloud, STOPWORDS 
import numpy as np
from PIL import Image

mask = np.array(Image.open("khayyam.jpg"))
stop_words = set(stopwords.words('english'))


wc = WordCloud(stopwords = stop_words,
			mask = mask, 
			background_color = "white",
			max_words = 2000,
			max_font_size = 500,
			random_state = 42, 
			width = mask.shape[1],
			height = mask.shape[0])

# Finally generate the wordcloud of the given text
wc.generate(text) 
plt.imshow(wc, interpolation = "None")

# Off the x and y axis
plt.axis('off')

# Now show the output cloud
plt.show()

