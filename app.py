# -*- coding: utf-8 -*-
"""
# MANIFESTO ANALYSIS

## IMPORTING LIBRARIES
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install tika
# !pip install clean-text
# !pip install gradio

# Commented out IPython magic to ensure Python compatibility.

import io
import random
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
#import tika
#from tika import parser
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from cleantext import clean
import textract

import nltk.corpus  
from nltk.text import Text
from io import StringIO
import sys 
import pandas as pd
# import cv2
import re

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from textblob import TextBlob
from PIL import Image
import os
import gradio as gr
from zipfile import ZipFile
import contractions
import unidecode


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')


"""## PARSING FILES"""

#def Parsing(parsed_text):
  #parsed_text=parsed_text.name
  #raw_party =parser.from_file(parsed_text) 
 # raw_party = raw_party['nt']
#  return clean(raw_party)
  
def Parsing(parsed_text):
  parsed_text=parsed_text.name
  raw_party =textract.process(parsed_text, encoding='ascii',method='pdfminer') 
  return clean(raw_party)


#Added more stopwords to avoid irrelevant terms
stop_words = set(stopwords.words('english'))
stop_words.update('ask','much','thank','etc.', 'e', 'We', 'In', 'ed','pa', 'This','also', 'A', 'fu','To','5','ing', 'er', '2')

"""## PREPROCESSING"""

def clean_text(text):
  '''
  The function which returns clean text
  '''
  text = text.encode("ascii", errors="ignore").decode("ascii")  # remove non-asciicharacters
  text=unidecode.unidecode(text)# diacritics remove
  text=contractions.fix(text) # contraction fix
  text = re.sub(r"\n", " ", text)
  text = re.sub(r"\n\n", " ", text)
  text = re.sub(r"\t", " ", text)
  text = re.sub(r"/ ", " ", text)
  text = text.strip(" ")
  text = re.sub(" +", " ", text).strip()  # get rid of multiple spaces and replace with a single
  
  text = [word for word in text.split() if word not in stop_words]
  text = ' '.join(text)
  return text

# text_Party=clean_text(raw_party)

def Preprocess(textParty):
  '''
  Removing special characters extra spaces
  '''
  text1Party = re.sub('[^A-Za-z0-9]+', ' ', textParty) 
  #Removing all stop words
  pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
  text2Party = pattern.sub('', text1Party)
  # fdist_cong = FreqDist(word_tokens_cong)
  return text2Party





# Using Concordance,you can see each time a word is used, along with its 
# immediate context. It can give you a peek into how a word is being used
# at the sentence level and what words are used with it. 

def concordance(text_Party,strng):
  word_tokens_party = word_tokenize(text_Party)
  moby = Text(word_tokens_party) 
  resultList = []
  for i in range(0,1):
      save_stdout = sys.stdout
      result = StringIO()
      sys.stdout = result
      moby.concordance(strng,lines=10,width=82)    
      sys.stdout = save_stdout      
  s=result.getvalue().splitlines()
  return result.getvalue()
  

def normalize(d, target=1.0):
   raw = sum(d.values())
   factor = target/raw
   return {key:value*factor for key,value in d.items()}

def fDistance(text2Party):
  '''
  most frequent words search
  '''
  word_tokens_party = word_tokenize(text2Party) #Tokenizing
  fdistance = FreqDist(word_tokens_party).most_common(10)
  mem={}
  for x in fdistance:
    mem[x[0]]=x[1]
  return normalize(mem)

def fDistancePlot(text2Party,plotN=30):
  '''
  most frequent words visualization
  '''
  word_tokens_party = word_tokenize(text2Party) #Tokenizing
  fdistance = FreqDist(word_tokens_party)
  plt.figure(figsize=(4,6))
  fdistance.plot(plotN)
  plt.savefig('distplot.png')
  plt.clf()



def getSubjectivity(text):
   return TextBlob(text).sentiment.subjectivity

# Create a function to get the polarity
def getPolarity(text):
   return  TextBlob(text).sentiment.polarity
  
  
def getAnalysis(score):
  if score < 0:
    return 'Negative'
  elif score == 0:
    return 'Neutral'
  else:
    return 'Positive'



def analysis(Manifesto,Search):
  raw_party = Parsing(Manifesto)
  text_Party=clean_text(raw_party)
  text_Party= Preprocess(text_Party)

  df = pd.DataFrame(raw_party.split('\n'), columns=['Content'])
  df['Subjectivity'] = df['Content'].apply(getSubjectivity)
  df['Polarity'] = df['Content'].apply(getPolarity)
  df['Analysis on Polarity'] = df['Polarity'].apply(getAnalysis)
  df['Analysis on Subjectivity'] = df['Subjectivity'].apply(getAnalysis)
  plt.title('Sentiment Analysis')
  plt.xlabel('Sentiment')
  plt.ylabel('Counts')
  plt.figure(figsize=(4,6))
  df['Analysis on Polarity'].value_counts().plot(kind ='bar')
  plt.savefig('./sentimentAnalysis.png')
  plt.clf()
  
  plt.figure(figsize=(4,6))
  df['Analysis on Subjectivity'].value_counts().plot(kind ='bar')
  plt.savefig('sentimentAnalysis2.png')
  plt.clf()  
  
  wordcloud = WordCloud(max_words=2000, background_color="white",mode="RGB").generate(text_Party)
  plt.figure(figsize=(4,3))
  plt.imshow(wordcloud, interpolation="bilinear")
  plt.axis("off")
  plt.savefig('wordcloud.png')
  plt.clf()  
  
  fdist_Party=fDistance(text_Party)
  fDistancePlot(text_Party)

#   img1=cv2.imread('/sentimentAnalysis.png')
#   img2=cv2.imread('/wordcloud.png')
#   img3=cv2.imread('/sentimentAnalysis2.png')
#   img4=cv2.imread('/distplot.png')
  
  searchRes=concordance(text_Party,Search)
  searChRes=clean(searchRes)
  searChRes=searchRes.replace(Search,"\u0332".join(Search))
  return searChRes,fdist_Party
# ,img4,img1,img2,img3

  
Search_txt=gr.inputs.Textbox()   
filePdf = gr.inputs.File()
text = gr.outputs.Textbox(label='SEARCHED OUTPUT')
mfw=gr.outputs.Label(label="Most Relevant Topics")
# mfw2=gr.outputs.Image(label="Most Relevant Topics Plot")
plot1=gr.outputs. Image(label='Sentiment Analysis')
plot2=gr.outputs.Image(label='Word Cloud')
plot3=gr.outputs.Image(label='Subjectivity')
plot4=gr.outputs.Image(label='Frequency Distribution')

io=gr.Interface(fn=analysis, inputs=[filePdf,Search_txt], outputs=[text,mfw,plot4,plot1,plot2,plot3],examples=[['Bjp_Manifesto_2019.pdf',],['Aap_Manifesto_2019.pdf',]], title='Manifesto Analysis')
io.launch(debug=False,share=True)


#examples=[['/Bjp_Manifesto_2019.pdf',],['/Aap_Manifesto_2019.pdf',]],












