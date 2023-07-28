import numpy as np
import re, string
from cogworks_data.language import get_data_path

from pathlib import Path
import json
from gensim.models import KeyedVectors


def cocodata():
  # load COCO metadata
  filename = get_data_path("captions_train2014.json")
  with Path(filename).open() as f:
    coco_data = json.load(f)
  return coco_data

def Glove():

  filename = "glove.6B.200d.txt.w2v"
  
  # this takes a while to load -- keep this in mind when designing your capstone project
  glove = KeyedVectors.load_word2vec_format(get_data_path(filename), binary=False)
  return glove


def process(text):
    punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))
    return punc_regex.sub('', text).lower().split()



def vocab_docs():
    coco_data = cocodata()
    docList = []
    vocabList = set()
    for i in range(len(coco_data["annotations"])):
        caption = process(coco_data["annotations"][i]["caption"])
        docList.append(set(caption))
        vocabList.update(caption)
    return list(vocabList), docList



def IDF():
    vocab, docs = vocab_docs()
    N = len(docs)
    nt = np.zeros(len(vocab))
    index = {vocab[i] : i for i in range(len(vocab))}
    #414,113 captions
    for doc in docs:
        #~10-15 words
        for words in doc:
            #O(1) since vocabSet is a set
            if words in index:
                #O(1) since using dict to access index
                nt[index[words]] += 1
    values = np.log10(N / nt)
    return {vocab[i] : values[i] for i in range(len(vocab))}



def embed(text, idf, glove):
    w = np.zeros(200)
    text = process(text)
    for word in text:
        if word in idf and word in glove:
            w+= idf[word]*glove[word]
    return w/np.abs(w)

