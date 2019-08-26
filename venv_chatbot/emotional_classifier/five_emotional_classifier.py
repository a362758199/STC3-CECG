#import os
# from google.colab import drive
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import load_model

modelPath = '/home/caoguangah/PycharmProjects/untitled/venv/emotional_classifier/emotional_classifier_saved_model/'
dataPath = '/home/caoguangah/PycharmProjects/untitled/venv/emotional_classifier/emotional_classifier_saved_data/'

def load_classifier(data_neg,data_pos,modelname):
  #rebuild dictionary
  neg=pd.read_table(dataPath + data_neg, header=None)  #-----訓練資料neg----
  pos=pd.read_table(dataPath + data_pos, header=None)  #-----訓練資料pos----
  pn = pd.concat([pos,neg],ignore_index=True)
  max_document_length = 1000
  texts = [' '.join(x) for x in pn[0]]
  # 實體化分詞器物件，最大詞匯量30000
  tokenizer = Tokenizer(num_words=30000)
  # 傳訓練資料 建字典
  tokenizer.fit_on_texts(texts)
  sequences = tokenizer.texts_to_sequences(texts)
  # 把序列设定为1000的长度，超过1000的部分舍弃，不到1000则补0
  sequences = pad_sequences(sequences, maxlen=1000, padding='post')
  sequences = np.array(sequences)
  dict_text= tokenizer.word_index
  #reload model
  model_name = modelPath + modelname   #----------要载入模型的地址-------
  model = load_model(model_name)
  return(dict_text, model)
  #print('load successfully')


# 预测函数


def predict(text, dict_text, modelname):
    # 分詞
    #cw = list(jieba.cut(text))
    #不分詞
  cw = list(text)
  word_id = []
    # 詞轉換成編號
  for word in cw:
    try:
      temp = dict_text[word]
      word_id.append(temp)
    except:
      word_id.append(0)
  word_id = np.array(word_id)
  word_id = word_id[np.newaxis,:]
  sequences = pad_sequences(word_id, maxlen=1000, padding='post')
    #result = np.argmax(model.predict(sequences))#傳回最大數組的index
  print(type(dict_text),modelname)
  result = modelname.predict(sequences)
    #print('emotion1：',result[0][1],sep = '')
  return (result[0][1])
  que = predict(text)
  return(que)
  #print(que)

