from flask import Flask, request, render_template, jsonify, abort
import requests

import torch
import numpy as np
import random
import pytorch_chatbot.main as pcm
import pytorch_chatbot.evaluate as pce
from emotional_classifier import five_emotional_classifier
from pytorch_chatbot.train import indexesFromSentence
from pytorch_chatbot.load import loadPrepareData
from pytorch_chatbot.model import nn, EncoderRNN, LuongAttnDecoderRNN
# from GenMethod import generate_method_ntcir
from GenMethod import generate_method_cgzy

########################
# To solve the following model loading error
#	result = unpickler.load()
#	ModuleNotFoundError: No module named 'load'
import sys
import pytorch_chatbot.load as load
if not 'load' in sys.modules:
  sys.modules['load'] = sys.modules['pytorch_chatbot.load']
############################

import subprocess
import json

def predictLoad(corpus, modelFile, n_layers=1, hidden_size=512):
  print('corpus={}\nmodelFile={}'.format(corpus,modelFile))

  torch.set_grad_enabled(False)
  voc, pairs = loadPrepareData(corpus)
  embedding = nn.Embedding(voc.n_words, hidden_size)
  encoder = EncoderRNN(voc.n_words, hidden_size, embedding, n_layers)
  attn_model = 'dot'
  decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.n_words, n_layers)

  checkpoint = torch.load(modelFile)
  encoder.load_state_dict(checkpoint['en'])
  decoder.load_state_dict(checkpoint['de'])

  # train mode set to false, effect only on dropout, batchNorm
  encoder.train(False)
  decoder.train(False)

  #try:
  encoder = encoder.to(device)
  decoder = decoder.to(device)
  #except:
  #  print('cannot get encoder/decoder')
  
  return encoder, decoder, voc

def predict(encoder, decoder, voc, question, top):
  result_list = []

  if(top==1):
    beam_size = 1
    output_words, _ = pce.evaluate(encoder, decoder, voc, question, beam_size)
    answer = ' '.join(output_words)
    answer = answer.replace('<EOS>','')
    result_list.append(answer)
    #print(output_words)
  else:
    beam_size = top
    output_words_list = pce.evaluate(encoder, decoder, voc, question, beam_size)
    count = 0;
    for output_words, score in output_words_list:
      count = count + 1
      if(count <= top):
        output_sentence = ' '.join(output_words)
        output_sentence = output_sentence.replace('<EOS>','')
        result_list.append(output_sentence)
        #print(" {:.3f} < {}".format(score, output_sentence))
  
  return result_list

def filter(voc, question):
  words = question.split()
  result = []
  for w in words:
    if(w in voc.word2index):
      result.append(w) 
  return ' '.join(result)

def load_all_classifier():
  # dict_text1, ecm1, dict_text2, ecm2, dict_text3, ecm3, dict_text4, ecm4, dict_text5, ecm5
  dict_text1, ecm1 = five_emotional_classifier.load_classifier('train_emotion1_neg.txt', 'train_emotion1_pos.txt',
                                                                 'emotion1_classification_model.h5')
  dict_text2, ecm2 = five_emotional_classifier.load_classifier('train_emotion2_neg.txt', 'train_emotion2_pos.txt',
                                                                 'emotion2_classification_model.h5')
  dict_text3, ecm3 = five_emotional_classifier.load_classifier('train_emotion3_neg.txt', 'train_emotion3_pos.txt',
                                                                 'emotion3_classification_model.h5')
  dict_text4, ecm4 = five_emotional_classifier.load_classifier('train_emotion4_neg.txt', 'train_emotion4_pos.txt',
                                                                 'emotion4_classification_model.h5')
  dict_text5, ecm5 = five_emotional_classifier.load_classifier('train_emotion5_neg.txt', 'train_emotion5_pos.txt',
                                                                 'emotion5_classification_model.h5')
  print('load classifiers successfully main')
  return  dict_text1, ecm1, dict_text2, ecm2, dict_text3, ecm3, dict_text4, ecm4, dict_text5, ecm5

'''
def count_emotion_score(text):
  #global emotion_score
  emotion_score = []
  emotion_score.append(five_emotional_classifier.predict(text, dict_text1, ecm1))
  emotion_score.append(five_emotional_classifier.predict(text, dict_text2, ecm2))
  emotion_score.append(five_emotional_classifier.predict(text, dict_text3, ecm3))
  emotion_score.append(five_emotional_classifier.predict(text, dict_text4, ecm4))
  emotion_score.append(five_emotional_classifier.predict(text, dict_text5, ecm5))
  # print(emotion_score, sep='\n')
  # label = emotion_score.index(max(emotion_score)) + 1
  # emotion_score = [float(f) for f in emotion_score]
  return emotion_score
'''
# -------------------------------
def sentence_test(voc,en,de,top,sentence):
    source = sentence.rstrip()
    seg_source = source
    fil_source = filter(voc, seg_source)
    target = predict(en, de, voc, fil_source, top)
    result = "\nsource: '%s'\nfilter: '%s'\n" % (seg_source,fil_source)
    
    for answer in target:
      result = result + "\t'%s'\n" % (answer)
    
    result = result + '\n'
    return result

def sentence_test_model(seg_corpus_name, iteration, top, sentence):
  en, de, voc = get_model(seg_corpus_name, iteration)
  return sentence_test(voc,en,de,top,sentence)

def get_model(seg_corpus_name, iteration):
  print('get_model: Models={}'.format(Models))
  model_key = '{}_{}'.format(seg_corpus_name,iteration)
  if model_key in Models:
    return Models[model_key]

  n_layers = 1
  hidden_size = 512
  modelFile = home_path + 'save/model/' + seg_corpus_name + '/1-1_512/' + str(iteration) + '_backup_bidir_model.tar'
  print('get_model: modelFile={}'.format(modelFile))
  en, de, voc = predictLoad(seg_corpus_name, modelFile, n_layers, hidden_size)
  #Models[model_key] = [en, de, voc]
  #return Models[model_key]
  return en, de, voc

def predict_model(en,de,voc,top,sentence):
    source = sentence.rstrip()
    seg_source = source
    fil_source = filter(voc, seg_source)
    target = predict(en, de, voc, fil_source, top)
    return fil_source, target

def file_test(voc,en,de,top,test_file_name):
  with open(test_file_name,"r") as f:
    jp_data = f.readlines()

  for i,source in enumerate(jp_data):
    source = source.rstrip()
    seg_source = source
    fil_source = filter(voc, seg_source)
    target = predict(en, de, voc, fil_source, top)
    print("%d:\nsource: '%s'\nfilter: '%s'" % (i+1,seg_source,fil_source))
    
    for answer in target:
      print("\t'%s'" % (answer))

def file_test_model(seg_corpus_name, iteration, top, test_file_name):
  n_layers = 1
  hidden_size = 512
  #iteration = 1200
  #top = 5
  modelFile = home_path + 'save/model/' + seg_corpus_name + '/1-1_512/' + str(iteration) + '_backup_bidir_model.tar'
  en, de, voc = predictLoad(seg_corpus_name, modelFile, n_layers, hidden_size)
  #print(modelFile)
  file_test(voc,en,de,top,test_file_name)
  
def print_voc(voc):
  print('tw+jp voc size=%d' % (len(voc.word2index)))
  print(voc.index2word)

def list_models(seg_corpus_name=''):
  if seg_corpus_name=='':
    modelPath = home_path + 'save/model/'
  else:
    modelPath = home_path + 'save/model/' + seg_corpus_name + '/1-1_512'

  out_bytes = subprocess.check_output(['ls','-l',modelPath],
                                    stderr=subprocess.STDOUT)
  out_text = out_bytes.decode('utf-8')
  return out_text

def get_model_list():
  modelPath = home_path + 'save/model/'

  out_bytes = subprocess.check_output(['ls','-l',modelPath],
                                    stderr=subprocess.STDOUT)
  out_text = out_bytes.decode('utf-8')

  model_list = []
  for line in out_text.split('\n'):
    fields = line.split()
    if len(fields) > 6:
      model_list.append(fields[-1])
  return model_list

def get_model_epoch_list(seg_corpus_name):
  modelPath = home_path + 'save/model/' + seg_corpus_name + '/1-1_512'

  out_bytes = subprocess.check_output(['ls','-l',modelPath],
                                    stderr=subprocess.STDOUT)
  out_text = out_bytes.decode('utf-8')

  model_list = []
  for line in out_text.split('\n'):
    fields = line.split()
    if len(fields) > 6:
      tar_field = fields[-1]
      tar_fields = tar_field.split('_')
      model_list.append(int(tar_fields[0]))
  return sorted(model_list, reverse=True)

def load_source(seg_corpus_name):
  path = home_path + 'data/' + seg_corpus_name + '.txt'
  
  with open(path) as inp:
    data = inp.readlines()

  print(len(data), len(data[0::2]), len(data[1::2]))

  data = { 'source': data[0::2], 'target': data[1::2] }
  return data


# --------------------------
app = Flask(__name__)

param0 = { 'model': 'ntcir_chinyi',
          'epoch':  60000,
          'topn' : 5,
          'query' : '好 漂亮 的 花',
          'result' : 'result area'
        }

@app.route('/')
def forms():
  return render_template('tl_json.html', param=param0)

@app.route('/translate/<model>/<int:epoch>/<int:topn>', methods=['GET', 'POST'])
def translate_long(model,epoch,topn):
  if request.method == 'POST':
    query = request.values['query']
  elif request.method == 'GET':
    query = request.args.get('query')

  return translate(model,epoch,topn,query)


@app.route('/translate', methods=['GET', 'POST'])
def translate_short():
  if request.method == 'POST':
    query = request.values['query']
    model = request.values['model']
    epoch = request.values['epoch']
    topn = request.values['topn']
  elif request.method == 'GET':
    query = request.args.get('query')
    model = request.args.get('model')
    epoch = request.args.get('epoch')
    topn = request.args.get('topn')

  return translate(model,epoch,topn,query)

def translate(model,epoch,topn,query):
  epoch = int(epoch)
  topn = int(topn)

  try:
    target = sentence_test_model(model,epoch,topn,query)
  except:
    target = 'internal error, retry a again'

  result = 'query="{}"\nresult="{}"\n'.format(query,target)
  
  param2 = { 'model': model,
          'epoch':  epoch,
          'topn' : topn,
          'query' : query,
          'result' : result
        }
  return render_template('tl_json.html', param=param2)

@app.route('/list/<model>')
def list_model(model):
  mlist = list_models(model)
  return '<pre>{}</pre>'.format(mlist)

@app.route('/list/')
def list():
  mlist = list_models()
  return '<pre>{}</pre>'.format(mlist)

@app.route('/react')
def hello():
  return render_template('react.html')

@app.route('/json')
def json():
  res = requests.post('http://localhost:30124/api/label_sentence', json={'query':'who are you'})
  json_dict = res.json()
  print('res={}'.format(res))

  if res.ok:
    print(res.json())
  else:
    print('error')

  return ' '
  #return translate(json_dict['model'],json_dict['epoch'],json_dict['topn'],json_dict['query'])


###########################################
# api 
###########################################
@app.route('/api/list_model', methods=['GET','POST'])
def api_list_model():
  mlist = get_model_list()
  result_json = []
  for model in mlist:
    elist = get_model_epoch_list(model)
    dict_elem = { 'model': model, 'epoch': elist}
    result_json.append(dict_elem)

  print('result_json={}'.format(result_json))
  return jsonify(result_json)

@app.route('/api/translate', methods=['GET','POST'])
def api_translate():
  #if not request.json:
  #  abort(400)
  content = request.get_json()
  print('request={}'.format(request))
  print('content={}'.format(content))
  '''
  model = request.json.get('model','ntcir_chinyi')
  epoch = request.json.get('epoch', 60000)
  topn = request.json.get('topn', 5)
  query = request.json.get('query', '好 漂亮 的 花')
  
  epoch = int(epoch)
  topn = int(topn)

  '''
  query = request.json.get('query', '好 漂亮 的 花')                        # CGZY
  print(query)
  input = generate_method_cgzy.cgzy_cm(str(query))
  model = input['model']
  epoch = input['epoch']
  topn = input['topn']
  # epoch = int(epoch)
  # topn = int(topn)
  print(input)                    #test

  #try:
    ## target = sentence_test_model(model,epoch,topn,query)
  en, de, voc = get_model(model,epoch)
  qfilter, target_list = predict_model(en, de, voc, topn, query)
  #except Exception as e:
  #  print('/api/translate: excpetion {}'.format(e))
  #  qfilter = 'unknown'
  # target_list = ['internal error, retry a again']

  # target_list = generate_method_ntcir.ntcir_ecs(target_list)                   # NTCIR emotion classification subsystem

  result_json = { 'model': model,
          'epoch':  epoch,
          'topn' : topn,
          'query' : query,
          'filter' : qfilter,
          'result' : target_list
        }
  print('result_json={}'.format(result_json))

  return jsonify(result_json)

'''
# emotion label
@app.route('/api/label_sentence', methods=['GET', 'POST'])
def api_label_sentence():
  #load_all_classifier()
  content = request.get_json()
  print('request={}'.format(request))
  print('content={}'.format(content))
  # emotion_label = request.json.get('emotional_label', 0)
  # emotion_label = int(emotion_label)

  text = request.json.get('query', '好 漂亮 的 花')
  prob, label = count_emotion_score(text)
  result_json = {
    'emotional_label': label,
    'prob': prob
  }

  print('result_json={}'.format(result_json))
  return jsonify(result_json)
'''
#######################################

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
id=1
torch.cuda.set_device(id)
print(torch.__version__)
Models = {}
# global dict_text1, ecm1, dict_text2, ecm2, dict_text3, ecm3, dict_text4, ecm4, dict_text5, ecm5     #
'''
dict_text1, ecm1, \
dict_text2, ecm2, \
dict_text3, ecm3, \
dict_text4, ecm4, \
dict_text5, ecm5 = load_all_classifier()
'''
print(generate_method_cgzy.cgzy_cm('好 漂亮 的 花' ))
#emolist, emolabel=count_emotion_score('who are you')
#print(emolist ,emolabel)
#home_path = '/home/seke/jupyter/translate2019/'
home_path = '/home/caoguangah/PycharmProjects/untitled/venv/'

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=30124)
