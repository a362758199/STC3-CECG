#
#  web_json.py		Ver. 2019.6.23.
#	web service at port 35135 for pytorch_chatbot
#
from flask import Flask, request, render_template, jsonify, abort
import requests

import torch
import random
import pytorch_chatbot.main as pcm
import pytorch_chatbot.evaluate as pce
from pytorch_chatbot.train import indexesFromSentence
from pytorch_chatbot.load import loadPrepareData
from pytorch_chatbot.model import nn, EncoderRNN, LuongAttnDecoderRNN

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
          'topn' : 10,
          'query' : '一起 加油',
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
  res = requests.post('http://localhost:35135/api/translate', json=param0)
  json_dict = res.json()
  print('res={}'.format(res))

  if res.ok:
    print( res.json())
  else:
    print('error')

  return translate(json_dict['model'],json_dict['epoch'],json_dict['topn'],json_dict['query'])

###########################################
# api 
###########################################
# Input: {}
# Output: [{'model': 'model1', 'epoch': [epoch1, epoch2, ... epochn]}, {model': 'model2', 'epoch: [epoch1, epoch2, ... epochn]}, ..]
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

# Input: {'model': 'model1', 'epoch':epoch1, 'topn':topn, 'query': query1}
# Output: {'model': 'model1', 'epoch':epoch1, 'topn':topn, 'query': 'query1',
#	'filter': 'filter1', 'result':['response1', 'response2', ..., 'responsen']}
@app.route('/api/translate', methods=['GET','POST'])
def api_translate():
  #if not request.json:
  #  abort(400)
  content = request.get_json()
  print('request={}'.format(request))
  print('content={}'.format(content))

  model = request.json.get('model','translation2019_train_83k')
  epoch = request.json.get('epoch', 6000)
  topn = request.json.get('topn', 10)
  query = request.json.get('query', 'this is a book')

  epoch = int(epoch)
  topn = int(topn)

  #try:
    ## target = sentence_test_model(model,epoch,topn,query)
  en, de, voc = get_model(model,epoch)
  qfilter, target_list = predict_model(en, de, voc, topn, query)
  #except Exception as e:
  #  print('/api/translate: excpetion {}'.format(e))
  #  qfilter = 'unknown'
  #  target_list = ['internal error, retry a again']

  result_json = { 'model': model,
          'epoch':  epoch,
          'topn' : topn,
          'query' : query,
	  'filter' : qfilter,
          'result' : target_list
        }
  print('result_json={}'.format(result_json))

  return jsonify(result_json)
  

#######################################

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
id=1
torch.cuda.set_device(id)
print(torch.__version__)
Models = {}

#home_path = '/home/seke/jupyter/translate2019/'
home_path = './'

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=35135)
