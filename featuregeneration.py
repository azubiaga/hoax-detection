from itertools import chain
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics
import sklearn
import pprint
from collections import Counter
import sys
from os import listdir
from os.path import isfile, join, exists
import json
import re
import time
import gensim
import numpy as np
from datetime import datetime
import math
import pickle

from math import log
from scipy.stats import entropy

import torch
from InferSent.models import InferSent

modeldataset = ''

def loadInfersentModel(users, maxtime = 0, window = 1.0):
  global infersent

  V = 2
  MODEL_PATH = 'InferSent/encoder/infersent%s.pkl' % V
  params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048, 'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
  infersent = InferSent(params_model)
  infersent.load_state_dict(torch.load(MODEL_PATH))

  W2V_PATH = 'InferSent/dataset/fastText/crawl-300d-2M.vec'
  #W2V_PATH = 'InferSent/dataset/GloVe/glove.840B.300d.txt'
  infersent.set_w2v_path(W2V_PATH)

  alltweets = []
  for user in users:
    userid = user[0]
    gt = user[1]

    with open(join(userid, 'rip.json'), 'r') as f:
      firstdatetime = datetime.strptime(json.loads(f.readline())['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
                    
    with open(join(userid, 'rip.json'), 'r') as f:
      for t in f:
        try:
          utweet = json.loads(t)
          utweetdatetime = datetime.strptime(utweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
          if (utweetdatetime - firstdatetime).total_seconds() <= maxtime and (utweetdatetime - firstdatetime).total_seconds() >= math.floor(maxtime - maxtime * window) and (nort == 0 or not 'retweeted_status' in t):
            alltweets.append(re.sub(r'([^\s\w]|_)+', '', utweet['text'].lower()))
          elif (utweetdatetime - firstdatetime).total_seconds() > maxtime:
            break
        except:
          continue

  infersent.build_vocab(alltweets, tokenize=True)

def loadW2vModel(dataset = 'rip-w2v-model'):
  global model
  global index2word_set
  global num_features

  try:
    model
  except:
    print 'Loading \'' + dataset + '\' w2v model...'
    model = gensim.models.Word2Vec.load(join('w2v-models', dataset))
    index2word_set = set(model.wv.index2word)
    num_features = model.wv.syn0.shape[1]
    print 'done!'

def loadW2vModels(datasets = ['rip-commemoration', 'rip-fake', 'rip-real']):
  global models
  global index2word_sets
  global num_featuress

  try:
    models
  except:
    models = []
    index2word_sets = []
    num_featuress = []
    for key, dataset in enumerate(datasets):
      print 'Loading \'' + dataset + '\' w2v model...'
      models.append(gensim.models.Word2Vec.load(join('w2v-models', dataset)))
      index2word_sets.append(set(models[key].wv.index2word))
      num_featuress.append(models[key].wv.syn0.shape[1])
      print 'done!'

def makeFeatureVec(words, getsum = 0): # Function to average all of the word vectors in a given sentence, set getsum to 1 if sum wanted instead
    global model
    global index2word_set
    global num_features

    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.

    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])

    if nwords > 1.0 and getsum == 0:
        featureVec = np.divide(featureVec,nwords)

    return featureVec

def makeFeatureMultiVec(words, getsum = 0): # Function to average all of the word vectors in a given sentence, set getsum to 1 if sum wanted instead
    global models
    global index2word_sets
    global num_featuress

    featureVec = []
    for key, model in enumerate(models):
        featVec = np.zeros((num_featuress[key],),dtype="float32")
        nwords = 0.

        for word in words:
            if word in index2word_sets[key]:
                nwords = nwords + 1.
                featVec = np.add(featVec, models[key][word])

        if nwords > 1.0 and getsum == 0:
            featVec = np.divide(featVec,nwords)

        featureVec.append(featVec)

    return np.concatenate(featureVec)

def makeInfersentVecFromTweets(tweets, userid, getsum = 0):
    global infersent

    featureVec = np.zeros((4096,),dtype="float32")
    pages = int(math.ceil(float(len(tweets)) / 100000))
    for page in range(0, pages):
        start = 100000 * page
        end = 100000 * (page + 1)
        inferVecs = infersent.encode(tweets[start:end], tokenize=True)
        for inferVec in inferVecs:
            featureVec = np.add(featureVec, inferVec)

    featureVec = np.divide(featureVec, len(tweets))

    return featureVec

def makeFeatureVecFromTweets(tweets, getsum = 0):
    global model
    num_features = model.wv.syn0.shape[1]

    featureVec = np.zeros((num_features,),dtype="float32")

    for tweet in tweets:
        featureVec = np.add(featureVec, makeFeatureVec(tweet, getsum))

    if len(tweets) > 0:
        featureVec = np.divide(featureVec, len(tweets))

    return featureVec

def makeFeatureMultiVecFromTweets(tweets, getsum = 0):
    global num_featuress

    featureVec = np.zeros((sum(num_featuress),),dtype="float32")

    for tweet in tweets:
        featureVec = np.add(featureVec, makeFeatureMultiVec(tweet, getsum))

    if len(tweets) > 0:
        featureVec = np.divide(featureVec, len(tweets))

    return featureVec

def user2infersentfeatures(user, maxtime = 0, window = 1.0):
  userid = user[0]
  gt = user[1]

  features = []

  with open(join(userid, 'rip.json'), 'r') as f: 
    firstdatetime = datetime.strptime(json.loads(f.readline())['created_at'], '%a %b %d %H:%M:%S +0000 %Y')

  tweets = []                                                                                                 
  with open(join(userid, 'rip.json'), 'r') as f:                                                                                
    for t in f:                                                                                                               
      try:                                                                                                                      
        utweet = json.loads(t)                                                                                                          
        utweetdatetime = datetime.strptime(utweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
        if (utweetdatetime - firstdatetime).total_seconds() <= maxtime and (utweetdatetime - firstdatetime).total_seconds() >= math.floor(maxtime - maxtime * window):
          tweets.append(re.sub(r'([^\s\w]|_)+', '', utweet['text'].lower()))
        elif (utweetdatetime - firstdatetime).total_seconds() > maxtime:
          break                                                                                                         
      except:
        continue

  if len(tweets) > 0:
    InfersentFeatureVec = makeInfersentVecFromTweets(tweets, userid, 0)
    for infersentfeature in np.nditer(InfersentFeatureVec):
      features.append(infersentfeature)
  else:
    features = [0] * 4096

  return features

def user2w2vfeatures(user, maxtime = 0, window = 1.0):
  userid = user[0]
  gt = user[1]

  loadW2vModel('rip-w2v-model')

  features = []

  with open(join(userid, 'rip.json'), 'r') as f:
    firstdatetime = datetime.strptime(json.loads(f.readline())['created_at'], '%a %b %d %H:%M:%S +0000 %Y')

  tweets = []
  with open(join(userid, 'rip.json'), 'r') as f:
    for t in f:
      try:
        utweet = json.loads(t)
        utweetdatetime = datetime.strptime(utweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
        if (utweetdatetime - firstdatetime).total_seconds() <= maxtime and (utweetdatetime - firstdatetime).total_seconds() >= math.floor(maxtime - maxtime * window):
          tweets.append(re.sub(r'([^\s\w]|_)+', '', utweet['text'].lower()).split())
        elif (utweetdatetime - firstdatetime).total_seconds() > maxtime:
          break
      except:
        continue

  Word2VecFeatureVec = makeFeatureVecFromTweets(tweets, 0)
  for w2vfeature in np.nditer(Word2VecFeatureVec):
    features.append(w2vfeature)

  return features

def user2multiw2vfeatures(user, maxtime = 0, window = 1.0):
  userid = user[0]
  gt = user[1]

  loadW2vModels()

  features = []

  with open(join(userid, 'rip.json'), 'r') as f:
    firstdatetime = datetime.strptime(json.loads(f.readline())['created_at'], '%a %b %d %H:%M:%S +0000 %Y')

  tweets = []
  with open(join(userid, 'rip.json'), 'r') as f:
    for t in f:
      try:
        utweet = json.loads(t)
        utweetdatetime = datetime.strptime(utweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
        if (utweetdatetime - firstdatetime).total_seconds() <= maxtime and (utweetdatetime - firstdatetime).total_seconds() >= math.floor(maxtime - maxtime * window):
          tweets.append(re.sub(r'([^\s\w]|_)+', '', utweet['text'].lower()).split())
        elif (utweetdatetime - firstdatetime).total_seconds() > maxtime:
          break
      except:
        continue

  Word2VecFeatureVec = makeFeatureMultiVecFromTweets(tweets, 0)
  for w2vfeature in np.nditer(Word2VecFeatureVec):
    features.append(w2vfeature)

  return features

def user2socialfeatures(user, maxtime = 0, window = 1.0):
  userid = user[0]
  gt = user[1]

  features = []

  with open(join(userid, 'rip.json'), 'r') as f:
    firstdatetime = datetime.strptime(json.loads(f.readline())['created_at'], '%a %b %d %H:%M:%S +0000 %Y')

  tweetcount = 0
  users = []
  rtingusers = []
  tweetlength = 0
  retweetspertweet = {}
  replyratio = 0
  linkratio = 0
  questionratio = 0
  exclratio = 0
  picratio = 0
  vocabulary = {}
  hashtags = {}
  mentions = {}
  languages = []
  avguserreputation = 0
  avgrtinguserreputation = 0
  with open(join(userid, 'rip.json'), 'r') as f:
    for t in f:
      try:
        utweet = json.loads(t)
        utweetdatetime = datetime.strptime(utweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y')                                                                                 
        if ((utweetdatetime - firstdatetime).total_seconds() <= maxtime and (utweetdatetime - firstdatetime).total_seconds() >= math.floor(maxtime - maxtime * window)):
          tweetcount += 1

          tweetlength += len(utweet['text'])

          if 'retweeted_status' in utweet:
            rtid = utweet['retweeted_status']['id_str']
            retweetspertweet[rtid] = 1 + retweetspertweet.get(rtid, 0)
          else:
            tid = utweet['id_str']
            retweetspertweet[tid] = retweetspertweet.get(tid, 0)

          if 'in_reply_to_status_id_str' in utweet and utweet['in_reply_to_status_id_str'] != '':
            replyratio += 1

          if 'http' in utweet['text']:
            linkratio += 1

          if '?' in utweet['text']:
            questionratio += 1

          if '!' in utweet['text']:
            exclratio += 1

          if 'pic.twitter' in utweet['text'] or 'instagr.am' in utweet['text'] or 'instagram.com' in utweet['text']:
            picratio += 1

          tokens = re.sub(r'([^\s\w]|_)+', '', utweet['text']).split()
          for token in tokens:
            vocabulary[token] = 1 + vocabulary.get(token, 0)

          if 'entities' in utweet and 'hashtags' in utweet['entities']:
            for hashtag in utweet['entities']['hashtags']:
              ht = hashtag['text']
              hashtags[ht] = 1 + hashtags.get(ht, 0)

          if 'entities' in utweet and 'user_mentions' in utweet['entities']:
            for mention in utweet['entities']['user_mentions']:
              ment = mention['screen_name']
              mentions[ment] = 1 + mentions.get(ment, 0)

          if 'lang' in utweet and utweet['lang'] != '' and utweet['lang'] != 'und' and not utweet['lang'] in languages:
            languages.append(utweet['lang'])

          if not 'retweeted_status' in utweet:
            user = utweet['user']['screen_name']
            if not user in users:
              users.append(user)
              followratio = len(str(utweet['user']['friends_count'])) / float(len(str(utweet['user']['followers_count'])))
              avguserreputation += followratio

          if 'retweeted_status' in utweet:
            user = utweet['user']['screen_name']
            if not user in rtingusers:
              rtingusers.append(user)
              followratio = len(str(utweet['user']['friends_count'])) / float(len(str(utweet['user']['followers_count'])))
              avgrtinguserreputation += followratio
        elif (utweetdatetime - firstdatetime).total_seconds() > maxtime:
          break
      except:
        continue

  if tweetcount == 0:
    tweetcount = 1
  diffusers = len(users) / float(tweetcount)
  diffrtingusers = len(rtingusers) / float(tweetcount)
  tweetlength /= float(tweetcount) * 140

  rtspertweet = 0
  for tid, retweets in retweetspertweet.iteritems():
    rtspertweet += retweets
  rtspertweetcount = len(retweetspertweet)
  if rtspertweetcount == 0:
    rtspertweetcount = 1
  rtspertweet /= float(rtspertweetcount)

  replyratio /= float(tweetcount)
  if maxtime > 0:
    tweetingrate = tweetcount / float(maxtime)
  else:
    tweetingrate = tweetcount
  linkratio /= float(tweetcount)
  questionratio /= float(tweetcount)
  exclratio /= float(tweetcount)
  picratio /= float(tweetcount)

  tokenspertweet = len(vocabulary) / float(tweetcount) # vocabularity diversity / unique tokens per tweet
  hashtagspertweet = len(hashtags) / float(tweetcount) # hashtag diversity
  mentionspertweet = len(mentions) / float(tweetcount) # user mention diversity

  langcount = len(languages)

  usercount = len(users)
  if usercount == 0:
    usercount = 1
  avguserreputation /= float(usercount)

  if len(rtingusers) > 0:
    avgrtinguserreputation /= float(len(rtingusers))
  else:
    avgrtinguserreputation = 0 

  features = [diffusers, diffrtingusers, tweetlength, rtspertweet, replyratio, tweetingrate, linkratio, questionratio, exclratio, picratio, tokenspertweet, hashtagspertweet, mentionspertweet, langcount, avguserreputation, avgrtinguserreputation]

  return features

def user2socialinfersentfeatures(user, maxtime = 0, window = 1.0):
  features = user2infersentfeatures(user, maxtime = maxtime, window = window)

  for feature in user2socialfeatures(user, maxtime = maxtime, window = window):
    features.append(feature)

  return features

def user2socialw2vfeatures(user, maxtime = 0, window = 1.0):
  features = user2w2vfeatures(user, maxtime = maxtime, window = window)

  for feature in user2socialfeatures(user, maxtime = maxtime, window = window):
    features.append(feature)

  return features

def user2socialmultiw2vfeatures(user, maxtime = 0, window = 1.0):
  features = user2multiw2vfeatures(user, maxtime = maxtime, window = window)

  for feature in user2socialfeatures(user, maxtime = maxtime, window = window):
    features.append(feature)

  return features
