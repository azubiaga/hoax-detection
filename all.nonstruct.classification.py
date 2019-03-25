#!/usr/bin/env python -W ignore::DeprecationWarning

import numpy as np

from itertools import chain
#import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics
import sklearn
import sys
from os import listdir
from os.path import isfile, join, exists
import json
import operator

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import featuregeneration
import evaluation

features = sys.argv[1]
maxtime = 0
if len(sys.argv) >= 3:
  maxtime = int(sys.argv[2])
window = 1.0
if len(sys.argv) >= 4:
  window = float(sys.argv[3])
trainmonths = 24
if len(sys.argv) >= 5:
  trainmonths = int(sys.argv[4])

print features + ' ' + str(maxtime) + ' ' + str(window) + ' ' + str(trainmonths)

classifs = ['maxent', 'randomforest', 'nb', 'svm']
classifiers = []

done = 0
for classifier in classifs:
  outfile = classifier + '_' + features + '_' + str(trainmonths) + '_' + str(maxtime)

  if exists(join('results-3way-' + str(window), outfile)):
    done += 1
  else:
    classifiers.append(classifier)

if done == len(classifs):
  print 'Done earlier!'
  sys.exit()

id_preds = {}
id_gts = {}
for classifier in classifiers:
  id_preds[classifier] = {}
  id_gts[classifier] = {}

model = {}
for fold in range(0, 10):
  with open(join('train', 'training-set.dat'), 'r') as f:
    trainids = [ line.split('\t')[0] for line in f.read().splitlines() if int(line.strip().split('\t')[1]) <= trainmonths ]

  with open(join('test', 'test-' + str(fold) + '.dat'), 'r') as f:
    testids = f.read().splitlines()

  try:
    for classifier in classifiers:
      model[classifier]
  except:
    fbow = {}
    train_users = []
    for userid in trainids:
      with open(join(userid.replace('final-data-24h', 'annotations'), 'annotation.json'), 'r') as fh:
        annotation = json.load(fh)
      if annotation['death'] == 'real':
        gt = 0
        userinstance = (userid, gt)
        train_users.append(userinstance)
      elif annotation['death'] == 'commemoration':
        gt = 1
        userinstance = (userid, gt)
        train_users.append(userinstance)
      elif annotation['death'] == 'fake':
        gt = 2
        userinstance = (userid, gt)
        train_users.append(userinstance)

  test_users = []
  for userid in testids:
    with open(join(userid.replace('final-data-24h', 'annotations'), 'annotation.json'), 'r') as fh:
      annotation = json.load(fh)
    if annotation['death'] == 'real':
      gt = 0
      userinstance = (userid, gt)
      test_users.append(userinstance)
    elif annotation['death'] == 'commemoration':
      gt = 1
      userinstance = (userid, gt)
      test_users.append(userinstance)
    elif annotation['death'] == 'fake':
      gt = 2
      userinstance = (userid, gt)
      test_users.append(userinstance)

  try:
    for classifier in classifiers:
      model[classifier]
  except:
    if features == 'w2v':
      X_train = [featuregeneration.user2w2vfeatures(user, maxtime, window) for user in train_users]
    if features == 'social':
      X_train = [featuregeneration.user2socialfeatures(user, maxtime, window) for user in train_users]
    if features == 'social_and_w2v':
      X_train = [featuregeneration.user2socialw2vfeatures(user, maxtime, window) for user in train_users]

    if features == 'multiw2v':
      X_train = [featuregeneration.user2multiw2vfeatures(user, maxtime, window) for user in train_users]
    if features == 'social_and_multiw2v':
      X_train = [featuregeneration.user2socialmultiw2vfeatures(user, maxtime, window) for user in train_users]

    if features == 'infersent':
      featuregeneration.loadInfersentModel(train_users + test_users, maxtime, window)
      X_train = [featuregeneration.user2infersentfeatures(user, maxtime, window) for user in train_users]
    if features == 'social_and_infersent':
      featuregeneration.loadInfersentModel(train_users + test_users, maxtime, window)
      X_train = [featuregeneration.user2socialinfersentfeatures(user, maxtime, window) for user in train_users]

    y_train = [user[1] for user in train_users]
    ids_train = [user[0] for user in train_users]

  if features == 'w2v':
    X_test = [featuregeneration.user2w2vfeatures(user, maxtime, window) for user in test_users]
  if features == 'social':
    X_test = [featuregeneration.user2socialfeatures(user, maxtime, window) for user in test_users]
  if features == 'social_and_w2v':
    X_test = [featuregeneration.user2socialw2vfeatures(user, maxtime, window) for user in test_users]

  if features == 'multiw2v':
    X_test = [featuregeneration.user2multiw2vfeatures(user, maxtime, window) for user in test_users]
  if features == 'social_and_multiw2v':
    X_test = [featuregeneration.user2socialmultiw2vfeatures(user, maxtime, window) for user in test_users]

  if features == 'infersent':
    X_test = [featuregeneration.user2infersentfeatures(user, maxtime, window) for user in test_users]
  if features == 'social_and_infersent':
    X_test = [featuregeneration.user2socialinfersentfeatures(user, maxtime, window) for user in test_users]

  y_test = [user[1] for user in test_users]
  ids_test = [user[0] for user in test_users]

  for classifier in classifiers:
    try:
      model[classifier]
    except:
      if classifier == "svm":
        model[classifier] = LinearSVC(class_weight='balanced')
      if classifier == "nb":
        model[classifier] = GaussianNB()
      if classifier == "randomforest":
        model[classifier] = RandomForestClassifier(class_weight='balanced')
      if classifier == "maxent":
        model[classifier] = LogisticRegression(class_weight='balanced')

      model[classifier].fit(X_train, y_train)

    y_pred = model[classifier].predict(X_test)

    acc = 0
    items = 0
    for k, tweetid in enumerate(ids_test):
      id_preds[classifier][tweetid] = y_pred[k]
      id_gts[classifier][tweetid] = y_test[k]

      items += 1
      if y_pred[k] == y_test[k]:
        acc += 1

      thread_gts = y_test[k]
      thread_preds = y_pred[k]

    if items > 0:
      outfile = classifier + '_' + features + '_' + str(trainmonths) + '_' + str(maxtime)
      print(outfile + ' - ' + str(fold) + ': ' + str(float(acc) / items))

for classifier in classifiers:
  outfile = classifier + '_' + features + '_' + str(trainmonths) + '_' + str(maxtime)
  with open(join('predictions-3way-' + str(window), outfile), 'w') as fw:
    for tweetid, gt in id_gts[classifier].iteritems():
      fw.write(str(tweetid) + '\t' + str(id_preds[classifier][tweetid]) + '\t' + str(gt) + '\n')

  evaluation.evaluate(id_preds[classifier], id_gts[classifier], outfile, window)
