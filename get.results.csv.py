from __future__ import print_function
import sys
from pprint import pprint
from os.path import exists

classifiers = ['maxent', 'nb', 'randomforest', 'svm']

with open('timevalues.txt', 'r') as fh:
  timeframes = [ v for v in fh.read().splitlines() if int(v) <= 18000 ]

with open('featurevalues.txt', 'r') as fh:
  features = fh.read().splitlines()

windows = ['0.1', '0.25', '0.5', '0.75', '1.0']
months = ['3', '6', '9', '12', '15', '18', '21', '24']

donecount = 0
allcount = 0
for window in windows:
  for month in months:
    with open('csv-results/results-' + window + '-' + month + '.csv', 'wb') as fw:
      for classifier in classifiers:
        for feature in features:
          fw.write(classifier + ',' + feature + ',')
          for timeframe in timeframes:
            allcount += 1
            if exists('results-3way-' + window + '/' + classifier + '_' + feature + '_' + month + '_' + timeframe):
              donecount += 1
              with open('results-3way-' + window + '/' + classifier + '_' + feature + '_' + month + '_' + timeframe, 'r') as fh:
                lines = fh.read().splitlines()
#                print('results-3way-' + window + '/' + classifier + '_' + feature + '_' + month + '_' + timeframe)
                if len(lines) < 6:
                  print('results-3way-' + window + '/' + classifier + '_' + feature + '_' + month + '_' + timeframe)
                scores = lines[-2].split(' ')
                fw.write(scores[35] + ',')
            else:
              fw.write('0.0,')
          fw.write('\n')
          print(str(donecount) + '/' + str(allcount), end='\r')
          sys.stdout.flush()

print(str(donecount) + '/' + str(allcount), end='\n')
