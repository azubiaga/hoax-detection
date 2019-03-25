from __future__ import print_function
import sys
from pprint import pprint
from os.path import exists
import evaluation2

classifiers = ['maxent']#, 'nb', 'randomforest', 'svm']

with open('timevalues.txt', 'r') as fh:
  timeframes = [ v for v in fh.read().splitlines() if int(v) <= 18000 ]

with open('featurevalues.txt', 'r') as fh:
  features = fh.read().splitlines()

windows = ['0.1', '0.25', '0.5', '0.75', '1.0']
months = ['3', '6', '9', '12', '15', '18', '21', '24']
nort = ['_nort', '']
add = ['_add', '']

allresults = {}
donecount = 0
allcount = 0
for window in windows:
  if not window in allresults:
    allresults[window] = {}
  for month in months:
    if not month in allresults[window]:
      allresults[window][month] = {}
    for classifier in classifiers:
      if not classifier in allresults[window][month]:
        allresults[window][month][classifier] = {}
      for feature in features:
        if not feature in allresults[window][month][classifier]:
          allresults[window][month][classifier][feature] = {}
        for timeframe in timeframes:
          if not timeframe in allresults[window][month][classifier][feature]:
            allresults[window][month][classifier][feature][timeframe] = {}
          for nrt in nort:
            if feature != 'social' or nrt != '_nort':
              if not nrt in allresults[window][month][classifier][feature][timeframe]:
                allresults[window][month][classifier][feature][timeframe][nrt] = {}
              for ad in add:
                if feature != 'social' or ad != '_add':
                  if not ad in allresults[window][month][classifier][feature][timeframe][nrt]:
                    allresults[window][month][classifier][feature][timeframe][nrt][ad] = {}

                  allcount += 1
                  if exists('predictions-3way-' + window + '/' + classifier + '_' + feature + nrt + ad + '_' + month + '_' + timeframe):
                    donecount += 1
                    id_preds = {}
                    id_gts = {}
                    with open('predictions-3way-' + window + '/' + classifier + '_' + feature + nrt + ad + '_' + month + '_' + timeframe, 'r') as fh:
                      for line in fh:
                        values = line.strip().split('\t')
                        id_preds[values[0]] = values[1]
                        id_gts[values[0]] = values[2]

                    classf1s, macrof1, microf1 = evaluation2.evaluate(id_preds, id_gts)

                    allresults[window][month][classifier][feature][timeframe][nrt][ad]['0'] = classf1s.get('0', 0)
                    allresults[window][month][classifier][feature][timeframe][nrt][ad]['1'] = classf1s.get('1', 0)
                    allresults[window][month][classifier][feature][timeframe][nrt][ad]['2'] = classf1s.get('2', 0)
                    allresults[window][month][classifier][feature][timeframe][nrt][ad]['macrof1'] = macrof1
                    allresults[window][month][classifier][feature][timeframe][nrt][ad]['microf1'] = microf1
#                  else:
#                    print(str(donecount) + '/' + str(allcount), end='\n')
#                    print('predictions-3way-' + window + '/' + classifier + '_' + feature + nrt + ad + '_' + month + '_' + timeframe, end='\n')
#                    print('error.')
                print(str(donecount) + '/' + str(allcount), end='\r')
                sys.stdout.flush()

print(str(donecount) + '/' + str(allcount), end='\n')

print('')

for resulttype in ['0', '1', '2', 'microf1', 'macrof1']:
  print(resulttype + '\n')

  with open('final-results-csv/01-feature-comparison-' + resulttype + '.csv', 'wb') as fw:
    print(',', end='')
    fw.write(',')
    for timeframe in timeframes:
      print(timeframe + ',', end='')
      fw.write(timeframe + ',')
    print('')
    fw.write('\n')
    for feature in features:
      print(feature + ',', end='')
      fw.write(feature + ',')
      for timeframe in timeframes:
        try:
          print(('%.3f' % allresults['1.0']['24']['maxent'][feature][timeframe][''][''][resulttype]) + ',', end='')
          fw.write('%.3f' % allresults['1.0']['24']['maxent'][feature][timeframe][''][''][resulttype] + ',')
        except:
          print('?.???', end=',')
      print('')
      fw.write(',\n')

  print('')
  print('')

  with open('final-results-csv/02-window-comparison-' + resulttype + '.csv', 'wb') as fw:
#    print(',', end='')
    fw.write(',')
    for timeframe in timeframes:
#      print(timeframe + ',', end='')
      fw.write(timeframe + ',')
#    print('')
    fw.write('\n')
    for window in windows:
#      print(window + ',', end='')
      fw.write(window + ',')
      for timeframe in timeframes:
        try:
#          print(('%.3f' % allresults[window]['24']['maxent']['social_and_multiw2v'][timeframe][''][''][resulttype]) + ',', end='')
          fw.write('%.3f' % allresults[window]['24']['maxent']['social_and_multiw2v'][timeframe][''][''][resulttype] + ',')
        except:
#          print('?.???', end='')
          fw.write('?.???')
#      print('')
      fw.write(',\n')

#  print('')
#  print('')

  with open('final-results-csv/03-month-comparison-' + resulttype + '.csv', 'wb') as fw:
#    print(',', end='')
    fw.write(',')
    for timeframe in timeframes:
#      print(timeframe + ',', end='')
      fw.write(timeframe + ',')
#    print('')
    fw.write('\n')
    for month in months:
#      print(month + ',', end='')
      fw.write(month + ',')
      for timeframe in timeframes:
        try:
#          print(('%.3f' % allresults['1.0'][month]['maxent']['social_and_multiw2v'][timeframe][''][''][resulttype]) + ',', end='')
          fw.write('%.3f' % allresults['1.0'][month]['maxent']['social_and_multiw2v'][timeframe][''][''][resulttype] + ',')
        except:
#          print('?.???', end='')
          fw.write('?.???')
#      print('')
      fw.write(',\n')
#
#  print('')
#  print('')
