import os

def getcatname(cat):
  if cat == 0:
    return 'real'
  if cat == 1:
    return 'commemoration'
  if cat == 2:
    return 'fake'
  if cat == 3:
    return 'non-death'
  return str(cat)

def evaluate(id_preds, id_gts):
  allitems = 0
  acc_by_class = {}
  for tweetid, gt in id_gts.iteritems():
    allitems += 1
    if not gt in acc_by_class:
      acc_by_class[gt] = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

  ok = 0
  for tweetid, gt in id_gts.iteritems():
    pred = id_preds[tweetid]
    if gt == pred:
      ok += 1
      acc_by_class[gt]['tp'] += 1
    else:
      acc_by_class[gt]['fn'] += 1
      acc_by_class[pred]['fp'] += 1

  catcount = 0
  itemcount = 0
  macro = {'p': 0, 'r': 0, 'f1': 0}
  micro = {'p': 0, 'r': 0, 'f1': 0}

  microtp = 0
  microfp = 0
  microtn = 0
  microfn = 0
  classf1s = {}
  for cat, acc in acc_by_class.iteritems():
    catcount += 1

    microtp += acc['tp']
    microfp += acc['fp']
    microtn += acc['tn']
    microfn += acc['fn']

    p = 0
    if (acc['tp'] + acc['fp']) > 0:
      p = float(acc['tp']) / (acc['tp'] + acc['fp'])

    r = 0
    if (acc['tp'] + acc['fn']) > 0:
      r = float(acc['tp']) / (acc['tp'] + acc['fn'])

    f1 = 0
    if (p + r) > 0:
      f1 = 2 * p * r / (p + r)

    classf1s[cat] = f1

    n = acc['tp'] + acc['fn']

    npred = acc['tp'] + acc['fp']

    macro['p'] += p
    macro['r'] += r
    macro['f1'] += f1

    itemcount += n

  try:
    micro['p'] = float(microtp) / float(microtp + microfp)
  except:
    micro['p'] = 0.0
  try:
    micro['r'] = float(microtp) / float(microtp + microfn)
  except:
    micro['r'] = 0.0
  try:
    micro['f1'] = 2 * float(micro['p']) * micro['r'] / float(micro['p'] + micro['r'])
  except:
    micro['f1'] = 0.0

  catcount = 3

  macrop = macro['p'] / catcount
  macror = macro['r'] / catcount
  macrof1 = macro['f1'] / catcount #2 * macrop * macror / (macrop + macror)

  microp = micro['p']
  micror = micro['r']
  microf1 = micro['f1']

  return classf1s, macrof1, microf1
