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

def evaluate(id_preds, id_gts, outfile, window):
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
  with open(os.path.join('results-3way-' + str(window), outfile), 'w') as fw:
    print ''.ljust(30),
    fw.write(''.ljust(30))
    print 'precision'.ljust(15),
    fw.write('precision'.ljust(15))
    print 'recall'.ljust(15),
    fw.write('recall'.ljust(15))
    print 'f1'.ljust(15),
    fw.write('f1'.ljust(15))
    print 'n'.ljust(15)
    fw.write('n'.ljust(15) + '\n')
    for cat, acc in acc_by_class.iteritems():
      if cat != "0":
        catcount += 1
        print getcatname(cat).ljust(30),
        fw.write(getcatname(cat).ljust(30))

        microtp += acc['tp']
        microfp += acc['fp']
        microtn += acc['tn']
        microfn += acc['fn']

        p = 0
        if (acc['tp'] + acc['fp']) > 0:
          p = float(acc['tp']) / (acc['tp'] + acc['fp'])
        print str("%.3f" % p).ljust(15),
        fw.write(str("%.3f" % p).ljust(15))

        r = 0
        if (acc['tp'] + acc['fn']) > 0:
          r = float(acc['tp']) / (acc['tp'] + acc['fn'])
        print str("%.3f" % r).ljust(15),
        fw.write(str("%.3f" % r).ljust(15))

        f1 = 0
        if (p + r) > 0:
          f1 = 2 * p * r / (p + r)
        print str("%.3f" % f1).ljust(15),
        fw.write(str("%.3f" % f1).ljust(15))

        n = acc['tp'] + acc['fn']
        print str(n).ljust(15)
        fw.write(str(n).ljust(15) + '\n')

        npred = acc['tp'] + acc['fp']

        macro['p'] += p
        macro['r'] += r
        macro['f1'] += f1

        itemcount += n

    micro['p'] = float(microtp) / float(microtp + microfp)
    micro['r'] = float(microtp) / float(microtp + microfn)
    micro['f1'] = 2 * float(micro['p']) * micro['r'] / float(micro['p'] + micro['r'])

    print ""
    print 'Macroevaluation'.ljust(30),
    fw.write('\n')
    fw.write('Macroevaluation'.ljust(30))
    macrop = macro['p'] / catcount
    macror = macro['r'] / catcount
    macrof1 = macro['f1'] / catcount #2 * macrop * macror / (macrop + macror)
    print str("%.3f" % (macrop)).ljust(15),
    fw.write(str("%.3f" % (macrop)).ljust(15))
    print str("%.3f" % (macror)).ljust(15),
    fw.write(str("%.3f" % (macror)).ljust(15))
    print str("%.3f" % (macrof1)).ljust(15),
    fw.write(str("%.3f" % (macrof1)).ljust(15))
    print str(itemcount).ljust(15)
    fw.write(str(itemcount).ljust(15) + '\n')

    print 'Microevaluation'.ljust(30),
    fw.write('Microevaluation'.ljust(30))
    microp = micro['p']
    micror = micro['r']
    microf1 = micro['f1']
    print str("%.3f" % (microp)).ljust(15),
    fw.write(str("%.3f" % (microp)).ljust(15))
    print str("%.3f" % (micror)).ljust(15),
    fw.write(str("%.3f" % (micror)).ljust(15))
    print str("%.3f" % (microf1)).ljust(15),
    fw.write(str("%.3f" % (microf1)).ljust(15))
    print str(itemcount).ljust(15)
    fw.write(str(itemcount).ljust(15) + '\n')
