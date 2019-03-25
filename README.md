# hoax-detection

This is the code for the paper titled 'Early Detection of Hoaxes in Social Media Using Class-specific Embeddings'

Preparation
-----------

1. Download InferSent onto the InferSent/ directory from: https://github.com/facebookresearch/InferSent

2. Populate the final-data-24h/ and w2v-models/ directories with data from: https://figshare.com/articles/Twitter_Death_Hoaxes_dataset/5688811

   2.1. The file 'deathhoaxesdataset.tar.bz2' contains the tweet IDs, which can be used to retrieve the data to be put on the final-data-24h/ directory.

   2.2. The file 'word2vecrip.tar.bz2' contains the word2vec models which should be copied to the w2v-models/ directory.

Running
-------

To run the classifier:

python all.nonstruct.classification.py [features] [time] [window] [months]

where:
  [features] is the feature set, e.g. multiw2v
  [time] is the time in seconds used for the classification, i.e. the earliness of predictions
  [window] is the percentage of the data used as the sliding window, ranging from 0 to 1
  [months] is the number of months used for training, from 1 to 24
