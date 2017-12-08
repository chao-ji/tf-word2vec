import pickle
import itertools
import numpy as np
from wv import Word2Vec
import tensorflow as tf
from datetime import datetime
from scipy.special import expit

sents = pickle.load(open("sents.pickle"))

sess = tf.InteractiveSession()
word2vec = Word2Vec(epochs=1,
                    start_alpha=0.025,
                    end_alpha=0.0001,
                    max_batch_size=256,
                    embedding_size=300,
                    num_neg_samples=5,
                    opts=[False, True, True, False],
                    log_every_n_steps=1000,
                    window=10,
                    min_word_count=10)


t0 = datetime.now()
print str(t0)
embeddings_final, weights_final = word2vec.train(sents, sess)
# Time Elapsed 
print str(datetime.now() - t0)
