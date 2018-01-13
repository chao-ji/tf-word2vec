import pickle
import itertools
import numpy as np
from word2vec import Word2Vec
from word2vec import WordVectors
import tensorflow as tf
from datetime import datetime

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("min_count", 10,            "Minimum word count to be considered as a vocabulary word")
tf.app.flags.DEFINE_integer("window", 10,               "Maximum number of words on either side of a window")
tf.app.flags.DEFINE_integer("size", 300,                "Embedding size")
tf.app.flags.DEFINE_integer("max_batch_size", 256,      "Maximum mini-batch size")
tf.app.flags.DEFINE_integer("epochs", 1,                "Number of runs over the entire corpus")
tf.app.flags.DEFINE_integer("log_every_n_steps", 10000, "Prints out average loss in every N steps (mini-batches)")
tf.app.flags.DEFINE_bool("hidden_layer_toggle", True,   "`True` for `sg`, `False` for `cbow`")
tf.app.flags.DEFINE_bool("output_layer_toggle", True,   "`True` for `ns`, `False` for `hs`")

def main(argv=None):
  FLAGS = tf.app.flags.FLAGS
  sents = pickle.load(open("sents.pickle"))
  sess = tf.InteractiveSession()
  
  word2vec = Word2Vec(**FLAGS.__dict__["__flags"])

  t0 = datetime.now()
  print str(t0)
  word_vectors = word2vec.train(sents, sess)
  # Time Elapsed 
  print str(datetime.now() - t0)

if __name__ == "__main__":
  tf.app.run()
  
