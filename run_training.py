r"""Executable for training Word2Vec models 

Example:
  python run_training.py \
    --filenames=/PATH/TO/FILE/file1.txt,/PATH/TO/FILE/file2.txt
    --out_dir=/PATH/TO/OUT_DIR/
    --batch_size=64
    --window_size=5
"""
import os
from datetime import datetime

import tensorflow as tf
import numpy as np

from dataset import Word2VecDataset
from word2vec import Word2VecModel

flags = tf.app.flags

flags.DEFINE_string('arch', 'skip_gram', 'Architecture (skip_gram or cbow).')
flags.DEFINE_string('algm', 'negative_sampling', 'Training algorithm '
    '(negative_sampling or hierarchical_softmax).')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_integer('max_vocab_size', 0, 'Maximum vocabulary size. If > 0, '
    'the top `max_vocab_size` most frequent words are kept in vocabulary.')
flags.DEFINE_integer('min_count', 10, 'Words whose counts < `min_count` are not'
    ' included in the vocabulary.')
flags.DEFINE_float('sample', 1e-3, 'Subsampling rate.')
flags.DEFINE_integer('window_size', 10, 'Num of words on the left or right side' 
    ' of target word within a window.')

flags.DEFINE_integer('embed_size', 300, 'Length of word vector.')
flags.DEFINE_integer('negatives', 5, 'Num of negative words to sample.')
flags.DEFINE_float('power', 0.75, 'Distortion for negative sampling.')
flags.DEFINE_float('alpha', 0.025, 'Initial learning rate.')
flags.DEFINE_float('min_alpha', 0.0001, 'Final learning rate.')
flags.DEFINE_boolean('add_bias', True, 'Whether to add bias term to dotproduct '
    'between syn0 and syn1 vectors.')

flags.DEFINE_integer('num_epochs', 1, 'Num of epochs to iterate training data.')
flags.DEFINE_integer('num_steps', 0, '(Optional) Num of steps to train model '
    'for. Defaults to 0. If > 0, use `num_epochs` to compute an estimated num '
    'of steps.')
flags.DEFINE_integer('log_per_steps', 10000, 'Every `log_per_steps` steps to '
    ' output logs.')
flags.DEFINE_list('filenames', [], 'Names of comma-separated input text files.')
flags.DEFINE_string('out_dir', '', 'Output directory.')

FLAGS = flags.FLAGS


def main(_):
  assert FLAGS.filenames, '`filenames` is missing.'
  assert FLAGS.out_dir, '`out_dir` is missing.'
  dataset = Word2VecDataset(arch=FLAGS.arch,
                            algm=FLAGS.algm,
                            batch_size=FLAGS.batch_size,
                            max_vocab_size=FLAGS.max_vocab_size,
                            min_count=FLAGS.min_count,
                            sample=FLAGS.sample,
                            window_size=FLAGS.window_size,
                            shuffle_buffer_size=None)
  dataset.build_vocab(FLAGS.filenames)

  num_steps = FLAGS.num_steps
  if num_steps <= 0:
    num_steps = dataset.estimate_num_steps(FLAGS.num_epochs)
    print('Use estimated num steps computed from `num_epochs`:', num_steps)

  word2vec = Word2VecModel(arch=FLAGS.arch,
                           algm=FLAGS.algm,
                           embed_size=FLAGS.embed_size,
                           batch_size=FLAGS.batch_size,
                           negatives=FLAGS.negatives,
                           power=FLAGS.power,
                           alpha=FLAGS.alpha,
                           min_alpha=FLAGS.min_alpha,
                           num_steps=num_steps,
                           add_bias=FLAGS.add_bias,
                           random_seed=0)

  to_be_run_dict = word2vec.train(dataset, FLAGS.filenames)

  with tf.Session() as sess:
    sess.run(dataset.iterator_initializer)
    sess.run(tf.tables_initializer())
    sess.run(tf.global_variables_initializer())

    average_loss = 0.

    for step in range(num_steps):
      result_dict = sess.run(to_be_run_dict)
      average_loss += result_dict['loss'].mean()
      if step % FLAGS.log_per_steps == 0:
        if step > 0:
          average_loss /= FLAGS.log_per_steps
        print('step:', step, 'average_loss:', average_loss, 
            'learning_rate:', result_dict['learning_rate'])
        average_loss = 0.

    syn0_final = sess.run(word2vec.syn0)

  np.save(os.path.join(FLAGS.out_dir, 'embed'), syn0_final)
  with open(os.path.join(FLAGS.out_dir, 'vocab.txt'), 'w') as fid:
    for w in dataset.table_words:
      fid.write(w + '\n')
  print('Word embeddings saved to', os.path.join(FLAGS.out_dir, 'embed.npy'))
  print('Vocabulary saved to', os.path.join(FLAGS.out_dir, 'vocab.txt'))

if __name__ == '__main__':
  tf.app.run()
