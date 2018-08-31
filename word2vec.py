import heapq

import numpy as np
import tensorflow as tf


class Word2VecModel(object):
  """Word2VecModel.
  """

  def __init__(self, arch, algm, embed_size, batch_size, negatives, power,
               alpha, min_alpha, num_steps, add_bias, random_seed):
    """Constructor.

    Args:
      arch: string scalar, architecture ('skip_gram' or 'cbow').
      algm: string scalar, training algorithm ('negative_sampling' or
        'hierarchical_softmax').
      embed_size: int scalar, length of word vector.
      batch_size: int scalar, batch size.
      negatives: int scalar, num of negative words to sample.
      power: float scalar, distortion for negative sampling. 
      alpha: float scalar, initial learning rate.
      min_alpha: float scalar, final learning rate.
      num_steps: int scalar, num of steps to train model for.
      add_bias: bool scalar, whether to add bias term to dotproduct 
        between syn0 and syn1 vectors.
      random_seed: int scalar, random_seed.
    """
    self._arch = arch
    self._algm = algm
    self._embed_size = embed_size
    self._batch_size = batch_size
    self._negatives = negatives
    self._power = power
    self._alpha = alpha
    self._min_alpha = min_alpha
    self._num_steps = num_steps
    self._add_bias = add_bias
    self._random_seed = random_seed

    self._syn0 = None

  def _build_loss(self, dataset, filenames, scope=None):
    """Builds the graph that leads from data tensors (`inputs`, `labels`)
    to loss. Has the side effect of setting attribute `syn0`.

    Args:
      dataset: a `Word2VecDataset` instance.
      filenames: a list of strings, holding names of text files.
      scope: string scalar, scope name.

    Returns:
      loss: float tensor, cross entropy loss. 
    """
    tensor_dict = dataset.get_tensor_dict(filenames)
    inputs, labels = tensor_dict['inputs'], tensor_dict['labels']

    syn0, syn1, biases = self._create_embeddings(dataset.table_words)
    self._syn0 = syn0
    with tf.variable_scope(scope, 'Loss', [inputs, labels, syn0, syn1, biases]):
      if self._algm == 'negative_sampling':
        loss = self._negative_sampling_loss(
            dataset.unigram_counts, inputs, labels, syn0, syn1, biases)
      elif self._algm == 'hierarchical_softmax':
        loss = self._hierarchical_softmax_loss(
            inputs, labels, syn0, syn1, biases)
      return loss

  def train(self, dataset, filenames):
    """Adds training related ops to the graph.

    Args:
      dataset: a `Word2VecDataset` instance.
      filenames: a list of strings, holding names of text files.

    Returns: 
      to_be_run_dict: dict mapping from names to tensors/operations, holding
        the following entries:
        { 'grad_update_op': optimization ops,
          'loss': cross entropy loss,
          'learning_rate': float-scalar learning rate}
    """
    loss = self._build_loss(dataset, filenames)
    global_step = tf.train.get_or_create_global_step()
    progress = tf.to_float(global_step) / float(self._num_steps)
    learning_rate = tf.maximum(
        self._alpha * (1 - progress) + self._min_alpha * progress,
        self._min_alpha)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grad_update_op = optimizer.minimize(loss, global_step=global_step)
    to_be_run_dict = {'grad_update_op': grad_update_op, 
                      'loss': loss, 
                      'learning_rate': learning_rate}
    return to_be_run_dict

  def _create_embeddings(self, table_words, scope=None):
    """Creates initial word embedding variables.

    Args:
      table_words: list of string, holding the list of vocabulary words. Index
        of each entry is the same as the word index into the vocabulary.
      scope: string scalar, scope name.

    Returns:
      syn0: float tensor of shape [vocab_size, embed_size], input word 
        embeddings (i.e. weights of hidden layer).
      syn1: float tensor of shape [syn1_rows, embed_size], output word
        embeddings (i.e. weights of output layer).
      biases: float tensor of shape [syn1_rows], biases added onto the logits.
    """
    if self._algm == 'negative_sampling':
      syn1_rows = len(table_words) 
    else:
      syn1_rows = len(table_words) - 1
    syn0_init_val = get_syn0_init_val(table_words,
                                      self._embed_size,
                                      self._random_seed)
    with tf.variable_scope(scope, 'Embedding'):
      syn0 = tf.get_variable('syn0', initializer=syn0_init_val, 
          dtype=tf.float32)
      syn1 = tf.get_variable('syn1', initializer=tf.random_uniform([syn1_rows,
          self._embed_size], -0.1, 0.1), dtype=tf.float32)
      biases = tf.get_variable('biases', initializer=tf.zeros([syn1_rows]),
          dtype=tf.float32)
    return syn0, syn1, biases

  def _negative_sampling_loss(
      self, unigram_counts, inputs, labels, syn0, syn1, biases):
    """Builds the loss for negative sampling.

    Args:
      unigram_counts: list of int, holding word counts. Index of each entry
        is the same as the word index into the vocabulary.
      inputs: int tensor of shape [batch_size] (skip_gram) or 
        [batch_size, 2*window_size+1] (cbow)
      labels: int tensor of shape [batch_size]
      syn0: float tensor of shape [vocab_size, embed_size], input word 
        embeddings (i.e. weights of hidden layer).
      syn1: float tensor of shape [syn1_rows, embed_size], output word
        embeddings (i.e. weights of output layer).
      biases: float tensor of shape [syn1_rows], biases added onto the logits.

    Returns:
      loss: float tensor of shape [batch_size, sample_size + 1].
    """
    sampled_values = tf.nn.fixed_unigram_candidate_sampler(
        true_classes=tf.expand_dims(labels, 1),
        num_true=1,
        num_sampled=self._batch_size*self._negatives,
        unique=True,
        range_max=len(unigram_counts),
        distortion=self._power,
        unigrams=unigram_counts)
 
    sampled = sampled_values.sampled_candidates
    sampled_mat = tf.reshape(sampled, [self._batch_size, self._negatives])
    inputs_syn0 = self._get_inputs_syn0(syn0, inputs) # [N, D]
    true_syn1 = tf.gather(syn1, labels) # [N, D]
    sampled_syn1 = tf.gather(syn1, sampled_mat) # [N, K, D]
    true_logits = tf.reduce_sum(tf.multiply(inputs_syn0, true_syn1), 1) # [N]
    sampled_logits = tf.reduce_sum(
        tf.multiply(tf.expand_dims(inputs_syn0, 1), sampled_syn1), 2) # [N, K]

    if self._add_bias:
      true_logits += tf.gather(biases, labels)  # [N]
      sampled_logits += tf.gather(biases, sampled_mat)  # [N, K]

    true_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.ones_like(true_logits), logits=true_logits)
    sampled_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.zeros_like(sampled_logits), logits=sampled_logits)
    loss = tf.concat(
        [tf.expand_dims(true_cross_entropy, 1), sampled_cross_entropy], 1)
    return loss

  def _hierarchical_softmax_loss(self, inputs, labels, syn0, syn1, biases):
    """Builds the loss for hierarchical softmax.

    Args:
      inputs: int tensor of shape [batch_size] (skip_gram) or 
        [batch_size, 2*window_size+1] (cbow)
      labels: int tensor of shape [batch_size, 2*max_depth+1]
      syn0: float tensor of shape [vocab_size, embed_size], input word 
        embeddings (i.e. weights of hidden layer).
      syn1: float tensor of shape [syn1_rows, embed_size], output word
        embeddings (i.e. weights of output layer).
      biases: float tensor of shape [syn1_rows], biases added onto the logits.

    Returns:
      loss: float tensor of shape [sum_of_code_len]
    """
    inputs_syn0_list = tf.unstack(self._get_inputs_syn0(syn0, inputs))
    codes_points_list = tf.unstack(labels)
    max_depth = (labels.shape.as_list()[1] - 1) // 2
    loss = []
    for inputs_syn0, codes_points in zip(inputs_syn0_list, codes_points_list):
      true_size = codes_points[-1]
      codes = codes_points[:true_size]
      points = codes_points[max_depth:max_depth+true_size]

      logits = tf.reduce_sum(
          tf.multiply(inputs_syn0, tf.gather(syn1, points)), 1)
      if self._add_bias:
        logits += tf.gather(biases, points)

      loss.append(tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.to_float(codes), logits=logits))
    loss = tf.concat(loss, axis=0)
    return loss

  def _get_inputs_syn0(self, syn0, inputs):
    """Builds the activations of hidden layer given input words embeddings 
    `syn0` and input word indices.

    Args:
      syn0: float tensor of shape [vocab_size, embed_size]
      inputs: int tensor of shape [batch_size] (skip_gram) or 
        [batch_size, 2*window_size+1] (cbow)

    Returns:
      inputs_syn0: [batch_size, embed_size]
    """
    if self._arch == 'skip_gram':
      inputs_syn0 = tf.gather(syn0, inputs)
    else:
      inputs_syn0 = []
      contexts_list = tf.unstack(inputs)
      for contexts in contexts_list:
        context_words = contexts[:-1]
        true_size = contexts[-1]
        inputs_syn0.append(
            tf.reduce_mean(tf.gather(syn0, context_words[:true_size]), axis=0))
      inputs_syn0 = tf.stack(inputs_syn0)
    return inputs_syn0


def seeded_vector(embed_size, seed_string):
  random = np.random.RandomState(hash(seed_string) & 0xffffffff)
  return (random.rand(embed_size) - 0.5) / embed_size


def get_syn0_init_val(table_words, embed_size, random_seed):
  return np.vstack([seeded_vector(embed_size, w + str(random_seed))
    for w in table_words]).astype(np.float32)


class WordVectors(object):
  """Word vectors of trained word2vec model. Provides APIs for retrieving
  word vector, and most similar words given a query word.
  """
  def __init__(self, syn0_final, vocab):
    """Constructor.

    Args:
      syn0_final: numpy array of shape [vocab_size, embed_size], final word
        embeddings.
      vocab_words: a list of strings, holding vocabulary words.
    """
    self._syn0_final = syn0_final
    self._vocab = vocab
    self._rev_vocab = dict([(w, i) for i, w in enumerate(vocab)])

  def __contains__(self, word):
    return word in self._rev_vocab

  def __getitem__(self, word):
    return self._syn0_final[self._rev_vocab[word]]

  def most_similar(self, word, k):
    """Finds the top-k words with smallest cosine distances w.r.t `word`.

    Args:
      word: string scalar, the query word.
      k: int scalar, num of words most similar to `word`.

    Returns:
      a list of 2-tuples with word and cosine similarities.
    """
    if word not in self._rev_vocab:
      raise ValueError("Word '%s' not found in the vocabulary" % word)
    if k >= self._syn0_final.shape[0]:
      raise ValueError("k = %d greater than vocabulary size" % k)

    v0 = self._syn0_final[self._rev_vocab[word]]
    sims = np.sum(v0 * self._syn0_final, 1) / (np.linalg.norm(v0) * 
        np.linalg.norm(self._syn0_final, axis=1))

    # maintain a sliding min-heap to keep track of k+1 largest elements
    min_pq = list(zip(sims[:k+1], range(k+1)))
    heapq.heapify(min_pq)
    for i in np.arange(k + 1, len(self._vocab)):
      if sims[i] > min_pq[0][0]:
        min_pq[0] = sims[i], i
        heapq.heapify(min_pq)
    min_pq = sorted(min_pq, key=lambda p: -p[0])
    return [(self._vocab[i], sim) for sim, i in min_pq[1:]]

