import heapq
import itertools
import numpy as np
import tensorflow as tf
from collections import defaultdict
from datetime import datetime

class VocabWord(object):
  """Stores attributes (`count`, `index`, `keep_prob`, ...) for each word in vocabulary"""
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

  def __lt__(self, other):
    return self.count < other.count

  def __str__(self):
    def _which_format(kw):
      val = self.__dict__[kw]
      if isinstance(val, float):
        return "%s=%g" % (kw, val)
      else:
        return "%s=%r" % (kw, val)

    vals = [_which_format(kw) for kw in sorted(self.__dict__)]
    return "%s(%s)" % (self.__class__.__name__, ', '.join(vals))


class Word2Vec(object):
  """Trains word2vec model. The model can be trained on two model architectures 
  "skip-gram" (`sg`) or "continuous bag-of-words" (`cbow`) and by two training algorithms 
  "negative sampling" (`ns`) or "hierarchical softmax" (`hs`). 
  """

  __slots__ = ("max_vocab_size", "min_count", "sample", "sorted_vocab", "window", "size",
    "norm_embed", "negatives", "power", "alpha", "min_alpha", "max_batch_size", "epochs",
    "log_every_n_steps", "hidden_layer_toggle", "output_layer_toggle", "ns_add_bias",
    "hs_add_bias", "clip_gradient", "seed", "_random_state", "_raw_corpus", "_raw_vocab", 
    "_unigram_count", "vocab", "vocab_size", "index2word", "corpus_size", 
    "total_sents", "_syn0", "_syn1", "_biases", "_progress", "_sents_covered")

  def __init__(self,
                max_vocab_size=None,      # Maximum vocabulary size
                min_count=5,              # Minimum word count to be considered as a vocabulary word
                sample=1e-3,              # Sub-sampling rate
                sorted_vocab=True,        # Sort the vocabulary in descending order of word count
                window=5,                 # Maximum number of words on either side of a window
                size=100,                 # Embedding size
                norm_embed=False,         # Normalize word embeddings to unit norm
                negatives=5,              # Number of negative words for `ns`
                power=0.75,               # Distortion of the sampling algorithm for `ns`
                alpha=0.025,              # Initial learning rate
                min_alpha=0.0001,         # Minimum learning rate
                max_batch_size=64,        # Maximum mini-batch size
                epochs=5,                 # Number of runs over the entire corpus
                log_every_n_steps=10000,  # Prints out average loss in every N steps (mini-batches)
                hidden_layer_toggle=True, # `True` for `sg`, `False` for `cbow`
                output_layer_toggle=True, # `True` for `ns`, `False` for `hs`
                ns_add_bias=True,         # Add bias to the logit in logistic regression for `ns`
                hs_add_bias=True,         # Add bias to the logit in losistic regression for `hs`
                clip_gradient=False,      # Clip the gradient in case of gradient explosion
                seed=1):
    self.max_vocab_size = max_vocab_size
    self.min_count = min_count
    self.sample = sample
    self.sorted_vocab = sorted_vocab
    self.window = window
    self.size = size
    self.norm_embed = norm_embed
    self.negatives = negatives
    self.power = power
    self.alpha = alpha
    self.min_alpha = min_alpha
    self.max_batch_size = max_batch_size
    self.epochs = epochs
    self.log_every_n_steps = log_every_n_steps
    self.hidden_layer_toggle = hidden_layer_toggle
    self.output_layer_toggle = output_layer_toggle
    self.ns_add_bias = ns_add_bias
    self.hs_add_bias = hs_add_bias
    self.clip_gradient = clip_gradient
    self.seed = seed

    self._random_state = np.random.RandomState(seed)
    self._progress = 0. 
    self._sents_covered = 0

  def _get_raw_vocab(self, sents):
    raw_vocab = defaultdict(int)
    min_reduce = 1
    for sent in sents:
      for word in sent:
        raw_vocab[word] += 1
      if self.max_vocab_size and len(raw_vocab) > self.max_vocab_size:
        for word in raw_vocab.keys():
          if raw_vocab[word] < min_reduce:
            raw_vocab.pop(word)
        min_reduce += 1
    return raw_vocab

  def build_vocab(self, sents):
    """Builds vocabulary in a one-pass-run of the corpus"""
    corpus_size = 0
    raw_vocab = self._get_raw_vocab(sents)
    vocab = dict()
    index2word = []
    for word, count in raw_vocab.iteritems():
      if count >= self.min_count:
        vocab[word] = VocabWord(count=count, index=len(index2word))
        index2word.append(word)
        corpus_size += count

    for word in index2word:
      count = vocab[word].count
      fraction = count / float(corpus_size)
      keep_prob = (np.sqrt(fraction / self.sample) + 1) * (self.sample / fraction)
      keep_prob = keep_prob if keep_prob < 1.0 else 1.0
      vocab[word].fraction = fraction
      vocab[word].keep_prob = keep_prob
      vocab[word].word = word

    if self.sorted_vocab:
      index2word.sort(key=lambda word: vocab[word].count, reverse=True)
      for i, word in enumerate(index2word):
        vocab[word].index = i

    self._raw_corpus = sents    
    self._raw_vocab = raw_vocab
    self._unigram_count = [vocab[word].count for word in index2word]
    self.vocab = vocab
    self.vocab_size = len(vocab)
    self.index2word = index2word
    self.corpus_size = corpus_size
    self.total_sents = len(sents) * self.epochs

  def _get_tarcon_generator(self, sents_iter):
    return (tarcon for sent in sents_iter for tarcon in self._tarcon_per_sent(sent)) 

  def _sg_ns(self, batch):
    return np.array(batch[0]), np.array(batch[1])

  def _cbow_ns(self, batch):
    segment_ids = np.repeat(xrange(len(batch[0])), map(len, batch[1]))
    return np.array([np.concatenate(batch[1]), segment_ids]).T, np.array(batch[0])

  def _sg_hs(self, batch):
    paths = [np.array([self.vocab[self.index2word[i]].point, self.vocab[self.index2word[i]].code]).T for i in batch[1]]
    code_lengths = map(len, paths)
    labels = np.vstack(paths)
    inputs = np.repeat(batch[0], code_lengths)
    return inputs, labels

  def _cbow_hs(self, batch):
    paths = [np.array([self.vocab[self.index2word[i]].point, self.vocab[self.index2word[i]].code]).T for i in batch[0]]
    code_lengths = map(len, paths)
    labels = np.vstack(paths)
    contexts_repeated = np.repeat(batch[1], code_lengths, axis=0)
    contexts_repeated_segment_ids = np.repeat(xrange(len(contexts_repeated)), map(len, contexts_repeated))
    inputs = np.array([np.concatenate(contexts_repeated), contexts_repeated_segment_ids]).T
    return inputs, labels

  def generate_batch(self, sents_iter):
    """Generates word indices in a mini-batch""" 
    def _yield_fn(batch):
      hidden_toggle = self.hidden_layer_toggle
      out_toggle = self.output_layer_toggle

      if hidden_toggle and out_toggle:
        return self._sg_ns(batch)
      elif (not hidden_toggle) and out_toggle:
        return self._cbow_ns(batch)
      elif hidden_toggle and (not out_toggle):
        return self._sg_hs(batch)
      elif (not hidden_toggle) and (not out_toggle):
        return self._cbow_hs(batch)

    tarcon_generator = self._get_tarcon_generator(sents_iter) 

    batch = []
    for tarcon in tarcon_generator:
      if len(batch) < self.max_batch_size:
        batch.append(tarcon)
      else:
        batch = zip(*batch)
        yield _yield_fn(batch) 
        batch = [tarcon]

    if batch: # last batch if not empty
      batch = zip(*batch)
      yield _yield_fn(batch)
      batch = []

  def _keep_word(self, word):
    return word in self.vocab and self._random_state.binomial(1, self.vocab[word].keep_prob)

  def _words_to_left(self, index_list, word_index, reduced_size):
    return map(lambda i: index_list[i], 
      xrange(max(word_index - self.window + reduced_size, 0), word_index))

  def _words_to_right(self, index_list, word_index, reduced_size):
    return map(lambda i: index_list[i], 
      xrange(word_index + 1, min(word_index + 1 + self.window - reduced_size, len(index_list))))

  def _tarcon_per_target(self, index_list, word_index):
    target = index_list[word_index]
    reduced_size = self._random_state.randint(self.window)
    left = self._words_to_left(index_list, word_index, reduced_size)
    right = self._words_to_right(index_list, word_index, reduced_size)
    contexts = left + right

    if contexts:
      if self.hidden_layer_toggle: # skip gram
        for context in contexts:
          yield target, context
      else: # cbow
        yield target, contexts

  def _tarcon_per_sent(self, sent):
    sent_subsampled= [self.vocab[word].index for word in sent if self._keep_word(word)]

    for word_index in xrange(len(sent_subsampled)):
      for tarcon in self._tarcon_per_target(sent_subsampled, word_index):
        yield tarcon

    self._sents_covered += 1
    self._progress = self._sents_covered / float(self.total_sents)

  def create_binary_tree(self):
    """Builds huffmann tree based on the frequencies of all vocabulary words"""
    vocab = self.vocab
    heap = list(vocab.itervalues())
    heapq.heapify(heap)
    for i in xrange(len(vocab) - 1):
      min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
      heapq.heappush(heap, VocabWord(count=min1.count+min2.count, index=i+len(vocab), left=min1, right=min2))

    max_depth, stack = 0, [(heap[0], [], [])]
    while stack:
      node, code, point = stack.pop()
      if node.index < len(vocab):
        node.code, node.point, node.codelen = code, point, len(point)
        max_depth = max(len(code), max_depth)
      else:
        point = np.array(list(point) + [node.index - len(vocab)], dtype=np.uint32)
        stack.append((node.left, np.array(list(code) + [0], dtype=np.uint8), point))
        stack.append((node.right, np.array(list(code) + [1], dtype=np.uint8), point))

  def _seeded_vector(self, seed_string):
    random = np.random.RandomState(hash(seed_string) & 0xffffffff)
    return (random.rand(self.size) - 0.5) / self.size

  def create_variables(self):
    """Defines `tf.Variable` and `tf.placeholfer`"""
    syn0_val = np.empty((self.vocab_size, self.size), dtype=np.float32)
    for i in xrange(self.vocab_size):
      syn0_val[i] = self._seeded_vector(self.index2word[i] + str(self.seed))
    syn1_rows = self.vocab_size if self.output_layer_toggle else self.vocab_size - 1

    self._syn0 = tf.get_variable("syn0", initializer=syn0_val, dtype=tf.float32)
    self._syn1 = tf.get_variable("syn1", 
      initializer=tf.truncated_normal([syn1_rows, self.size], 
      stddev=1.0/np.sqrt(self.size)), dtype=tf.float32)
    self._biases = tf.get_variable("biases", initializer=tf.zeros([syn1_rows]), dtype=tf.float32)
    inputs = tf.placeholder(dtype=tf.int64)
    labels = tf.placeholder(dtype=tf.int64)
    return inputs, labels

  def _input_to_hidden(self, syn0, inputs):
    if self.hidden_layer_toggle: # skip_gram
      return tf.nn.embedding_lookup(syn0, inputs)
    else: # cbow
      return tf.segment_mean(tf.nn.embedding_lookup(syn0, inputs[:, 0]), inputs[:, 1])
  
  def loss_ns(self, inputs, labels):
    """Loss for negative sampling (`ns`)
    V=vocab_size, D=embed_size, N=batch_size, K=negative_samples"""
    # [V, D], [V, D]
    syn0, syn1 = self._syn0, self._syn1
    sampled_values = tf.nn.fixed_unigram_candidate_sampler(
      true_classes=tf.expand_dims(labels, 1),
      num_true=1,
      num_sampled=self.max_batch_size*self.negatives,
      unique=True,
      range_max=self.vocab_size,
      distortion=self.power,
      unigrams=self._unigram_count)
    # [N * K]
    sampled = sampled_values.sampled_candidates
    # [N, K]
    sampled_mat = tf.reshape(sampled, [self.max_batch_size, self.negatives])
    sampled_mat = sampled_mat[:tf.shape(labels)[0]]
    # [N, D]
    inputs_syn0 = self._input_to_hidden(syn0, inputs)
    # [N, D]
    true_syn1 = tf.nn.embedding_lookup(syn1, labels)
    # [N, K, D]
    sampled_syn1 = tf.nn.embedding_lookup(syn1, sampled_mat)
    # [N]
    true_logits = tf.reduce_sum(tf.multiply(inputs_syn0, true_syn1), 1)
    # [N, K] 
    sampled_logits = tf.reduce_sum(tf.multiply(tf.expand_dims(inputs_syn0, 1), sampled_syn1), 2)

    if self.ns_add_bias:
      true_logits += tf.nn.embedding_lookup(self._biases, labels)
      sampled_logits += tf.nn.embedding_lookup(self._biases, sampled_mat)
    # [N]
    true_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.ones_like(true_logits), logits=true_logits)
    # [N, K]
    sampled_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.zeros_like(sampled_logits), logits=sampled_logits)
    # [N, K+1]
    loss = tf.concat([tf.expand_dims(true_cross_entropy, 1), sampled_cross_entropy], 1)
    return loss

  def loss_hs(self, inputs, labels):
    """Loss for hierarchical sampling (`hs`) 
    V=vocab_size, D=embed_size"""
    # [V, D], [V, D]
    syn0, syn1 = self._syn0, self._syn1
    # [SUM(CODE_LENGTHS), D]
    inputs_syn0 = self._input_to_hidden(syn0, inputs)
    # [SUM(CODE_LENGTHS), D]
    labels_syn1 = tf.nn.embedding_lookup(syn1, labels[:, 0])
    # [SUM(CODE_LENGTHS)]
    logits_batch = tf.reduce_sum(tf.multiply(inputs_syn0, labels_syn1), 1)

    if self.hs_add_bias:
      logits_batch += tf.nn.embedding_lookup(self._biases, labels[:, 0])
    # [SUM(CODE_LENGTHS)]
    labels_batch = tf.cast(labels[:, 1], tf.float32)
    # [SUM(CODE_LENGTHS)]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_batch, logits=logits_batch)
    return loss

  def _get_sent_iter(self, sents):
    return itertools.chain(*itertools.tee(sents, self.epochs))

  def _wrap_syn0(self, syn0_final):
    return WordVectors(syn0_final, self.vocab, self.index2word)

  def _get_train_step(self, lr, loss):
    sgd = tf.train.GradientDescentOptimizer(lr)
    if self.clip_gradient:
      gradients, variables = zip(*sgd.compute_gradients(loss))
      gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
      return sgd.apply_gradients(zip(gradients, variables))
    else:
      return sgd.minimize(loss)

  def train(self, sents, sess):
    """Trains word2vec model.

    Args:
      `sents`: an iterable of list of strings
      `sess`: TensorFlow session

    Returns:
      `WordVectors` instance
    """
    self.build_vocab(sents)
    if not self.output_layer_toggle:
      self.create_binary_tree()

    sents_iter = self._get_sent_iter(sents)
    batch_iter = self.generate_batch(sents_iter)
    progress = tf.placeholder(dtype=tf.float32, shape=[])
    lr = tf.maximum(self.alpha * (1 - progress) + self.min_alpha * progress, self.min_alpha) 

    inputs, labels = self.create_variables()
    loss = self.loss_ns(inputs, labels) if self.output_layer_toggle \
            else self.loss_hs(inputs, labels)

    train_step = self._get_train_step(lr, loss)
    sess.run(tf.global_variables_initializer())
    average_loss = 0.

    for step, batch in enumerate(batch_iter):
      feed_dict = {inputs: batch[0], labels: batch[1], progress: self._progress} 

      _, loss_val, lr_val = sess.run([train_step, loss, lr], feed_dict)

      average_loss += loss_val.mean()
      if step % self.log_every_n_steps == 0:
        if step > 0:
          average_loss /= self.log_every_n_steps
        print "step =", step, "average_loss =", average_loss, "learning_rate =", lr_val
        average_loss = 0. 

    syn0_final = self._syn0.eval()
    if self.norm_embed:
      syn0_final = syn0_final / np.linalg.norm(syn0_final, axis=1) 
    return self._wrap_syn0(syn0_final)


class WordVectors(object):
  """Trained word2vec model. Stores the index-to-word mapping, vocabulary and 
  final word embeddings"""

  def __init__(self, syn0_final, vocab, index2word):
    self.syn0_final = syn0_final
    self.vocab = vocab
    self.index2word = index2word

  def __contains__(self, word):
    return word in self.vocab

  def __getitem__(self, word):
    return self.syn0_final[self.vocab[word].index]

  def most_similar(self, word, k):
    """Finds the top-k words with smallest cosine distances w.r.t `word`"""
    if word not in self.vocab:
      raise ValueError("Word '%s' not found in the vocabulary" % word)
    if k >= self.syn0_final.shape[0]:
      raise ValueError("k = %d greater than vocabulary size" % k)

    v0 = self.syn0_final[self.vocab[word].index]
    sims = np.sum(v0*self.syn0_final, 1) / (np.linalg.norm(v0)*np.linalg.norm(self.syn0_final, axis=1)) 

    # maintain a sliding min priority queue to keep track of k+1 largest elements
    min_pq = zip(sims[:k+1], xrange(k+1))
    heapq.heapify(min_pq)
    for i in np.arange(k+1, len(self.vocab)):
      if sims[i] > min_pq[0][0]:
        min_pq[0] = sims[i], i
        heapq.heapify(min_pq)
    min_pq = sorted(min_pq, key=lambda p: -p[0]) 
    return [(self.index2word[i], sim) for sim, i in min_pq[1:]]
