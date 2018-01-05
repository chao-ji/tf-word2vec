import heapq
import itertools
import numpy as np
import tensorflow as tf
from datetime import datetime

class VocabWord(object):
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
  __slots__ = ("max_vocab_size", "min_count", "sample", "sorted_vocab", "window", "size",
    "norm_embeddings", "negatives", "power", "alpha", "min_alpha", "max_batch_size", "epochs",
    "log_every_n_steps", "hidden_layer_toggle", "output_layer_toggle", "ns_add_bias",
    "hs_add_bias", "clip_gradient", "seed", "_random_state", "_raw_vocab", "_unigram_count",
    "vocab", "vocabulary_size", "index2word", "index2word", "num_words", "total_sents", "_syn0",
    "_syn1", "_biases", "_progress", "_sents_covered")

  def __init__(self,
                max_vocab_size=None,
                min_count=5,
                sample=1e-3,
                sorted_vocab=True,                
                window=5,
                size=100,
                norm_embeddings=False,
                negatives=5,
                power=0.75,
                alpha=0.025,
                min_alpha=0.0001,
                max_batch_size=64,
                epochs=5,
                log_every_n_steps=10000,
                hidden_layer_toggle=True,
                output_layer_toggle=True,
                ns_add_bias=True,
                hs_add_bias=True,
                clip_gradient=False,
                seed=1):
    self.max_vocab_size = max_vocab_size
    self.min_count = min_count
    self.sample = sample
    self.sorted_vocab = sorted_vocab
    self.window = window
    self.size = size
    self.norm_embeddings = norm_embeddings
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

  def build_vocab(self, sents):
    num_words = 0
    raw_vocab = self._get_raw_vocab(sents)
    vocab = dict()
    index2word = []
    for word, count in raw_vocab.iteritems():
      if count >= self.min_count:
        vocab[word] = VocabWord(count=count, index=len(index2word)) # {"count": count, "index": len(index2word)}
        index2word.append(word)
        num_words += count

    for word in index2word:
      count = vocab[word].count
      fraction = count / float(num_words)
      keep_prob = (np.sqrt(fraction / self.sample) + 1) * (self.sample / fraction)
      keep_prob = keep_prob if keep_prob < 1.0 else 1.0
      vocab[word].fraction = fraction
      vocab[word].keep_prob = keep_prob
      vocab[word].word = word

    if self.sorted_vocab:
      index2word.sort(key=lambda word: vocab[word].count, reverse=True)
      for i, word in enumerate(index2word):
        vocab[word].index = i
    
    self._raw_vocab = raw_vocab
    self._unigram_count = [vocab[word].count for word in index2word]
    self.vocab = vocab
    self.vocabulary_size = len(vocab)
    self.index2word = index2word
    self.num_words = num_words
    self.total_sents = len(sents) * self.epochs

  def _get_raw_vocab(self, sents):
    raw_vocab = dict()
    word_count_cutoff = 1
    for sent in sents:
      for word in sent:
        raw_vocab[word] = raw_vocab[word] + 1 if word in raw_vocab else 1
      if self.max_vocab_size and len(raw_vocab) > self.max_vocab_size:
        for word in raw_vocab.keys():
          if raw_vocab[word] < word_count_cutoff:
            raw_vocab.pop(word)
        word_count_cutoff += 1
    return raw_vocab

  def _get_tarcon_generator(self, sents_iter):
    return (tarcon for sent in sents_iter for tarcon in self._tarcon_per_sent(sent)) 

  def generate_batch(self, sents_iter):
    vocab, index2word = self.vocab, self.index2word

    def _sg_ns(batch):
      return np.array(batch[0]), np.array(batch[1])
    def _cbow_ns(batch):
      segment_ids = np.repeat(xrange(len(batch[0])), map(len, batch[1]))
      return np.array([np.concatenate(batch[1]), segment_ids]).T, np.array(batch[0])
    def _sg_hs(batch):
      paths = [np.array([vocab[index2word[i]].point, vocab[index2word[i]].code]).T for i in batch[1]]
      code_lengths = map(len, paths)
      labels = np.vstack(paths)
      inputs = np.repeat(batch[0], code_lengths)
      return inputs, labels
    def _cbow_hs(batch):
      paths = [np.array([vocab[index2word[i]].point, vocab[index2word[i]].code]).T for i in batch[0]] 
      code_lengths = map(len, paths)
      labels = np.vstack(paths)
      contexts_repeated = np.repeat(batch[1], code_lengths)
      contexts_repeated_segment_ids = np.repeat(xrange(len(contexts_repeated)), map(len, contexts_repeated))
      inputs = np.array([np.concatenate(contexts_repeated), contexts_repeated_segment_ids]).T
      return inputs, labels

    def _yield_fn(batch):
      hidden_toggle = self.hidden_layer_toggle
      out_toggle = self.output_layer_toggle

      if hidden_toggle and out_toggle:
        return _sg_ns(batch)
      elif (not hidden_toggle) and out_toggle:
        return _cbow_ns(batch)
      elif hidden_toggle and (not out_toggle):
        return _sg_hs(batch)
      elif (not hidden_toggle) and (not out_toggle):
        return _cbow_hs(batch)

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
    """Determine if input word will be kept."""
    return word in self.vocab and self._random_state.binomial(1, self.vocab[word].keep_prob)

  def _tarcon_per_target(self, sent_trimmed, word_index):
    target = sent_trimmed[word_index]
    reduced_size = self._random_state.randint(self.window)
    left = map(lambda i: sent_trimmed[i],
              xrange(max(word_index - self.window + reduced_size, 0), word_index))
    right = map(lambda i: sent_trimmed[i],
              xrange(word_index + 1, min(word_index + 1 + self.window - reduced_size, len(sent_trimmed))))
    contexts = left + right

    if contexts:
      if self.hidden_layer_toggle: # skip gram
        for context in contexts:
          yield target, context
      else: # cbow
        yield target, contexts

  def _tarcon_per_sent(self, sent):
    """Generator: yields 2-tuples of tar(get) and con(text) words per sentences."""
    sent_trimmed = [self.vocab[word].index for word in sent if self._keep_word(word)]

    for word_index in xrange(len(sent_trimmed)):
      for tarcon in self._tarcon_per_target(sent_trimmed, word_index):
        yield tarcon

    self._sents_covered += 1
    self._progress = self._sents_covered / float(self.total_sents)

  def create_binary_tree(self):
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

  def initialize_variables(self):
    syn0_val = np.empty((self.vocabulary_size, self.size), dtype=np.float32)
    for i in xrange(self.vocabulary_size):
      syn0_val[i] = self._seeded_vector(self.index2word[i] + str(self.seed))

    self._syn0 = tf.Variable(syn0_val, dtype=tf.float32)
    self._syn1 = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.size],
                                stddev=1.0/np.sqrt(self.size)), dtype=tf.float32)
    self._biases = tf.Variable(tf.zeros([self.vocabulary_size]), dtype=tf.float32)
    inputs = tf.placeholder(dtype=tf.int64, shape=[None] if self.hidden_layer_toggle else [None, 2])
    labels = tf.placeholder(dtype=tf.int64, shape=[None] if self.output_layer_toggle else [None, 2])
    return inputs, labels

  def _input_to_hidden(self, syn0, inputs):
    if self.hidden_layer_toggle: # skip_gram
      return tf.nn.embedding_lookup(syn0, inputs)
    else: # cbow
      return tf.segment_mean(tf.nn.embedding_lookup(syn0, inputs[:, 0]), inputs[:, 1])
  
  def loss_ns(self, inputs, labels):
    # [V, D], [V, D]
    syn0, syn1 = self._syn0, self._syn1
    sampled_values = tf.nn.fixed_unigram_candidate_sampler(
      true_classes=tf.expand_dims(labels, 1),
      num_true=1,
      num_sampled=self.max_batch_size*self.negatives,
      unique=True,
      range_max=self.vocabulary_size,
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

  def _save_embedding(self, syn0_final):
    return WordEmbeddings(syn0_final, self.vocab, self.index2word)

  def _get_train_step(self, lr, loss):
    sgd = tf.train.GradientDescentOptimizer(lr)
    if self.clip_gradient:
      gradients, variables = zip(*sgd.compute_gradients(loss))
      gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
      return sgd.apply_gradients(zip(gradients, variables))
    else:
      return sgd.minimize(loss)

  def train(self, sents, sess):
    self.build_vocab(sents)
    if not self.output_layer_toggle:
      self.create_binary_tree()

    sents_iter = self._get_sent_iter(sents)
    batch_iter = self.generate_batch(sents_iter)

    progress = tf.placeholder(dtype=tf.float32, shape=[])
    lr = tf.maximum(self.alpha * (1 - progress) + self.min_alpha * progress, self.min_alpha) 

    inputs, labels = self.initialize_variables()

    if self.output_layer_toggle: # negative sampling
      loss = self.loss_ns(inputs, labels)
    else: # hierarchical softmax
      loss = self.loss_hs(inputs, labels)

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
    if self.norm_embeddings:
      norm = np.linalg.norm(syn0_final, axis=1)
      syn0_final = syn0_final / norm

    return self._save_embedding(syn0_final)


class WordEmbeddings(object):
  def __init__(self, syn0_final, vocab, index2word):
    self.syn0_final = syn0_final
    self.vocab = vocab
    self.index2word = index2word

  def __contains__(self, word):
    return word in self.vocab

  def __getitem__(self, word):
    return self.syn0_final[self.vocab[word].index]

  def most_similar(self, word, k):
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
