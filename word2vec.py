import heapq
import itertools
import numpy as np
import tensorflow as tf
from datetime import datetime
from scipy.spatial.distance import cosine

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

dtype = tf.float32

class Word2Vec(object):
  def __init__(self,
                max_vocab_size=None,
                min_word_count=5,
                subsample=1e-3,
                sorted_vocab=True,                
                window=5,
                embedding_size=100,
                norm_embeddings=False,
                num_neg_samples=5,
                neg_sample_distortion=0.75,
                start_alpha=0.025,
                end_alpha=0.0001,
                max_batch_size=64,
                epochs=5,
                log_every_n_steps=10000,
                opts=[True,False,True,False],
                seed=1):
    if not(opts[0] ^ opts[1]):
      raise ValueError("Exactly one of the two model architectures (`skip gram` or `cbow`) need to be specified.")
    if not(opts[2] ^ opts[3]):
      raise ValueError("Exactly one of the two training algorithms (`neg sampling` or `hierarchical_softmax`) need to be specified.")
    self.max_vocab_size = max_vocab_size
    self.min_word_count = min_word_count
    self.subsample = subsample
    self.sorted_vocab = sorted_vocab
    self.window = window
    self.embedding_size = embedding_size
    self.norm_embeddings = norm_embeddings
    self.num_neg_samples = num_neg_samples
    self.neg_sample_distortion = neg_sample_distortion
    self.start_alpha=start_alpha
    self.end_alpha=end_alpha
    self.max_batch_size = max_batch_size
    self.epochs = epochs
    self.log_every_n_steps=log_every_n_steps
    self.opts = opts
    self.seed = seed

    self.random_state = np.random.RandomState(seed)

    self._raw_vocab = None
    self._counter = None
    self._vocab = None
    self._vocabulary_size = None
    self._index2word = None
    self._num_words = None
    self._total_sents = None

    self._syn0 = None
    self._syn1 = None

    self._progress = 0. 
    self._sents_covered = 0

  @property
  def vocab(self):
    return self._vocab

  @property
  def vocabulary_size(self):
    return self._vocabulary_size

  @property
  def index2word(self):
    return self._index2word

  @property
  def num_words(self):
    return self._num_words

  def build_vocab(self, sents):
    num_words = 0
    raw_vocab = self._get_raw_vocab(sents)
    vocab = dict()
    index2word = []
    for word, count in raw_vocab.iteritems():
      if count >= self.min_word_count:
        vocab[word] = VocabWord(count=count, index=len(index2word)) # {"count": count, "index": len(index2word)}
        index2word.append(word)
        num_words += count

    for word in index2word:
      count = vocab[word].count
      fraction = count / float(num_words)
      keep_prob = (np.sqrt(fraction / self.subsample) + 1) * (self.subsample / fraction)
      keep_prob = keep_prob if keep_prob < 1.0 else 1.0
      vocab[word].fraction = fraction
      vocab[word].keep_prob = keep_prob
      vocab[word].word = word

    if self.sorted_vocab:
      index2word.sort(key=lambda word: vocab[word].count, reverse=True)
      for i, word in enumerate(index2word):
        vocab[word].index = i
    
    self._raw_vocab = raw_vocab
    self._counter = [vocab[word].count for word in index2word]
    self._vocab = vocab
    self._vocabulary_size = len(vocab)
    self._index2word = index2word
    self._num_words = num_words
    self._total_sents = float(len(sents) * self.epochs)

  def _prune_vocab(self, raw_vocab, word_count_cutoff):     
    for word in raw_vocab.keys():
      if raw_vocab[word] < word_count_cutoff:
        raw_vocab.pop(word) 

  def _get_raw_vocab(self, sents):
    raw_vocab = dict()
    word_count_cutoff = 1
    for sent in sents:
      for word in sent:
        raw_vocab[word] = raw_vocab[word] + 1 if word in raw_vocab else 1
      if self.max_vocab_size and len(raw_vocab) > self.max_vocab_size:
        self._prune_vocab(raw_vocab, word_count_cutoff)
        word_count_cutoff += 1
    return raw_vocab

  def generate_batch(self, sents_iter):
    vocab, index2word = self._vocab, self._index2word

    def _sg_ns(batch):
      return np.array(batch[0]), np.array(batch[1])
    def _cbow_ns(batch):
      segment_ids = np.repeat(xrange(len(batch[0])), map(len, batch[1]))
      return np.array([np.concatenate(batch[1]), segment_ids]).T, np.array(batch[0])
    def _sg_hs(batch):
      tmp = [np.array([vocab[index2word[i]].point, vocab[index2word[i]].code]).T for i in batch[1]]
      code_lengths = map(len, tmp)
      segment_ids = np.repeat(xrange(len(batch[0])), code_lengths).reshape((-1, 1))
      labels = np.hstack([np.vstack(tmp), segment_ids])
      inputs = np.repeat(batch[0], code_lengths)
      return inputs, labels
    def _cbow_hs(batch):
      tmp = [np.array([vocab[index2word[i]].point, vocab[index2word[i]].code]).T for i in batch[0]] 
      code_lengths = map(len, tmp)
      segment_ids = np.repeat(xrange(len(batch[0])), code_lengths).reshape((-1, 1))
      labels = np.hstack([np.vstack(tmp), segment_ids])
      contexts_repeated = np.repeat(batch[1], code_lengths)
      contexts_repeated_segment_ids = np.repeat(xrange(len(contexts_repeated)), map(len, contexts_repeated))
      inputs = np.array([np.concatenate(contexts_repeated), contexts_repeated_segment_ids]).T
      return inputs, labels

    def _yield_fn(batch):
      opts = self.opts
      if opts[0] and opts[2]:
        return _sg_ns(batch)
      elif opts[1] and opts[2]:
        return _cbow_ns(batch)
      elif opts[0] and opts[3]:
        return _sg_hs(batch)
      elif opts[1] and opts[3]:
        return _cbow_hs(batch)

    tarcon_generator = (tarcon for sent in sents_iter for tarcon in self._tarcon_per_sent(sent))

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
    """Determine if input word will be kept

    Args:
      `word`: string
    Returns/Yields:
      bool
    """
    return word in self._vocab and self.random_state.binomial(1, self._vocab[word].keep_prob)

  def _tarcon_per_sent(self, sent):
    """Generator: yields 2-tuples of tar(get) and con(text) words per sentences

    Args:
      `sent`: list of strings
    Returns/Yields:
      2-tuple of word indices
    """
    sent_trimmed = [self._vocab[word].index for word in sent if self._keep_word(word)]

    def _tarcon_per_target(word_index):
      target = sent_trimmed[word_index]
      reduced_size = self.random_state.randint(self.window)
      before = map(lambda i: sent_trimmed[i],
                xrange(max(word_index - self.window + reduced_size, 0), word_index))
      after = map(lambda i: sent_trimmed[i],
                xrange(word_index + 1, min(word_index + 1 + self.window - reduced_size, len(sent_trimmed))))
      contexts = before + after

      if contexts:
        if self.opts[0]: # skip gram
          for context in contexts:
            yield target, context
        else: # cbow
          yield target, contexts

    for word_index in xrange(len(sent_trimmed)):
      for tarcon in _tarcon_per_target(word_index):
        yield tarcon

    self._sents_covered += 1
    self._progress = self._sents_covered / self._total_sents

  def create_binary_tree(self):
    vocab = self._vocab
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

  def initialize_variables(self):
    def seeded_vector(seed_string):
      random = np.random.RandomState(hash(seed_string) & 0xffffffff)
      return (random.rand(self.embedding_size) - 0.5) / self.embedding_size

    syn0_val = np.empty((self._vocabulary_size, self.embedding_size), dtype=np.float32)
    for i in xrange(self._vocabulary_size):
      syn0_val[i] = seeded_vector(self._index2word[i] + str(self.seed))

    self._syn0 = tf.Variable(syn0_val, dtype=dtype)
    self._syn1 = tf.Variable(tf.truncated_normal([self._vocabulary_size, self.embedding_size],
                                stddev=1.0/np.sqrt(self.embedding_size)), dtype=dtype)

    inputs = tf.placeholder(dtype=tf.int64, shape=[None] if self.opts[0] else [None, 2])
    labels = tf.placeholder(dtype=tf.int64, shape=[None] if self.opts[2] else [None, 3])

    return inputs, labels

  def loss_ns(self, inputs, labels):
    # [V, D], [V, D]
    syn0, syn1 = self._syn0, self._syn1
    sampled_values = tf.nn.fixed_unigram_candidate_sampler(
      true_classes=tf.expand_dims(labels, 1),
      num_true=1,
      num_sampled=self.max_batch_size * self.num_neg_samples,
      unique=False,
      range_max=self._vocabulary_size,
      distortion=0.75,
      unigrams=self._counter)
    # [N * K]
    sampled = sampled_values.sampled_candidates
    # [N, K]
    sampled_mat = tf.reshape(sampled, [self.max_batch_size, self.num_neg_samples])
    sampled_mat = sampled_mat[:tf.shape(labels)[0]]
    # [N, D]
    if self.opts[0]: # skip_gram
      inputs_syn0 = tf.nn.embedding_lookup(syn0, inputs)
    else: # cbow
      inputs_syn0 = tf.segment_mean(tf.nn.embedding_lookup(syn0, inputs[:, 0]), inputs[:, 1])
    # [N, D]
    true_syn1 = tf.nn.embedding_lookup(syn1, labels)
    # [N, K, D]
    sampled_syn1 = tf.nn.embedding_lookup(syn1, sampled_mat)
    # [N]
    true_logits = tf.reduce_sum(tf.multiply(inputs_syn0, true_syn1), 1)          
    # [N, K] 
    sampled_logits = tf.reduce_sum(tf.multiply(tf.expand_dims(inputs_syn0, 1), sampled_syn1), 2)  
    # [N]
    true_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.ones_like(true_logits), logits=true_logits)
    # [N, K]
    sampled_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.zeros_like(sampled_logits), logits=sampled_logits)
    # [N]
    loss = tf.reduce_mean(tf.concat([tf.expand_dims(true_cross_entropy, 1), sampled_cross_entropy], 1), 1)
    return loss

  def loss_hs(self, inputs, labels):
    # [V, D], [V, D]
    syn0, syn1 = self._syn0, self._syn1
    # [SUM(CODE_LENGTHS), D]
    if self.opts[0]: # skip_gram
      inputs_syn0 = tf.nn.embedding_lookup(syn0, inputs)
    else: # cbow
      inputs_syn0 = tf.segment_mean(tf.nn.embedding_lookup(syn0, inputs[:, 0]), inputs[:, 1])
    # [SUM(CODE_LENGTHS), D]
    labels_syn1 = tf.nn.embedding_lookup(syn1, labels[:, 0])
    # [SUM(CODE_LENGTHS)]
    logits_batch = tf.reduce_sum(tf.multiply(inputs_syn0, labels_syn1), 1)
    # [SUM(CODE_LENGTHS)]
    labels_batch = tf.cast(labels[:, 1], dtype)
    # [SUM(CODE_LENGTHS)]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_batch, logits=logits_batch)
    # [N]
    loss = tf.segment_mean(loss, labels[:, 2])
    return loss

  def train(self, sents, sess):
    self.build_vocab(sents)
    if self.opts[3]:
      self.create_binary_tree()

    sents_iter = itertools.chain(*itertools.tee(sents, self.epochs))
    X_iter = self.generate_batch(sents_iter)

    progress = tf.placeholder(dtype=dtype, shape=[])
    lr = tf.maximum(self.start_alpha * (1 - progress) + self.end_alpha * progress, self.end_alpha) 

    inputs, labels = self.initialize_variables()

    if self.opts[2]: # negative sampling
      loss = self.loss_ns(inputs, labels)
    else: # hierarchical softmax
      loss = self.loss_hs(inputs, labels)

    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    sess.run(tf.global_variables_initializer())
    average_loss = 0.

    for step, batch in enumerate(X_iter):
      feed_dict = {inputs: batch[0], labels: batch[1]} 
      feed_dict[progress] = self._progress

      _, loss_val, lr_val = sess.run([train_step, loss, lr], feed_dict)

      average_loss += loss_val.mean()
      if step % self.log_every_n_steps == 0:
        if step > 0:
          average_loss /= self.log_every_n_steps
        print "step =", step, "average_loss =", average_loss, "learning_rate =", lr_val
        average_loss = 0. 

    syn0_final, syn1_final = self._syn0.eval(), self._syn1.eval()
    if self.norm_embeddings:
      norm =  np.sqrt(np.square(syn0_final).sum(axis=1, keepdims=True)) 
      syn0_final = syn0_final / norm

    return Embeddings(syn0_final, self.vocab, self.index2word)

class Embeddings(object):
  def __init__(self, syn0_final, vocab, index2word):
    self.syn0_final = syn0_final
    self.vocab = vocab
    self.index2word = index2word

  def most_similar(self, word, k):
    if word not in self.vocab:
      raise ValueError("Word '%s' not found in the vocabulary" % word)
    if k >= self.syn0_final.shape[0]:
      raise ValueError("k = %d greater than vocabulary size" % k)

    v0 = self.syn0_final[self.vocab[word].index]

    sims = [(i, 1-cosine(v, v0)) for (i, v) in enumerate(self.syn0_final)]
    sims.sort(key=lambda p: -p[1])

    return [(self.index2word[index], sim) for (index, sim) in sims[:k+1]]
