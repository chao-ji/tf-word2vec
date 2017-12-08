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
    self.vocab = None
    self.vocabulary_size = None
    self.index2word = None
    self.num_words = None
    self._total_sents = None

    self.syn0 = None
    self.syn1 = None

    self._progress = 0. 
    self._sents_covered = 0

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
    self.vocab = vocab
    self.vocabulary_size = len(vocab)
    self.index2word = index2word
    self.num_words = num_words
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
    vocab, index2word = self.vocab, self.index2word

    def sg_ns(batch):
      return np.array(batch[0]), np.array(batch[1]), len(batch[0])
    def cbow_ns(batch):
      ids = np.repeat(xrange(len(batch[0])), map(len, batch[1]))
      return np.array([np.concatenate(batch[1]),  ids]).T, np.array(batch[0]), len(batch[0])
    def sg_hs(batch):
      tmp = [np.array([vocab[index2word[i]].point, vocab[index2word[i]].code]).T for i in batch[1]]
      lengths = map(len, tmp)
      ids = np.repeat(xrange(len(batch[0])), lengths).reshape((-1, 1))
      labels = np.hstack([np.vstack(tmp), ids])
      inputs = np.repeat(batch[0], lengths)
      return inputs, labels, len(batch[0])
    def cbow_hs(batch):
      tmp = [np.array([vocab[index2word[i]].point, vocab[index2word[i]].code]).T for i in batch[0]] 
      lengths = map(len, tmp)
      ids = np.repeat(xrange(len(batch[0])), lengths).reshape((-1, 1))
      labels = np.hstack([np.vstack(tmp), ids])
      contexts_rep = np.repeat(batch[1], lengths)
      contexts_rep_ids = np.repeat(xrange(len(contexts_rep)), map(len, contexts_rep))
      inputs = np.array([np.concatenate(contexts_rep), contexts_rep_ids]).T
      return inputs, labels, len(batch[0])

    def _yield_fn(batch):
      opts = self.opts
      if opts[0] and opts[2]:
        return sg_ns(batch)
      elif opts[1] and opts[2]:
        return cbow_ns(batch)
      elif opts[0] and opts[3]:
        return sg_hs(batch)
      elif opts[1] and opts[3]:
        return cbow_hs(batch)

    generator = (v for sent in sents_iter for v in self._tarcon_per_sent(sent))

    batch = []
    for v in generator:
      if len(batch) < self.max_batch_size:
        batch.append(v)
      else:
        batch = zip(*batch)
        yield _yield_fn(batch) 
        batch = [v]

    if batch:
      batch = zip(*batch)
      yield _yield_fn(batch)
      batch = []

  def _keep_word(self, word):
    return word in self.vocab and self.random_state.binomial(1, self.vocab[word].keep_prob)

  def _tarcon_per_sent(self, sent):
    sent_trimmed = [self.vocab[word].index for word in sent if self._keep_word(word)]

    def tarcon_per_target(word_index):
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

    for target in xrange(len(sent_trimmed)):
      for v in tarcon_per_target(target):
        yield v

    self._sents_covered += 1
    self._progress = self._sents_covered / self._total_sents

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

  def initialize_variables(self):
    def seeded_vector(seed_string):
      random = np.random.RandomState(hash(seed_string) & 0xffffffff)
      return (random.rand(self.embedding_size) - 0.5) / self.embedding_size

    syn0_val = np.empty((self.vocabulary_size, self.embedding_size), dtype=np.float32)
    for i in xrange(self.vocabulary_size):
      syn0_val[i] = seeded_vector(self.index2word[i] + str(self.seed))

    self.syn0 = tf.Variable(syn0_val, dtype=tf.float32)
#    self.syn1 = tf.Variable(tf.zeros([self.vocabulary_size, self.embedding_size]))
    self.syn1 = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                stddev=1.0/np.sqrt(self.embedding_size)), dtype=tf.float32)

    inputs = tf.placeholder(dtype=tf.int64, shape=[None] if self.opts[0] else [None, 2])
    labels = tf.placeholder(dtype=tf.int64, shape=[None] if self.opts[2] else [None, 3])
    real_batch_size = tf.placeholder(dtype=tf.int32, shape=[])

    return inputs, labels, real_batch_size

  def loss_ns(self, inputs=None, labels=None, real_batch_size=None):
    # [V, D], [V, D]
    syn0, syn1 = self.syn0, self.syn1
    sampled_values = tf.nn.fixed_unigram_candidate_sampler(
      true_classes=tf.expand_dims(labels, 1),
      num_true=1,
      num_sampled=self.max_batch_size * self.num_neg_samples,
      unique=True,
      range_max=self.vocabulary_size,
      distortion=0.75,
      unigrams=self._counter)
    # [N * K]
    sampled = sampled_values.sampled_candidates
    # [N, K]
    sampled_mat = tf.reshape(sampled, [self.max_batch_size, self.num_neg_samples])
    sampled_mat = sampled_mat[:real_batch_size]
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
    neg_loss = true_cross_entropy + tf.reduce_sum(sampled_cross_entropy, 1)
    return neg_loss

  def loss_hs(self, inputs=None, labels=None, real_batch_size=None):
    # [V, D], [V, D]
    syn0, syn1 = self.syn0, self.syn1
    if self.opts[0]: # skip_gram
      inputs_syn0 = tf.nn.embedding_lookup(syn0, inputs)
    else: # cbow
      inputs_syn0 = tf.segment_mean(tf.nn.embedding_lookup(syn0, inputs[:, 0]), inputs[:, 1])
    labels_syn1 = tf.nn.embedding_lookup(syn1, labels[:, 0])
    logits_batch = tf.reduce_sum(tf.multiply(inputs_syn0, labels_syn1), 1)
    labels_batch = tf.cast(labels[:, 1], tf.float32)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_batch, logits=logits_batch)
    hs_loss = tf.segment_sum(loss, labels[:, 2])
    return hs_loss

  def train(self, sents, sess):
    self.build_vocab(sents)
    if self.opts[3]:
      self.create_binary_tree()

    sents_iter = itertools.chain(*itertools.tee(sents, self.epochs))
    X_iter = self.generate_batch(sents_iter)

    progress = tf.placeholder(dtype=tf.float32, shape=[])
    lr = tf.maximum(self.start_alpha * (1 - progress) + self.end_alpha * progress, self.end_alpha) 

    inputs, labels, real_batch_size = self.initialize_variables()

    if self.opts[2]: # negative sampling
      loss = self.loss_ns(inputs=inputs, labels=labels, real_batch_size=real_batch_size)
    else: # hierarchical softmax
      loss = self.loss_hs(inputs=inputs, labels=labels, real_batch_size=real_batch_size)

    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    sess.run(tf.global_variables_initializer())
    average_loss = 0.

    for step, batch in enumerate(X_iter):
      feed_dict = {inputs: batch[0], labels: batch[1], real_batch_size: batch[2]} 
      feed_dict[progress] = self._progress

      _, loss_val, lr_val = sess.run([train_step, loss, lr], feed_dict)

      average_loss += loss_val.mean()
      if step % self.log_every_n_steps == 0:
        if step > 0:
          average_loss /= self.log_every_n_steps
        print "step =", step, "average_loss =", average_loss, "learning_rate =", lr_val
        average_loss = 0. 

    syn0_final, syn1_final = self.syn0.eval(), self.syn1.eval()
    if self.norm_embeddings:
      norm =  np.sqrt(np.square(syn0_final).sum(axis=1, keepdims=True)) 
      syn0_final = syn0_final / norm

    return syn0_final, syn1_final 
