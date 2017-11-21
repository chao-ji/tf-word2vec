import itertools
import numpy as np
import tensorflow as tf

class Word2Vec(object):
  def __init__(self,
                embedding_size=100,
                window=5,
                min_word_count=5,
                max_vocab_size=None,
                subsample=1e-3,
                sorted_vocab=True,
                neg_sample_distortion=0.75,
                start_alpha=0.025,
                end_alpha=0.0001,
                skip_gram=True,
                cbow=False,
                negative_sampling=True,
                hierarchical_softmax=False,
                max_batch_size=64,
                epochs=5,
                num_neg_samples=5,
                seed=1):
    self.embedding_size = embedding_size
    self.window = window
    self.min_word_count = min_word_count
    self.max_vocab_size = max_vocab_size
    self.subsample = subsample
    self.sorted_vocab = sorted_vocab
    self.neg_sample_distortion = neg_sample_distortion
    self.start_alpha=start_alpha
    self.end_alpha=end_alpha

    if not(skip_gram ^ cbow):
      raise ValueError("Precisely one of the two model architectures (Skip-gram or CBOW) should be specified.")
    if not(negative_sampling ^ hierarchical_softmax):
      raise ValueError("Precisely one of the two mechanisms (Negative-sampleing or Hierachical-softmax) should be specified.")

    self.skip_gram = skip_gram
    self.cbow = cbow
    self.negative_sampling = negative_sampling
    self.hierarchical_softmax = hierarchical_softmax
    self.max_batch_size = max_batch_size
    self.epochs = epochs
    self.num_neg_samples = num_neg_samples
    self.seed = seed

    self.random_state = np.random.RandomState(seed)

    self._raw_vocab = None
    self._counter = None
    self.vocab = None
    self.vocabulary_size = None
    self.index2word = None
    self.num_words = None
    self._total_target_words = None

    self.embeddings = None
    self.weights = None
    self.biases = None

    self._progress = None
    self._target_words_covered = 0

  def build_vocab(self, sents):
    num_words = 0
    raw_vocab = self._get_raw_vocab(sents)
    vocab = dict()
    index2word = []
    for word, count in raw_vocab.iteritems():
      if count >= self.min_word_count:
        vocab[word] = {"count": count, "index": len(index2word)}
        index2word.append(word)
        num_words += count

    for word in index2word:
      count = vocab[word]["count"]
      fraction = count / float(num_words)
      keep_prob = (np.sqrt(fraction / self.subsample) + 1) * (self.subsample / fraction)
      keep_prob = keep_prob if keep_prob < 1.0 else 1.0
      vocab[word]["fraction"] = fraction
      vocab[word]["keep_prob"] = keep_prob

    if self.sorted_vocab:
      index2word.sort(key=lambda word: vocab[word]["count"], reverse=True)
      for i, word in enumerate(index2word):
        vocab[word]["index"] = i
    
    self._raw_vocab = raw_vocab
    self._counter = [vocab[word]["count"] for word in index2word]
    self.vocab = vocab
    self.vocabulary_size = len(vocab)
    self.index2word = index2word
    self.num_words = num_words
    self._total_target_words = float(self.num_words * self.epochs)

  def _prune_vocab(self, raw_vocab, word_count_cutoff):     
    for word in raw_vocab.keys():
      if raw_vocab[word] < word_count_cutoff:
        raw_vocab.pop(word) 

  def _get_raw_vocab(self, sents):
    raw_vocab = dict() 
    word_count_cutoff = 1
    max_vocab_size = self.max_vocab_size
    for sent in sents:
      for word in sent:
        raw_vocab[word] = raw_vocab[word] + 1 if word in raw_vocab else 1 
      if max_vocab_size and len(raw_vocab) > max_vocab_size:
        self._prune_vocab(raw_vocab, word_count_cutoff)
        word_count_cutoff += 1
    return raw_vocab

  def generate_batch(self, sents_iter):
    generator = (v for sent in sents_iter for v in self._tarcon_per_sent(sent))
    
    batch, size = [], 0
    for v in generator:
      if size < self.max_batch_size:
        batch.append(v)
        size += 1
      else:
        yield np.vstack(batch)
        batch = [v]
        size = 1

    if size > 0:
      yield np.pad(np.vstack(batch),
            [[0, self.max_batch_size - len(batch)], [0, 0]],
            mode="constant",
            constant_values=[-1])
      batch, size = [], 0
      
  def _tarcon_per_sent(self, sent):
    keep_word = lambda word: (word in self.vocab) and self.random_state.binomial(1, self.vocab[word]["keep_prob"])
    sent_trimmed = [self.vocab[word]["index"] for word in sent if keep_word(word)]
    
    def tarcon_per_target(word_index):
      target = sent_trimmed[word_index]
      reduced_size = self.random_state.randint(self.window)
      before = map(lambda i: sent_trimmed[i],
                xrange(max(word_index - self.window + reduced_size, 0), word_index))
      after = map(lambda i: sent_trimmed[i],
                xrange(word_index + 1, min(word_index + 1 + self.window - reduced_size, len(sent_trimmed))))

      context = before + after
      if self.skip_gram:
        for con in context:
          tc = np.array([[target, con]])        
          yield tc
      else:
        tc = -np.ones((1, 2 * self.window + 1))
        tc[0, 0] = target
        tc[0, 1:1+len(context)] = context
        yield tc

    for target in xrange(len(sent_trimmed)):
      for v in tarcon_per_target(target):
        yield v

    self._progress = self._target_words_covered / self._total_target_words
    self._target_words_covered += len(sent)

  def initialize_variables(self):
    def seeded_vector(seed_string):
      random = np.random.RandomState(hash(seed_string) & 0xffffffff)
      return (random.rand(self.embedding_size) - 0.5) / self.embedding_size

    embeddings_val = np.empty((self.vocabulary_size, self.embedding_size), dtype=np.float32)
    for i in xrange(self.vocabulary_size):
      embeddings_val[i] = seeded_vector(self.index2word[i] + str(self.seed))

    self.embeddings = tf.Variable(embeddings_val)
    self.weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                stddev=1.0/np.sqrt(self.embedding_size)))
    self.biases = tf.Variable(tf.zeros([self.vocabulary_size]))

  def logits_skip_gram(self, labels, inputs):
    real_batch_size = tf.reduce_sum(tf.cast(tf.greater_equal(labels, 0), tf.int32))

    # [V, D]
    embeddings = self.embeddings
    # [V, D]
    weights = self.weights
    # [V]
    biases = self.biases

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

    # [N, D]
    inputs_embeddings = tf.nn.embedding_lookup(embeddings, inputs)

    # [N, D]
    true_weights = tf.nn.embedding_lookup(weights, labels)

    # [N]
    true_biases = tf.nn.embedding_lookup(biases, labels)

    # [N, K, D]
    sampled_weights = tf.nn.embedding_lookup(weights, sampled_mat)

    # [N, K]
    sampled_biases = tf.nn.embedding_lookup(biases, sampled_mat)

    # [N]
    true_logits = tf.reduce_sum(tf.multiply(inputs_embeddings, true_weights), 1) + true_biases

    # [N, K] 
    sampled_logits = tf.reduce_sum(tf.multiply(tf.expand_dims(inputs_embeddings, 1), sampled_weights), 2) + sampled_biases
    
    true_logits, sampled_logits = true_logits[:real_batch_size], sampled_logits[:real_batch_size]
    return true_logits, sampled_logits

  def logits_cbow(self, labels, inputs):
    embeddings = self.embeddings
    weights = self.weights
    biases = self.biases

    sampled_values = tf.nn.fixed_unigram_candidate_sampler(
      true_classes=tf.expand_dims(labels, 1),
      num_true=1,
      num_sampled=self.max_batch_size * self.num_neg_samples,
      unique=True,
      range_max=self.vocabulary_size,
      distortion=0.75,
      unigrams=self._counter)

    sampled = sampled_values.sampled_candidates

    sampled_mat = tf.reshape(sampled, [self.max_batch_size, self.num_neg_samples])


     

  def loss(self, true_logits, sampled_logits):
    # [N]
    true_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.ones_like(true_logits), logits=true_logits)
    # [N, K]
    sampled_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

    # [N]
    neg_loss = tf.reduce_sum(tf.concat([tf.expand_dims(true_cross_entropy, 1), sampled_cross_entropy], 1), 1)

    return neg_loss

  def train(self, sents):
    self.build_vocab(sents)
    self.initialize_variables()

    sents_iter = itertools.chain(*itertools.tee(sents, self.epochs))
    X_iter = self.generate_batch(sents_iter)

    labels = tf.placeholder(dtype=tf.int64, shape=[self.max_batch_size])
    inputs = tf.placeholder(dtype=tf.int64, shape=[self.max_batch_size])

    progress = tf.placeholder(dtype=tf.float32, shape=[])
    lr = tf.maximum(self.start_alpha * (1 - progress) + self.end_alpha * progress, self.end_alpha) 

    true_logits, sampled_logits = self.logits_skip_gram(labels, inputs)
    neg_loss = self.loss(true_logits, sampled_logits)

    train_step = tf.train.GradientDescentOptimizer(lr).minimize(neg_loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    average_loss = 0.
    for step, X in enumerate(X_iter):
      inputs_val, labels_val = X[:, 0], X[:, 1]
      feed_dict = {inputs: inputs_val, labels: labels_val, progress: self._progress}

      _, neg_loss_val, lr_val = sess.run([train_step, neg_loss, lr], feed_dict)

      average_loss += neg_loss_val.mean()
      if step % 10000 == 0:
        if step > 0:
          average_loss /= 10000
        print "step =", step, "average_loss =", average_loss, "learning_rate =", lr_val
        average_loss = 0. 
  
    embeddings_final, weights_final, biases_final = self.embeddings.eval(), self.weights.eval(), self.biases.eval()
    return embeddings_final, weights_final, biases_final, np.vstack([inputs_val, labels_val])
