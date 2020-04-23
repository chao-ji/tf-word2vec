"""Defines word2vec model using tf.keras API.
"""
import tensorflow as tf

from dataset import WordTokenizer
from dataset import Word2VecDatasetBuilder
 

class Word2VecModel(tf.keras.Model):
  """Word2Vec model."""
  def __init__(self, 
               unigram_counts, 
               arch='skip_gram',
               algm='negative_sampling', 
               hidden_size=300, 
               batch_size=256, 
               negatives=5, 
               power=0.75,
               alpha=0.025,
               min_alpha=0.0001,
               add_bias=True,
               random_seed=0):
    """Constructor.

    Args:
      unigram_counts: a list of ints, the counts of word tokens in the corpus. 
      arch: string scalar, architecture ('skip_gram' or 'cbow').
      algm: string scalar, training algorithm ('negative_sampling' or
        'hierarchical_softmax').
      hidden_size: int scalar, length of word vector.
      batch_size: int scalar, batch size.
      negatives: int scalar, num of negative words to sample.
      power: float scalar, distortion for negative sampling. 
      alpha: float scalar, initial learning rate.
      min_alpha: float scalar, final learning rate.
      add_bias: bool scalar, whether to add bias term to dotproduct 
        between syn0 and syn1 vectors.
      random_seed: int scalar, random_seed.
    """
    super(Word2VecModel, self).__init__()
    self._unigram_counts = unigram_counts
    self._arch = arch
    self._algm = algm
    self._hidden_size = hidden_size
    self._vocab_size = len(unigram_counts)
    self._batch_size = batch_size
    self._negatives = negatives
    self._power = power
    self._alpha = alpha
    self._min_alpha = min_alpha
    self._add_bias = add_bias
    self._random_seed = random_seed

    self._input_size = (self._vocab_size if self._algm == 'negative_sampling'
                            else self._vocab_size - 1)

    self.add_weight('syn0',
                    shape=[self._vocab_size, self._hidden_size],
                    initializer=tf.keras.initializers.RandomUniform(
                        minval=-0.5/self._hidden_size,
                        maxval=0.5/self._hidden_size))
    
    self.add_weight('syn1',
                    shape=[self._input_size, self._hidden_size],
                    initializer=tf.keras.initializers.RandomUniform(
                        minval=-0.1, maxval=0.1))

    self.add_weight('biases', 
                    shape=[self._input_size], 
                    initializer=tf.keras.initializers.Zeros()) 

  def call(self, inputs, labels):
    """Runs the forward pass to compute loss.

    Args:
      inputs: int tensor of shape [batch_size] (skip_gram) or 
        [batch_size, 2*window_size+1] (cbow) 
      labels: int tensor of shape [batch_size] (negative_sampling) or
        [batch_size, 2*max_depth+1] (hierarchical_softmax)

    Returns:
      loss: float tensor, cross entropy loss. 
    """
    if self._algm == 'negative_sampling':
      loss = self._negative_sampling_loss(inputs, labels)
    elif self._algm == 'hierarchical_softmax':
      loss = self._hierarchical_softmax_loss(inputs, labels)
    return loss
 
  def _negative_sampling_loss(self, inputs, labels):
    """Builds the loss for negative sampling.

    Args:
      inputs: int tensor of shape [batch_size] (skip_gram) or 
        [batch_size, 2*window_size+1] (cbow)
      labels: int tensor of shape [batch_size]

    Returns:
      loss: float tensor of shape [batch_size, sample_size + 1].
    """

    syn0, syn1, biases = self.weights

    sampled_values = tf.random.fixed_unigram_candidate_sampler(
        true_classes=tf.expand_dims(labels, 1),
        num_true=1,
        num_sampled=self._batch_size*self._negatives,
        unique=True,
        range_max=len(self._unigram_counts),
        distortion=self._power,
        unigrams=self._unigram_counts)

    sampled = sampled_values.sampled_candidates
    sampled_mat = tf.reshape(sampled, [self._batch_size, self._negatives])
    inputs_syn0 = self._get_inputs_syn0(inputs) # [N, D]
    true_syn1 = tf.gather(syn1, labels) # [N, D]
    sampled_syn1 = tf.gather(syn1, sampled_mat) # [N, K, D]
    true_logits = tf.reduce_sum(tf.multiply(inputs_syn0, true_syn1), 1) # [N]

    sampled_logits = tf.einsum('ijk,ikl->il', tf.expand_dims(inputs_syn0, 1), 
        tf.transpose(sampled_syn1, (0, 2, 1)))

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

  def _hierarchical_softmax_loss(self, inputs, labels):
    """Builds the loss for hierarchical softmax.

    Args:
      inputs: int tensor of shape [batch_size] (skip_gram) or 
        [batch_size, 2*window_size+1] (cbow)
      labels: int tensor of shape [batch_size, 2*max_depth+1]

    Returns:
      loss: float tensor of shape [sum_of_code_len]
    """
    syn0, syn1, biases = self.weights
    max_depth = (labels.shape.as_list()[1] - 1) // 2

    def func(args):
      inputs_syn0, codes_points = args
      true_size = codes_points[-1]
      codes = codes_points[:true_size]
      points = codes_points[max_depth:max_depth+true_size]

      logits = tf.reduce_sum(
          tf.multiply(inputs_syn0, tf.gather(syn1, points)), 1)
      if self._add_bias:
        logits += tf.gather(biases, points)

      losses = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.cast(codes, 'float32'), logits=logits)
      loss = tf.reduce_sum(losses)
      count = tf.size(losses)

      loss = tf.reshape(loss, (1,)) 
      count = tf.reshape(count, (1,))
      result = tf.concat([tf.cast(loss, 'float32'), tf.cast(count, 'float32')], axis=0)
      return result

    result = tf.map_fn(func, (self._get_inputs_syn0(inputs), labels), 'float32')
    result = tf.reduce_sum(result, axis=0)
    loss = result[0] / result[1]

    return loss

  def _get_inputs_syn0(self, inputs):
    """Builds the activations of hidden layer given input words embeddings 
    `syn0` and input word indices.

    Args:
      inputs: int tensor of shape [batch_size] (skip_gram) or 
        [batch_size, 2*window_size+1] (cbow)

    Returns:
      inputs_syn0: [batch_size, embed_size]
    """
    syn0, _, _ = self.weights
    if self._arch == 'skip_gram':
      inputs_syn0 = tf.gather(syn0, inputs)
    else:

      def func(contexts):
        context_words = contexts[:-1]
        true_size = contexts[-1]
        result = tf.reduce_mean(tf.gather(syn0, context_words[:true_size]), axis=0)
        return result

      inputs_syn0 = tf.map_fn(func, inputs, dtype='float32')
      
    return inputs_syn0
