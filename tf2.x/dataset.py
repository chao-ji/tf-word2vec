"""Defines word tokenizer and word2vec dataset builder.
"""
import heapq
import itertools
import collections

import numpy as np
import tensorflow as tf

OOV_ID = -1


class WordTokenizer(object):
  """Vanilla word tokenizer that spits out space-separated tokens from raw text 
  string. Note for non-space separated languages, the corpus must be 
  pre-tokenized such that tokens are space-delimited.
  """
  def __init__(self, max_vocab_size=0, min_count=10, sample=1e-3):
    """Constructor.

    Args:
      max_vocab_size: int scalar, maximum vocabulary size. If > 0, only the top 
        `max_vocab_size` most frequent words will be kept in vocabulary.
      min_count: int scalar, words whose counts < `min_count` will not be 
        included in the vocabulary.
      sample: float scalar, subsampling rate.
    """
    self._max_vocab_size = max_vocab_size
    self._min_count = min_count
    self._sample = sample

    self._vocab = None 
    self._table_words = None
    self._unigram_counts = None
    self._keep_probs = None

  @property
  def unigram_counts(self):
    return self._unigram_counts

  @property
  def table_words(self):
    return self._table_words

  def _build_raw_vocab(self, filenames):
    """Builds raw vocabulary by iterate through the corpus once and count the 
    unique words.

    Args:
      filenames: list of strings, holding names of text files.

    Returns: 
      raw_vocab: a list of 2-tuples holding the word (string) and count (int),
        sorted in descending order of word count. 
    """
    lines = []
    for fn in filenames:
      with tf.io.gfile.GFile(fn) as f:
        lines.append(f)
    lines = itertools.chain(*lines)

    raw_vocab = collections.Counter()
    for line in lines:
      raw_vocab.update(line.strip().split())
    raw_vocab = raw_vocab.most_common()
    # truncate to have at most `max_vocab_size` vocab words
    if self._max_vocab_size > 0:
      raw_vocab = raw_vocab[:self._max_vocab_size]
    return raw_vocab
   
  def build_vocab(self, filenames):
    """Builds the vocabulary.

    Has the side effect of setting the following attributes: for each word 
    `word` we have

    vocab[word] = index
    table_words[index] = word `word`
    unigram_counts[index] = count of `word` in vocab
    keep_probs[index] = keep prob of `word` for subsampling

    Args:
      filenames: list of strings, holding names of text files.
    """
    raw_vocab = self._build_raw_vocab(filenames)
    raw_vocab = [(w, c) for w, c in raw_vocab if c >= self._min_count]
    self._corpus_size = sum(list(zip(*raw_vocab))[1])

    self._vocab = {}
    self._table_words = []
    self._unigram_counts = []
    self._keep_probs = []
    for index, (word, count) in enumerate(raw_vocab):
      frac = count / float(self._corpus_size)
      keep_prob = (np.sqrt(frac / self._sample) + 1) * (self._sample / frac)
      keep_prob = np.minimum(keep_prob, 1.0)
      self._vocab[word] = index
      self._table_words.append(word)
      self._unigram_counts.append(count)
      self._keep_probs.append(keep_prob)

  def encode(self, string):
    """Split raw text string into tokens (space-separated) and tranlate to token 
    ids.

    Args:
      string: string scalar, the raw text string to be tokenized.

    Returns:
      ids: a list of ints, the token ids of the tokenized string.
    """
    tokens = string.strip().split()
    ids = [self._vocab[token] if token in self._vocab else OOV_ID 
        for token in tokens]
    return ids


class Word2VecDatasetBuilder(object):
  """Builds a tf.data.Dataset instance that generates matrices holding word
  indices for training Word2Vec models.
  """
  def __init__(self,
               tokenizer,
               arch='skip_gram',
               algm='negative_sampling',
               epochs=1,
               batch_size=32,
               window_size=5):
    """Constructor.

    Args:
      epochs: int scalar, num times the dataset is iterated.
      batch_size: int scalar, the returned tensors in `get_tensor_dict` have
        shapes [batch_size, :]. 
      window_size: int scalar, num of words on the left or right side of
        target word within a window.
    """
    self._tokenizer = tokenizer
    self._arch = arch
    self._algm = algm
    self._epochs = epochs
    self._batch_size = batch_size
    self._window_size = window_size

    self._max_depth = None

  def _build_binary_tree(self, unigram_counts):
    """Builds a Huffman tree for hierarchical softmax. Has the side effect
    of setting `max_depth`.

    Args:
      unigram_counts: list of int, holding word counts. Index of each entry
        is the same as the word index into the vocabulary.

    Returns:
      codes_points: an int numpy array of shape [vocab_size, 2*max_depth+1]
        where each row holds the codes (0-1 binary values) padded to
        `max_depth`, and points (non-leaf node indices) padded to `max_depth`,
        of each vocabulary word. The last entry is the true length of code
        and point (<= `max_depth`).
    """
    vocab_size = len(unigram_counts)
    heap = [[unigram_counts[i], i] for i in range(vocab_size)]
    # initialize the min-priority queue, which has length `vocab_size`
    heapq.heapify(heap)

    # insert `vocab_size` - 1 internal nodes, with vocab words as leaf nodes.
    for i in range(vocab_size - 1):
      min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
      heapq.heappush(heap, [min1[0] + min2[0], i + vocab_size, min1, min2])
    # At this point we have a len-1 heap, and `heap[0]` will be the root of 
    # the binary tree; where internal nodes store
    # 1. key (frequency)
    # 2. vocab index
    # 3. left child
    # 4. right child
    # and leaf nodes store
    # 1. key (frequencey)
    # 2. vocab index

    # Traverse the Huffman tree rooted at `heap[0]` in the order of 
    # Depth-First-Search. Each stack item stores the
    # 1. `node`
    # 2. code of the `node` (list)
    # 3. point of the `node` (list)
    #
    # `point` is the list of vocab IDs of the internal nodes along the path from 
    # the root up to `node` (not included)
    # `code` is the list of labels (0 or 1) of the edges along the path from the
    # root up to `node` 
    # they are empty lists for the root node `heap[0]`
    node_list = []
    max_depth, stack = 0, [[heap[0], [], []]] # stack: [root, codde, point]
    while stack:
      node, code, point = stack.pop()
      if node[1] < vocab_size:
        # leaf node: len(node) == 2
        node.extend([code, point, len(point)])
        max_depth = np.maximum(len(code), max_depth)
        node_list.append(node)
      else:
        # internal node: len(node) == 4
        point = np.array(list(point) + [node[1]-vocab_size])
        stack.append([node[2], np.array(list(code)+[0]), point])
        stack.append([node[3], np.array(list(code)+[1]), point])

    # `len(node_list[i]) = 5`
    node_list = sorted(node_list, key=lambda items: items[1])
    # Stores the padded codes and points for each vocab word
    codes_points = np.zeros([vocab_size, max_depth*2+1], dtype=np.int64)
    for i in range(len(node_list)):
      length = node_list[i][4] # length of code or point
      codes_points[i, -1] = length
      codes_points[i, :length] = node_list[i][2] # code
      codes_points[i, max_depth:max_depth+length] = node_list[i][3] # point
    self._max_depth = max_depth
    return codes_points

  def build_dataset(self, filenames):
    """Generates tensor dict mapping from tensor names to tensors.

    Args:
      filenames: list of strings, holding names of text files.
      
    Returns:
      dataset: a tf.data.Dataset instance, holding the a tuple of tensors
        (inputs, labels, progress)
        when arch=='skip_gram', algm=='negative_sampling'
          inputs: [N],                    labels: [N]
        when arch=='cbow', algm=='negative_sampling'
          inputs: [N, 2*window_size+1],   labels: [N]
        when arch=='skip_gram', algm=='hierarchical_softmax'
          inputs: [N],                    labels: [N, 2*max_depth+1]
        when arch=='cbow', algm=='hierarchical_softmax'
          inputs: [N, 2*window_size+1],   labels: [N, 2*max_depth+1]
        progress: [N], the percentage of sentences covered so far. Used to 
          compute learning rate.
    """
    unigram_counts = self._tokenizer._unigram_counts
    keep_probs = self._tokenizer._keep_probs

    if self._algm == 'hierarchical_softmax':
      codes_points = tf.constant(self._build_binary_tree(unigram_counts))
    elif self._algm == 'negative_sampling':
      codes_points = None
    else:
      raise ValueError('algm must be hierarchical_softmax or negative_sampling')
   
    keep_probs = tf.cast(tf.constant(keep_probs), 'float32')

    # total num of sentences (lines) across text files times num of epochs
    num_sents = sum([len(list(tf.io.gfile.GFile(fn))) 
        for fn in filenames]) * self._epochs

    def generator_fn():
      for _ in range(self._epochs):
        for fn in filenames:
          with tf.io.gfile.GFile(fn) as f:
            for line in f:
              yield self._tokenizer.encode(line)

    # dataset: [([int], float)]
    dataset = tf.data.Dataset.zip((
        tf.data.Dataset.from_generator(generator_fn, tf.int64, [None]),
        tf.data.Dataset.from_tensor_slices(tf.range(num_sents) / num_sents)))
    # dataset: [([int], float)]
    dataset = dataset.map(lambda indices, progress: 
        (subsample(indices, keep_probs), progress))
    # dataset: [([int], float)]
    dataset = dataset.filter(lambda indices, progress: 
        tf.greater(tf.size(indices), 1))  # sentence must have at least 2 tokens
    # dataset: [((None, None), float)]
    dataset = dataset.map(lambda indices, progress: (generate_instances(
        indices, self._arch, self._window_size, self._max_depth, codes_points), 
        progress))
    # dataset: [((None, None)), (None,)]
    dataset = dataset.map(lambda instances, progress: (
        # replicate `progress` to size `tf.shape(instances)[:1]`
        instances, tf.fill(tf.shape(instances)[:1], progress)))
    dataset = dataset.flat_map(lambda instances, progress: 
        # form a dataset by unstacking `instances` in the first dimension,
        tf.data.Dataset.from_tensor_slices((instances, progress)))
    # batch the dataset
    dataset = dataset.batch(self._batch_size, drop_remainder=True)

    def prepare_inputs_labels(tensor, progress):
      if self._arch == 'skip_gram':
        if self._algm == 'negative_sampling':
          tensor.set_shape([self._batch_size, 2])
        else:
          tensor.set_shape([self._batch_size, 2*self._max_depth+2])
        inputs = tensor[:, :1]
        labels = tensor[:, 1:]

      else:
        if self._algm == 'negative_sampling':
          tensor.set_shape([self._batch_size, 2*self._window_size+2])
        else:
          tensor.set_shape([self._batch_size,
              2*self._window_size+2*self._max_depth+2])
        inputs = tensor[:, :2*self._window_size+1]
        labels = tensor[:, 2*self._window_size+1:]

      if self._arch == 'skip_gram':
        inputs = tf.squeeze(inputs, axis=1)
      if self._algm == 'negative_sampling':
        labels = tf.squeeze(labels, axis=1)
      progress = tf.cast(progress, 'float32')
      return inputs, labels, progress

    dataset = dataset.map(lambda tensor, progress: 
        prepare_inputs_labels(tensor, progress))

    return dataset


def subsample(indices, keep_probs):
  """Filters out-of-vocabulary words and then applies subsampling on words in a 
  sentence. Words with high frequencies have lower keep probs.

  Args:
    indices: rank-1 int tensor, the word indices within a sentence.
    keep_probs: rank-1 float tensor, the prob to drop the each vocabulary word. 

  Returns:
    indices: rank-1 int tensor, the word indices within a sentence after 
      subsampling.
  """
  indices = tf.boolean_mask(indices, tf.not_equal(indices, OOV_ID))
  keep_probs = tf.gather(keep_probs, indices)
  randvars = tf.random.uniform(tf.shape(keep_probs), 0, 1)
  indices = tf.boolean_mask(indices, tf.less(randvars, keep_probs))
  return indices


def generate_instances(
    indices, arch, window_size, max_depth=None, codes_points=None):
  """Generates matrices holding word indices to be passed to Word2Vec models 
  for each sentence. The shape and contents of output matrices depends on the 
  architecture ('skip_gram', 'cbow') and training algorithm ('negative_sampling'
  , 'hierarchical_softmax').

  It takes as input a list of word indices in a subsampled-sentence, where each
  word is a target word, and their context words are those within the window 
  centered at a target word. For skip gram architecture, `num_context_words` 
  instances are generated for a target word, and for cbow architecture, a single
  instance is generated for a target word.

  If `codes_points` is not None ('hierarchical softmax'), the word to be 
  predicted (context word for 'skip_gram', and target word for 'cbow') are 
  represented by their 'codes' and 'points' in the Huffman tree (See 
  `_build_binary_tree`). 

  Args:
    indices: rank-1 int tensor, the word indices within a sentence after
      subsampling.
    arch: scalar string, architecture ('skip_gram' or 'cbow').
    window_size: int scalar, num of words on the left or right side of
      target word within a window.
    max_depth: (Optional) int scalar, the max depth of the Huffman tree. 
    codes_points: (Optional) an int tensor of shape [vocab_size, 2*max_depth+1] 
      where each row holds the codes (0-1 binary values) padded to `max_depth`, 
      and points (non-leaf node indices) padded to `max_depth`, of each 
      vocabulary word. The last entry is the true length of code and point 
      (<= `max_depth`).
    
  Returns:
    instances: an int tensor holding word indices, with shape being
      when arch=='skip_gram', algm=='negative_sampling'
        shape: [N, 2]
      when arch=='cbow', algm=='negative_sampling'
        shape: [N, 2*window_size+2]
      when arch=='skip_gram', algm=='hierarchical_softmax'
        shape: [N, 2*max_depth+2]
      when arch=='cbow', algm='hierarchical_softmax'
        shape: [N, 2*window_size+2*max_depth+2]
  """
  def per_target_fn(index, init_array):
    """Generate inputs and labels for each target word.

    `index` is the index of the target word in `indices`.
    """
    reduced_size = tf.random.uniform([], maxval=window_size, dtype='int32')
    left = tf.range(tf.maximum(index - window_size + reduced_size, 0), index)
    right = tf.range(index + 1, 
        tf.minimum(index + 1 + window_size - reduced_size, tf.size(indices)))
    context = tf.concat([left, right], axis=0)
    context = tf.gather(indices, context)

    if arch == 'skip_gram':
      # replicate `indices[index]` to match the size of `context`
      # [N, 2]
      window = tf.stack([tf.fill(tf.shape(context), indices[index]), 
                        context], axis=1)
    elif arch == 'cbow':
      true_size = tf.size(context)
      # pad `context` to length `2 * window_size`
      window = tf.concat([tf.pad(context, [[0, 2*window_size-true_size]]), 
                          [true_size, indices[index]]], axis=0)
      # [1, 2*window_size + 2]
      window = tf.expand_dims(window, axis=0)
    else:
      raise ValueError('architecture must be skip_gram or cbow.')

    if codes_points is not None:
      # [N, 2*max_depth + 2] or [1, 2*window_size+2*max_depth+2]
      window = tf.concat([window[:, :-1], 
                          tf.gather(codes_points, window[:, -1])], axis=1)
    return index + 1, init_array.write(index, window)

  size = tf.size(indices)
  # initialize a tensor array of length `tf.size(indices)`
  init_array = tf.TensorArray('int64', size=size, infer_shape=False)
  _, result_array = tf.while_loop(lambda i, ta: i < size,
                                  per_target_fn, 
                                  [0, init_array],
                                      back_prop=False)
  instances = tf.cast(result_array.concat(), 'int64')
  if arch == 'skip_gram':
    if max_depth is None:
      instances.set_shape([None, 2])
    else:
      instances.set_shape([None, 2*max_depth+2])
  else:
    if max_depth is None:
      instances.set_shape([None, 2*window_size+2])
    else:
      instances.set_shape([None, 2*window_size+2*max_depth+2])

  return instances
