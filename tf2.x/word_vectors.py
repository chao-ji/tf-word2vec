"""Defines wrapper class for final word vectors.
"""
import heapq
import numpy as np


class WordVectors(object):
  """Word vectors of trained Word2Vec model. Provides APIs for retrieving
  word vector, and most similar words given a query word.
  """
  def __init__(self, syn0_final, vocab):
    """Constructor.

    Args:
      syn0_final: numpy array of shape [vocab_size, embed_size], final word
        embeddings.
      vocab: a list of strings, holding vocabulary words.
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
