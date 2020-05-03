"""Defines utility functions.
"""
import tensorflow as tf


def get_train_step_signature(
    arch, algm, batch_size, window_size=None, max_depth=None):
  """Get the training step signatures for `inputs`, `labels` and `progress` 
  tensor.

  Args:
    arch: string scalar, architecture ('skip_gram' or 'cbow').
    algm: string scalar, training algorithm ('negative_sampling' or
      'hierarchical_softmax').

  Returns:
    train_step_signature: a list of three tf.TensorSpec instances,
      specifying the tensor spec (shape and dtype) for `inputs`, `labels` and
      `progress`.
  """
  if arch=='skip_gram': 
    inputs_spec = tf.TensorSpec(shape=(batch_size,), dtype='int64') 
  elif arch == 'cbow':
    inputs_spec = tf.TensorSpec(
        shape=(batch_size, 2*window_size+1), dtype='int64')
  else:
    raise ValueError('`arch` must be either "skip_gram" or "cbow".')

  if algm == 'negative_sampling':
    labels_spec = tf.TensorSpec(shape=(batch_size,), dtype='int64') 
  elif algm == 'hierarchical_softmax':
    labels_spec = tf.TensorSpec(
        shape=(batch_size, 2*max_depth+1), dtype='int64')
  else:
    raise ValueError('`algm` must be either "negative_sampling" or '
        '"hierarchical_softmax".')

  progress_spec = tf.TensorSpec(shape=(batch_size,), dtype='float32')

  train_step_signature = [inputs_spec, labels_spec, progress_spec]
  return train_step_signature
