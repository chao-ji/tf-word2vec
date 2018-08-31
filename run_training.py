import tensorflow as tf
from datetime import datetime
from dataset import Word2VecDataset
from word2vec import Word2VecModel

arch = 'skip_gram'
algm = 'negative_sampling'
batch_size = 256
max_vocab_size = None
min_count = 10
sample = 1e-3
window_size = 10
filenames = ['/home/chaoji/Desktop/sents.txt']

dataset = Word2VecDataset(arch=arch,
                          algm=algm,
                          batch_size=batch_size,
                          max_vocab_size=max_vocab_size,
                          min_count=min_count,
                          sample=sample,
                          window_size=window_size)
dataset.build_vocab(filenames)

num_steps = dataset.estimate_num_steps(5)
print(num_steps)
num_steps = 50000

word2vec = Word2VecModel(arch=arch,
                         algm=algm,
                         embed_size=300,
                         batch_size=batch_size,
                         negatives=5,
                         power=0.75,
                         alpha=0.025,
                         min_alpha=0.0001,
                         num_steps=num_steps,
                         add_bias=True,
                         random_seed=0)

to_be_run_dict = word2vec.train(dataset, filenames)

sess = tf.Session()

sess.run(dataset.iterator_initializer)
sess.run(tf.tables_initializer())
sess.run(tf.global_variables_initializer())

average_loss = 0.
t0 = datetime.now()
for i in range(num_steps):
  result_dict = sess.run(to_be_run_dict)
  average_loss += result_dict['loss'].mean()
  if i % 10000 == 0:
    if i > 0:
      average_loss /= 10000
    print('i = ', i, 'average_loss =', average_loss, 'learning_rate =', result_dict['learning_rate'])
    average_loss = 0.
print(str(datetime.now() - t0))

syn0_final = sess.run(word2vec.syn0)

