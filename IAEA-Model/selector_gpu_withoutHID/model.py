# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import sys
import time
import numpy as np
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector
FLAGS = tf.app.flags.FLAGS

class SentenceSelector(object):
  """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

  def __init__(self, hps, vocab):
    self._hps = hps
    self._vocab = vocab

    # There are 2 graph mode: ['comput_loss', 'not_compute_loss']
    if hps.mode == 'train':
      self._graph_mode = 'compute_loss'    # 计算ext_loss
    elif hps.mode == 'eval':
      if hps.model == 'end2end':
        if hps.eval_method == 'rouge':
          self._graph_mode = 'not_compute_loss'
        else:
          self._graph_mode = 'compute_loss'
      elif hps.model == 'selector':
        self._graph_mode = 'compute_loss'
    elif hps.mode == 'evalall':
      self._graph_mode = 'not_compute_loss'
    # self.vocab, self.embed = self.load_Glove()
    # self.vocab_size = len(self.vocab)
    # self.embedding_dim = len(self.embed[0])

  def _add_placeholders(self):
    """Add placeholders to the graph. These are entry points for any input data."""
    hps = self._hps
    self._art_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_art_len, hps.max_sent_len], name='art_batch')   # 3维 （批次大小，max timesteps of sentence-level encoder max timesteps of word-level encoder）
    self._art_lens = tf.placeholder(tf.int32, [hps.batch_size], name='art_lens')  # 1维 （批次大小） 每个存的就是文章句子的长度是多少，
    self._sent_lens = tf.placeholder(tf.int32, [hps.batch_size, hps.max_art_len], name='sent_lens')   # 2维 （批次大小 , 最大能容纳的句子大小）  存的是每个句子的词有多少
    # 处理不定长问题
    self._art_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_art_len], name='art_padding_mask') # 2维  （批次大小 , max timesteps of sentence-level encoder）  1代表句子是原来就有的
    self._sent_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_art_len, hps.max_sent_len], name='sent_padding_mask')  # 3维 （批次大小，max timesteps of sentence-level encoder max timesteps of word-level encoder） 1代表词是原先就有的
    # if FLAGS.embedding:
    #   self.embedding_place = tf.placeholder(tf.float32, [self._vocab.size(), hps.emb_dim])
    #add by chao 9.14
    # self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embedding_dim])

    if self._graph_mode == 'compute_loss':
      self._target_batch = tf.placeholder(tf.float32, [hps.batch_size, hps.max_art_len], name='target_batch') # 2维  （批次大小 , max timesteps of sentence-level encoder） 1代表这个句子被提出来了
      # self._target_batch_our = tf.placeholder(tf.float32, [hps.max_art_len.value, hps.batch_size.value, hps.max_art_len.value], name = 'target_batch_our')


  def _make_feed_dict(self, batch, just_enc=False):
    """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.
       max_art_len means maximum number of sentences of one article.
       max_sent_len means maximum number of words of one article sentence.

    Args:
      batch: Batch object
    """
    hps = self._hps
    feed_dict = {}
    feed_dict[self._art_batch] = batch.art_batch # (batch_size, max_art_len, max_sent_len)    # 每次批次每句话 每个词对应的id
    feed_dict[self._art_lens] = batch.art_lens   # (batch_size, )   # (每个批次的句子长度)
    feed_dict[self._sent_lens] = batch.sent_lens # (batch_size, max_art_len)  # 每个批次每个句子对应的词有几个
    feed_dict[self._art_padding_mask] = batch.art_padding_mask # (batch_size, max_art_len)    # 真实的句子是1，pad的是0 都是id组成
    feed_dict[self._sent_padding_mask] = batch.sent_padding_mask # (batch_size, max_art_lens, max_sent_len)   # 真实的词是1，pad是0 都是id

    # add by chao 9.14
    # embedding = np.asarray(self.embed)
    # feed_dict[self.embedding_placeholder] = embedding


    if self._graph_mode == 'compute_loss':
      feed_dict[self._target_batch] = batch.target_batch_selector # (batch_size, max_art_len)   # 是ground true的句子为1，不是的为0
      # feed_dict[self._target_batch_our] = batch.target_batch_selector
    return feed_dict


  def _add_encoder(self, encoder_inputs, seq_len, name, dropout_keep_prob=None):
    """Add a single-layer bidirectional GRU encoder to the graph.

    Args:
      encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
      seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

    Returns:
      encoder_outputs:
        A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
    """
    with tf.variable_scope(name):
      cell_fw = tf.contrib.rnn.GRUCell(self._hps.hidden_dim_selector)    # GRU  hidden_dim_selector = 200
      cell_bw = tf.contrib.rnn.GRUCell(self._hps.hidden_dim_selector)
      (encoder_outputs, outputs_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
      encoder_outputs = tf.concat(axis=2, values=encoder_outputs) # concatenate the forwards and backwards states，hidden维度是word embedding维度 * 2
      # outputs_state = tf.concat(axis=1, values=outputs_state)
    return encoder_outputs,outputs_state    # encoder_outputs是每个时刻的输出

  def _gru_partial_output(self, input, name):   # 实际是一个Decoder
    """
      add a gru to remember the partial output summary
    :param input:
    :param name:
    :return:
    """

    with tf.variable_scope(name):
      gru_cells = tf.nn.rnn_cell.GRUCell(self._hps.hidden_dim_selector)
      init_state = gru_cells.zero_state()
      (encoder_outputs, outputs_state) = tf.nn.dynamic_rnn(gru_cells, input, dtype=tf.float32, swap_memory=True,init_state=init_state)
      return encoder_outputs, outputs_state

  def word_embedding_matrix(self, vocab, dim):
    # first and second vector are pad and unk words
    glove_path = 'data/Glove/glove.twitter.27B.100d.txt'
    with open(glove_path, 'r', encoding="utf-8") as f:
      word_vocab = []
      embedding_matrix = []
      word_vocab.extend(['UNK', 'PAD', 'START', 'STOP'])
      embedding_matrix.append(np.random.uniform(-1.0, 1.0, (1, dim))[0])
      embedding_matrix.append(np.random.uniform(-1.0, 1.0, (1, dim))[0])
      embedding_matrix.append(np.random.uniform(-1.0, 1.0, (1, dim))[0])
      embedding_matrix.append(np.random.uniform(-1.0, 1.0, (1, dim))[0])

      for line in f:
        if line.split()[0] in vocab._word_to_id:    # word_to_id
          word_vocab.append(line.split()[0])
          embedding_matrix.append([float(i) for i in line.split()[1:]])   # Glove里的向量 找到每个word对应的向量

    return {'word_vocab': word_vocab, 'embedding_matrix': np.reshape(embedding_matrix, [-1, dim]).astype(np.float32)}


  def tweet_sequencer(self, sent_feats, art_feats):
    """a logistic layer gets score as to whether the sentence belongs to the summary
       sent_feats: sentence representations, shape (batch_size, max_art_len of this batch, hidden_dim)
       art_feats: article representations, shape (batch_size, hidden_dim)"""
    # 先训练一个打分器，做一个简单的排序

    hidden_dim = self._hps.hidden_dim_selector
    batch_size = self._hps.batch_size
    # max_art_len = sent_feats.shape[1]

    with tf.variable_scope('Sequencer'):
      w_content = tf.get_variable('w_content', [hidden_dim, 1], dtype=tf.float32, initializer=self.trunc_norm_init)    # 句子带的上下文信息
      w_salience = tf.get_variable('w_salience', [hidden_dim, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)   # 句子相当于文档的突出性
      w_novelty = tf.get_variable('w_novelty', [hidden_dim, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias = tf.get_variable('bias', [1], dtype=tf.float32, initializer=tf.zeros_initializer())

      # s is the dynamic representation of the summary at the j-th sentence
      s = tf.zeros([batch_size, hidden_dim])

      logits = []    # logits before the sigmoid layer
      probs = []

      for i in range(self._hps.max_art_len):
        content_feats = tf.matmul(sent_feats[:, i, :], w_content) # (batch_size, 1)
        salience_feats = tf.reduce_sum(tf.matmul(sent_feats[:, i, :], w_salience) * art_feats, 1, keep_dims=True) # (batch_size, 1)
        novelty_feats = tf.reduce_sum(tf.matmul(sent_feats[:, i, :], w_novelty) * tf.tanh(s), 1, keep_dims=True) # (batch_size, 1)
        logit = content_feats + salience_feats - novelty_feats + bias # (batch_size, 1)
        logits.append(logit)

        p = tf.sigmoid(logit) # (batch_size, 1)
        probs.append(p)
        s += tf.multiply(sent_feats[:, i, :], p)

      return tf.concat(logits, 1), tf.concat(probs, 1)  # (batch_size, max_art_len)   拼接到列里面




  def _add_classifier(self, sent_feats, art_feats):
    """a logistic layer makes a binary decision as to whether the sentence belongs to the summary
       sent_feats: sentence representations, shape (batch_size, max_art_len of this batch, hidden_dim)
       art_feats: article representations, shape (batch_size, hidden_dim)"""
    hidden_dim = self._hps.hidden_dim_selector
    batch_size = self._hps.batch_size
    #max_art_len = sent_feats.shape[1]

    with tf.variable_scope('classifier'):
      w_content = tf.get_variable('w_content', [hidden_dim, 1], dtype=tf.float32, initializer=self.trunc_norm_init)
      w_salience = tf.get_variable('w_salience', [hidden_dim, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      w_novelty = tf.get_variable('w_novelty', [hidden_dim, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
      bias = tf.get_variable('bias', [1], dtype=tf.float32, initializer=tf.zeros_initializer())

      # s is the dynamic representation of the summary at the j-th sentence
      s = tf.zeros([batch_size, hidden_dim])

      logits = [] # logits before the sigmoid layer
      probs = []

      for i in range(self._hps.max_art_len):
        content_feats = tf.matmul(sent_feats[:, i, :], w_content) # (batch_size, 1)
        salience_feats = tf.reduce_sum(tf.matmul(sent_feats[:, i, :], w_salience) * art_feats, 1, keep_dims=True) # (batch_size, 1)
        novelty_feats = tf.reduce_sum(tf.matmul(sent_feats[:, i, :], w_novelty) * tf.tanh(s), 1, keep_dims=True) # (batch_size, 1)
        logit = content_feats + salience_feats - novelty_feats + bias # (batch_size, 1)
        logits.append(logit)

        p = tf.sigmoid(logit) # (batch_size, 1)
        probs.append(p)
        s += tf.multiply(sent_feats[:, i, :], p)

      return tf.concat(logits, 1), tf.concat(probs, 1)  # (batch_size, max_art_len)   拼接到列里面


  def _add_emb_vis(self, embedding_var):
    """Do setup so that we can view word embedding visualization in Tensorboard, as described here:
    https://www.tensorflow.org/get_started/embedding_viz
    Make the vocab metadata file, then make the projector config file pointing to it."""
    train_dir = os.path.join(FLAGS.log_root, "train")
    vocab_metadata_path = os.path.join(train_dir, "vocab_metadata.tsv")
    self._vocab.write_metadata(vocab_metadata_path) # write metadata file
    summary_writer = tf.summary.FileWriter(train_dir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = vocab_metadata_path
    projector.visualize_embeddings(summary_writer, config)

  # def collect_final_step_of_gru(gru_representation, lengths):
  #   # lstm_representation: [batch_size, passsage_length, dim]
  #   # lengths: [batch_size]
  #   lengths = tf.maximum(lengths, tf.zeros_like(lengths, dtype=tf.int32))
  #
  #   batch_size = tf.shape(lengths)[0]   # 获取bach_size的数目
  #   batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
  #   indices = tf.stack((batch_nums, lengths), axis=1)  # shape (batch_size, 2)
  #   result = tf.gather_nd(gru_representation, indices, name='last-backward-gru')
  #   return result  # [batch_size, dim]

  def _add_sent_selector(self):     # Add the whole sequence-to-sequence model to the graph.
    """Add the whole sequence-to-sequence model to the graph."""
    hps = self._hps
    vsize = self._vocab.size() # size of the vocabulary
    # symbols = {0: 'PAD', 1: 'UNK', 2:'START', 3:'STOP'}
    #vocab = self._vocab
    #matrix = self.word_embedding_matrix(vocab, 100)   # 加载GLOVE词向量
    # load_embedding_matrix = matrix['embedding_matrix']
    # shape_word_vocab = matrix['word_vocab']
    # int_to_vocab = {}
    # for index_no, word in enumerate(shape_word_vocab,start=4):
    #   int_to_vocab[index_no] = word
    # int_to_vocab.update(symbols)
    # vocab_to_int = {word: index_no for index_no, word in int_to_vocab.items()}
    # # encoded_data = []

    with tf.variable_scope('SentSelector'):    # 设置指定作用域来共享变量
      # Some initializers
      self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)  # 生成均匀分布的随机数
      self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)  # 从截断的正态分布中输出随机值

      ####################################################################
      # Add embedding matrix (shared by the encoder and decoder inputs)  #
      ####################################################################
      with tf.variable_scope('embedding'):    # embedding 层

        if FLAGS.embedding:
          self.embedding_place = self._vocab.getWordEmbedding()
          self.embedding = tf.Variable(self.embedding_place, trainable=False)
          print("embedding load")
          # print(self.embedding_place)
        else:
          self.embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

        # self.embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32,
        #                                   initializer = tf.constant_initializer(np.array(load_embedding_matrix)),
        #                                  trainable=False)    # variable name "embedding"  2维 [词汇集大小，编码维度大小]
        # 暂时先不把它改成带Glove的,（已经改成Glove）
        if hps.mode == "train": self._add_emb_vis(self.embedding) # add to tensorboard
        # 得到每个批次每个句子每个词的embedding
        emb_batch = tf.nn.embedding_lookup(self.embedding, self._art_batch) # _art_batch [hps.batch_size.value, hps.max_art_len.value, hps.max_sent_len.value] 结果tensor with shape (batch_size, max_art_len, max_sent_len, emb_size)  四维   # 转换成embedding的向量化

      ########################################
      # Add the two encoders.                #
      ########################################
      # Add word-level encoder to encode each sentence.
      #(3200,50,128),形状转化,第一个编码器的输入序列

      sent_enc_inputs = tf.reshape(emb_batch, [-1, hps.max_sent_len, hps.emb_dim]) # (batch_size*max_art_len, max_sent_len, emb_dim)  变成三维，-1表示自动计算乘积  第二维是句子拥有的单词的数量 第三维是每个单词拥有的维度
      #(3200)
      # 3200一个批次的句子长度 相当于一个topic的句子长度，因为batch_size = 5
      # 这个是拿来存每个句子的单词有多少个的
      sent_lens = tf.reshape(self._sent_lens, [-1]) # (batch_size*max_art_len, )
      # 得到每个句子在句子级别的embedding，就是一个句子可以用好多个hidden cell来表示，每个hidden都是有维度的，目前应该是128. 句子的embedding用（单词数，hidden-dim）表示
      #(3200,50,200*2)
      sent_enc_outputs,sent_outputs_state = self._add_encoder(sent_enc_inputs, sent_lens, name='first_sent_encoder') # 第一个(batch_size*max_art_len, max_sent_len, hidden_dim*2)  # 每个句子中每个词的GRU
      sent_enc_outputs, sent_outputs_state = self._add_encoder(sent_enc_outputs, sent_lens, name='second_sent_encoder')
      # state_backward = sent_outputs_state[1][-1][-1]
      # add by chao 9.12
      # art_enc_outputs = self._add_encoder(sent_enc_outputs, sent_lens, name='art_encoder')    # 第二个参数代表序列，这里还是句子的大小 # (batch_size*max_art_len, max_sent_len, hidden_dim*4)

      # Add sentence-level encoder to produce sentence representations.
      # sentence-level encoder input: average-pooled, concatenated hidden states of the word-level bi-LSTM.
      #(3200,50,1)
      sent_padding_mask = tf.reshape(self._sent_padding_mask, [-1, hps.max_sent_len, 1]) # (batch_size*max_art_len, max_sent_len, 1)  # 原来(batch_size, max_art_lens, max_sent_len) 自带词是1
      #(3200,1)
      sent_lens_float = tf.reduce_sum(sent_padding_mask, axis=1)    # 一行的数据求和，就是各列求和其实（batch_size * max_art_len，1），算出一句话具体有几个词
      self.sent_lens_float = tf.where(sent_lens_float > 0.0, sent_lens_float, tf.ones(sent_lens_float.get_shape().as_list()))   #    # 返回符合条件的数据。当条件为真，取x对应的数据；当条件为假，取y对应的数据
      art_enc_inputs = tf.reduce_sum(sent_enc_outputs * sent_padding_mask, axis=1) / self.sent_lens_float # (batch_size*max_art_len, hidden_dim*2)  词相加除以词的个数，得出这个句子的输入
      art_enc_inputs = tf.reshape(art_enc_inputs, [hps.batch_size, -1, hps.hidden_dim_selector*2]) # (batch_size, max_art_len, hidden_dim*2)
      # art_enc_outputs,art_outputs_state = self._add_encoder(art_enc_inputs, self._art_lens, name='art_encoder') # (batch_size, max_art_len, hidden_dim*2)    # 句子在文档级的输出

      # Get each sentence representation and the document representation.  不需要document representation. 需要doucment级别的sentence representation
      sent_feats = tf.contrib.layers.fully_connected(art_enc_inputs, hps.hidden_dim_selector, activation_fn=tf.tanh) # (batch_size, max_art_len, hidden_dim)   # 句子的embedding

      art_padding_mask = tf.expand_dims(self._art_padding_mask, 2) # (batch_size, max_art_len, 1)    （批次大小 , max timesteps of sentence-level encoder）
      art_feats = tf.reduce_sum(sent_feats * art_padding_mask, axis=1) / tf.reduce_sum(art_padding_mask, axis=1) # (batch_size, hidden_dim * 2)    句子相加得到一个简单的文章表示
      art_feats = tf.contrib.layers.fully_connected(art_feats, hps.hidden_dim_selector, activation_fn=tf.tanh) # (batch_size, hidden_dim)   # 文档的embedding 在用一个全连接层得到最后 一个文章的表示
      # art_feats = tf.layers.dropout(art_feats, rate=FLAGS.dropout_rate, training=FLAGS.is_training)  # dropout when train

      # 得到一个开始对每个句子的打分，不用rank
      logits, self.probs = self.tweet_sequencer(sent_feats, art_feats)  # (batch_size, max_art_len)
      self.probs = self.probs * self._art_padding_mask  # 得到每个句子被抽到的概率  # （5，50）  每个句子被抽到的概率
      print(logits)
      print("classify end")
      print(self.probs)
      self.probs = self.probs * self._art_padding_mask  # （5，50）  每个句子被抽到的概率
      self.avg_prob = tf.reduce_mean(tf.reduce_sum(self.probs, 1) / tf.reduce_sum(self._art_padding_mask,
                                                                                  1))  # 分母是当前批次的句子个数 mean之前是(5,)，求出每个句子选到的平均概率
      tf.summary.scalar('avg_prob', self.avg_prob)

      ########################################
      # Add the classifier.                  #
      ########################################
      # logits, self.probs = self._add_classifier(sent_feats, art_feats) # (batch_size, max_art_len)
      # self.probs = self.probs * self._art_padding_mask   # （5，50）  每个句子被抽到的概率
      # self.avg_prob = tf.reduce_mean(tf.reduce_sum(self.probs, 1) / tf.reduce_sum(self._art_padding_mask, 1))   # 分母是当前批次的句子个数 mean之前是(5,)，求出每个句子选到的平均概率
      # tf.summary.scalar('avg_prob', self.avg_prob)

      ################################################
      # Calculate the loss                           #
      ################################################
      if self._graph_mode == 'compute_loss':
        with tf.variable_scope('loss'):
          losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self._target_batch) # (batch_size, max_art_len)  # 这个操作的输入logits是未经缩放的
          loss = tf.reduce_sum(losses * self._art_padding_mask, 1) / tf.reduce_sum(self._art_padding_mask, 1) # (batch_size,)
          self._loss = tf.reduce_mean(loss)
          tf.summary.scalar('loss', self._loss)


  def _add_train_op(self):
    """Sets self._train_op, the op to run for training."""
    # Take gradients of the trainable variables w.r.t. the loss function to minimize
    hps = self._hps
    tvars = tf.trainable_variables()
    loss_to_minimize = self._loss
    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    # Clip the gradients
    # with tf.device("/gpu:3"):
    grads, global_norm = tf.clip_by_global_norm(gradients, hps.max_grad_norm)

    # Add a summary
    tf.summary.scalar('global_norm', global_norm)

    # Apply adagrad optimizer
    optimizer = tf.train.AdagradOptimizer(hps.lr, initial_accumulator_value=hps.adagrad_init_acc)
    # with tf.device("/gpu:3"):
    self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')


  def build_graph(self):    # selector的图结构
    """Add the placeholders, model, global step, train_op and summaries to the graph"""
    tf.logging.info('Building graph...')
    t0 = time.time()
    self._add_placeholders()    # 占位变量的设置
    # with tf.device("/gpu:3"):
    self._add_sent_selector()   # 具体的过程
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self._hps.mode == 'train':
      self._add_train_op()
    self._summaries = tf.summary.merge_all()
    t1 = time.time()
    tf.logging.info('Time to build graph: %i seconds', t1 - t0)


  def run_train_step(self, sess, batch):
    """This function will only be called when hps.model == selector
       Runs one training iteration. Returns a dictionary containing train op,
       summaries, loss, probs and global_step."""
    hps = self._hps
    feed_dict = self._make_feed_dict(batch)

    to_return = {
        'train_op': self._train_op,
        'summaries': self._summaries,
        'loss': self._loss,
        'probs': self.probs,
        'global_step': self.global_step,
    }
    return sess.run(to_return, feed_dict)

  def run_eval_step(self, sess, batch, probs_only=False):
    """This function will be called when hps.model == selector or end2end
       Runs one evaluation iteration. Returns a dictionary containing summaries,
       loss, global_step and (optionally) probs.

       probs_only: when evaluating the selector, only output the sent probs
    """
    hps = self._hps
    feed_dict = self._make_feed_dict(batch)
    to_return = {'probs': self.probs}

    if not probs_only:
      to_return['summaries'] = self._summaries
      to_return['loss'] = self._loss
      to_return['global_step'] = self.global_step

    return sess.run(to_return, feed_dict)

