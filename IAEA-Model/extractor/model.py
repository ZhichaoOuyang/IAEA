#!/usr/bin/env python
# -*- coding:utf-8 -*-

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
      self._graph_mode = 'compute_loss'    # calculate ext_loss
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
    self._art_batch = tf.placeholder(tf.int32, [hps.batch_size, hps.max_art_len, hps.max_sent_len], name='art_batch')   # 3 dimension （batch size，max timesteps of sentence-level encoder, max timesteps of word-level encoder）
    self._art_lens = tf.placeholder(tf.int32, [hps.batch_size], name='art_lens')  # 1 dimension（batch size） Each saved is the length of the sentence of the article
    self._sent_lens = tf.placeholder(tf.int32, [hps.batch_size, hps.max_art_len], name='sent_lens')   # 2 dimension （batch size , Maximum sentence size）  What is stored is how many words in each sentence
    # Dealing with variable length issues
    self._art_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_art_len], name='art_padding_mask') # 2 dimension  （batch size , max timesteps of sentence-level encoder）
    self._sent_padding_mask = tf.placeholder(tf.float32, [hps.batch_size, hps.max_art_len, hps.max_sent_len], name='sent_padding_mask')  # 3 dimension （batch size，max timesteps of sentence-level encoder, max timesteps of word-level encoder）
    # if FLAGS.embedding:
    #   self.embedding_place = tf.placeholder(tf.float32, [self._vocab.size(), hps.emb_dim])
    #add by chao 9.14
    # self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embedding_dim])

    if self._graph_mode == 'compute_loss':
      self._target_batch = tf.placeholder(tf.float32, [hps.batch_size, hps.max_art_len], name='target_batch') # 2 dimension  （batch_size , max timesteps of sentence-level encoder） 1 means this sentence has been extracted
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
    feed_dict[self._art_batch] = batch.art_batch # (batch_size, max_art_len, max_sent_len)    # Each batch, each sentence, the id corresponding to each word
    feed_dict[self._art_lens] = batch.art_lens   # (batch_size, )   # (Sentence length for each batch)
    feed_dict[self._sent_lens] = batch.sent_lens # (batch_size, max_art_len)  # How many words correspond to each sentence in each batch
    feed_dict[self._art_padding_mask] = batch.art_padding_mask # (batch_size, max_art_len)    # The real sentence is 1, and the pad is 0, which are composed of id
    feed_dict[self._sent_padding_mask] = batch.sent_padding_mask # (batch_size, max_art_lens, max_sent_len)   # The real word is 1, the pad is 0 and both are id


    if self._graph_mode == 'compute_loss':
      feed_dict[self._target_batch] = batch.target_batch_selector # (batch_size, max_art_len)   # Sentences that are ground true are 1, and those that are not are 0
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
      encoder_outputs = tf.concat(axis=2, values=encoder_outputs) # concatenate the forwards and backwards states，hidden dimension = word embedding dimension * 2
      # outputs_state = tf.concat(axis=1, values=outputs_state)
    return encoder_outputs,outputs_state    # encoder_outputs is the output at each moment

  def _gru_partial_output(self, input, name):   # Is actually a decoder
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
    glove_path = 'data/Glove/glove.twitter.27B.100d.txt'    # change your own file path
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
          embedding_matrix.append([float(i) for i in line.split()[1:]])   # Vector in Glove Find the vector corresponding to each word

    return {'word_vocab': word_vocab, 'embedding_matrix': np.reshape(embedding_matrix, [-1, dim]).astype(np.float32)}


  def tweet_sequencer(self, sent_feats, art_feats):
    """a logistic layer gets score as to whether the sentence belongs to the summary
       sent_feats: sentence representations, shape (batch_size, max_art_len of this batch, hidden_dim)
       art_feats: article representations, shape (batch_size, hidden_dim)"""
    # First train a scorer and do a simple sort

    hidden_dim = self._hps.hidden_dim_selector
    batch_size = self._hps.batch_size
    # max_art_len = sent_feats.shape[1]

    with tf.variable_scope('Sequencer'):
      w_content = tf.get_variable('w_content', [hidden_dim, 1], dtype=tf.float32, initializer=self.trunc_norm_init)    # Contextual information
      w_salience = tf.get_variable('w_salience', [hidden_dim, hidden_dim], dtype=tf.float32, initializer=self.trunc_norm_init)   # The prominence of the sentence relative to the document
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

      return tf.concat(logits, 1), tf.concat(probs, 1)  # (batch_size, max_art_len)   Spliced into the column


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

      return tf.concat(logits, 1), tf.concat(probs, 1)  # (batch_size, max_art_len)   Spliced into the column


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

  def _add_sent_selector(self):     # Add the whole sequence-to-sequence model to the graph.
    """Add the whole sequence-to-sequence model to the graph."""
    hps = self._hps
    vsize = self._vocab.size() # size of the vocabulary

    with tf.variable_scope('SentSelector'):    # Set the specified scope to share variables
      # Some initializers
      self.rand_unif_init = tf.random_uniform_initializer(-hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)  # Generate uniformly distributed random numbers
      self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)  # Output random values from truncated normal distribution

      ####################################################################
      # Add embedding matrix (shared by the encoder and decoder inputs)  #
      ####################################################################
      with tf.variable_scope('embedding'):    # embedding layer

        if FLAGS.embedding:
          self.embedding_place = self._vocab.getWordEmbedding()
          self.embedding = tf.Variable(self.embedding_place, trainable=False)
          print("embedding load")
          # print(self.embedding_place)
        else:
          self.embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init)

        if hps.mode == "train": self._add_emb_vis(self.embedding) # add to tensorboard
        # Get the embedding of each word of each sentence in each batch
        emb_batch = tf.nn.embedding_lookup(self.embedding, self._art_batch) # _art_batch [hps.batch_size.value, hps.max_art_len.value, hps.max_sent_len.value]   result tensor with shape (batch_size, max_art_len, max_sent_len, emb_size)    # Vectorization converted to embedding

      ########################################
      # Add the two encoders.                #
      ########################################
      # Add word-level encoder to encode each sentence.

      sent_enc_inputs = tf.reshape(emb_batch, [-1, hps.max_sent_len, hps.emb_dim]) # (batch_size*max_art_len, max_sent_len, emb_dim)  It becomes three-dimensional, -1 means the product is automatically calculated. The second two-dimensional is the number of words in the sentence. The third-dimensional is the number of words each word has.

      # This is how many words are used to store each sentence
      sent_lens = tf.reshape(self._sent_lens, [-1]) # (batch_size*max_art_len, )
      # Get the embedding of each sentence at the sentence level, that is, a sentence can be represented by multiple hidden cells, and each hidden cell has dimensions, which should be 128 at present. The embedding of the sentence is represented by (number of words, hidden-dim)

      sent_enc_outputs,sent_outputs_state = self._add_encoder(sent_enc_inputs, sent_lens, name='first_sent_encoder') # (batch_size*max_art_len, max_sent_len, hidden_dim*2)  # GRU for each word in each sentence
      sent_enc_outputs, sent_outputs_state = self._add_encoder(sent_enc_outputs, sent_lens, name='second_sent_encoder')

      # Add sentence-level encoder to produce sentence representations.
      # sentence-level encoder input: average-pooled, concatenated hidden states of the word-level bi-LSTM.

      sent_padding_mask = tf.reshape(self._sent_padding_mask, [-1, hps.max_sent_len, 1]) # (batch_size*max_art_len, max_sent_len, 1)
      #(3200,1)
      sent_lens_float = tf.reduce_sum(sent_padding_mask, axis=1)    # The sum of the data in a row is the sum of the columns (batch_size * max_art_len, 1), and calculate how many words there are in a sentence
      self.sent_lens_float = tf.where(sent_lens_float > 0.0, sent_lens_float, tf.ones(sent_lens_float.get_shape().as_list()))   # Return data that meets the conditions. When the condition is true, take the data corresponding to x; when the condition is false, take the data corresponding to y
      art_enc_inputs = tf.reduce_sum(sent_enc_outputs * sent_padding_mask, axis=1) / self.sent_lens_float # (batch_size*max_art_len, hidden_dim*2)  Add the words and divide by the number of words to get the input of this sentence
      art_enc_inputs = tf.reshape(art_enc_inputs, [hps.batch_size, -1, hps.hidden_dim_selector*2]) # (batch_size, max_art_len, hidden_dim*2)
      # art_enc_outputs,art_outputs_state = self._add_encoder(art_enc_inputs, self._art_lens, name='art_encoder') # (batch_size, max_art_len, hidden_dim*2)    # Sentence output at the document level

      # Get each sentence representation and the document representation.
      sent_feats = tf.contrib.layers.fully_connected(art_enc_inputs, hps.hidden_dim_selector, activation_fn=tf.tanh) # (batch_size, max_art_len, hidden_dim)   # Sentence embedding
      # sent_feats = tf.layers.dropout(sent_feats, rate=FLAGS.dropout_rate, training=FLAGS.is_training)  # dropout when train

      weight_Dense = np.load("data/inconsistent_weigh.npy")  # change your own file path
      weight_Dense_2 = tf.convert_to_tensor(weight_Dense)
      weight_Dense = tf.get_variable("weight_Dense", initializer=weight_Dense_2, trainable=False)
      bias_Dense = np.load("data/inconsistent_bias.npy")  # change your own file path
      bias_Dense_2 = tf.convert_to_tensor(bias_Dense)
      bias_Dense = tf.get_variable("bias_Dense", initializer=bias_Dense_2, trainable=False)
      print((weight_Dense.shape, " ", bias_Dense.shape))

      art_padding_mask = tf.expand_dims(self._art_padding_mask, 2) # (batch_size, max_art_len, 1)    （batch_size , max timesteps of sentence-level encoder）
      art_feats = tf.reduce_sum(sent_feats * art_padding_mask, axis=1) / tf.reduce_sum(art_padding_mask, axis=1) # (batch_size, hidden_dim * 2)    Add sentences to get a simple article representation
      art_feats = tf.contrib.layers.fully_connected(art_feats, hps.hidden_dim_selector, activation_fn=tf.tanh) # (batch_size, hidden_dim)   # The embedding of the document uses a fully connected layer to get the representation of the last article
      # art_feats = tf.layers.dropout(art_feats, rate=FLAGS.dropout_rate, training=FLAGS.is_training)  # dropout when train

      # Get a starting score for each sentence, without rank
      init_logits, self.init_probs = self.tweet_sequencer(sent_feats, art_feats)  # (batch_size, max_art_len)
      self.init_probs = self.init_probs * self._art_padding_mask  # Get the probability of each sentence being selected # (5, 50) The probability of each sentence being selected

      firstTenOldSummary = sent_feats[:, :10, :]
      laterNewTweet = sent_feats[:, 10:hps.max_art_len, :]

      firstTenOldSummary_logist = self.init_probs[:, :10]
      laterNewTweet_logist = self.init_probs[:, 10:hps.max_art_len]

      final_score = self.init_probs
      zeor_tensor = tf.zeros([hps.hidden_dim_selector], dtype=tf.float32)

      def inconsistent(old, new):  # The probability of inconsistency is greater than 0.5

          temp = firstTenOldSummary[:, old, :]  # Save inconsistent old summary information
          part1 = firstTenOldSummary[:, 0:old, :]  # Previous part
          part2 = firstTenOldSummary[:, (old + 1):, :]  # Last part

          # s is the dynamic representation of the summary at the j-th sentence
          s = tf.zeros([hps.batch_size, hps.hidden_dim_selector])
          content = tf.concat([part1, part2], 1)  # (batch,num，200)
          with tf.variable_scope("Sequencer", reuse=True):
              w_content = tf.get_variable('w_content')
              w_salience = tf.get_variable('w_salience')
              w_novelty = tf.get_variable('w_novelty')
              bias = tf.get_variable('bias')
              for i in range(9):
                  content_feats = tf.matmul(content[:, i, :], w_content)  # (, 1)
                  salience_feats = tf.reduce_sum(tf.matmul(content[:, i, :], w_salience) * art_feats,
                                                 1,
                                                 keep_dims=True)  # (, 1)
                  novelty_feats = tf.reduce_sum(tf.matmul(content[:, i, :], w_novelty) * tf.tanh(s), 1,
                                                keep_dims=True)  # (, 1)
                  logit = content_feats + salience_feats - novelty_feats + bias  # (5, 1)
                  # logits.append(logit)
                  p = tf.sigmoid(logit)  # (batch_size, 1)
                  # probs.append(p)
                  s += tf.multiply(content[:, i, :], p)  # s_j
              new_sent = tf.reshape(laterNewTweet[:, new - 10, :], [hps.batch_size, 1, hps.hidden_dim_selector])
              content_feats = tf.matmul(new_sent[:, 0, :], w_content)
              salience_feats = tf.reduce_sum(tf.matmul(new_sent[:, 0, :], w_salience) * art_feats[:, :], 1,
                                              keep_dims=True)
              novelty_feats = tf.reduce_sum(tf.matmul(new_sent[:, 0, :], w_novelty) * tf.tanh(s), 1,
                                            keep_dims=True)  # (, 1)
              logist = salience_feats + content_feats - novelty_feats + bias
              p = tf.sigmoid(logist)  # Probability of whether you need to enter this summary

          mid = tf.reshape(laterNewTweet[:, new - 10, :], [hps.batch_size, 1, hps.hidden_dim_selector])  # Add the new tweet
          # Update the summary list, put the new one in and the old one out
          update_old_summary = tf.concat([part1, mid, part2], axis=1)  # firstTenOldSummary[b] 也需要返回

          # old summary score
          temp_score = firstTenOldSummary_logist[:, old] * (1 - result2[:, 0])

          # Put in the score of the new tweet
          part1 = firstTenOldSummary_logist[:, 0:old]
          part2 = firstTenOldSummary_logist[:, (old + 1):]
          mid = laterNewTweet_logist[:, new - 10] * result2[:, 0] + (tf.ones(1) - result2[:, 0]) * p[:, 0]

          mid = tf.reshape(mid, [hps.batch_size, 1])
          # Update the score of the new tweet to the old summary
          update_old_summary_logist = tf.concat([part1, mid, part2], axis=1)

          update_final_score = []
          for batch in range(hps.batch_size):
              part1 = final_score[batch, 0:summary_index[batch, old]]
              part2 = final_score[batch, (summary_index[batch, old] + 1):]
              update_final_score_batch = tf.concat([part1, tf.reshape(temp_score[batch], [1]), part2], axis=0)
              update_final_score.append(update_final_score_batch)
          update_final_score = tf.reshape(tf.concat(update_final_score, 0), [hps.batch_size, hps.max_art_len])  # Update the score of the old summary to the final score. The old summary can also be the subscript of the new tweet, so write this.
          # Update the index of the old summary
          update_summary_index[:, old] = new

          part1 = update_final_score[:, 0:new]
          part2 = update_final_score[:, (new + 1):]
          update_final_score2 = tf.concat([part1, mid, part2], axis=1)  # Update the score of the new tweet to the final score

          # Set the tweet that entered the old summary in the new tweet to 0, and it will not be traversed to him next time.
          part1 = laterNewTweet[:, 0:new - 10, :]
          part2 = laterNewTweet[:, (new - 10 + 1):, :]
          mid = tf.constant(0, shape=[hps.batch_size, 1, hps.hidden_dim_selector], dtype=tf.float32)
          update_new_tweet = tf.concat([part1, mid, part2], axis=1)

          # Set the tweet score of the new tweet that entered the old summary to 0
          part1 = laterNewTweet_logist[:, 0: new - 10]
          part2 = laterNewTweet_logist[:, (new - 10 + 1):]
          mid = tf.constant(0, shape=[hps.batch_size, 1], dtype=tf.float32)
          # Update the score of the new tweet to the old summary
          update_new_tweet_logist = tf.concat([part1, mid, part2], axis=1)

          # The content in the updated summary, the score in the updated summary, the content in the new tweet after the update (the one is set to 0), the score in the new tweet after the update (the one is set to 0), the final sentence after the update, after the update The score of the final sentence of the current batch
          return {'update_old_summary': update_old_summary, 'update_old_summary_logist': update_old_summary_logist,
                  'update_new_tweet': update_new_tweet, 'update_new_tweet_logist': update_new_tweet_logist
              , 'update_final_score': update_final_score2, 'summary_index': update_summary_index}

      def consistent(old, new):  # Still consistent
          update_old_summary = firstTenOldSummary
          update_old_summary_logist = firstTenOldSummary_logist
          update_new_tweet = laterNewTweet
          update_new_tweet_logist = laterNewTweet_logist
          update_final_score = final_score
          update_summary_index2 = summary_index

          # The content in the updated summary, the score in the updated summary, the content in the new tweet after the update (the one is set to 0), the score in the new tweet after the update (the one is set to 0), the final sentence after the update, after the update The score of the final sentence of the current batch
          return {'update_old_summary': update_old_summary, 'update_old_summary_logist': update_old_summary_logist,
                  'update_new_tweet': update_new_tweet, 'update_new_tweet_logist': update_new_tweet_logist
              , 'update_final_score': update_final_score, 'summary_index': update_summary_index2}
      flag = 0
      index = []
      summary_index = []
      for i in range(10):
          index.append(i)
      for d in range(hps.batch_size):
          summary_index.append(index)
      summary_index = np.array(summary_index).astype(np.int32)
      summary_index = summary_index.reshape(hps.batch_size, 10)
      update_summary_index = summary_index
      update_summary_index2 = summary_index

      for i in range(10, hps.max_art_len):
          for j in range(10):
              new = i
              old = j

              current_oldSummary = firstTenOldSummary[:, j, :]
              current_newTweet = laterNewTweet[:, i - 10, :]
              subed = tf.subtract(current_oldSummary, current_newTweet)
              result2 = (tf.nn.sigmoid(tf.matmul(subed, weight_Dense) + bias_Dense, name="inconsistent"))
              current_oldSummary_logist = firstTenOldSummary_logist[:, j]
              current_newTweet_logist = laterNewTweet_logist[:, i - 10]
              aeqb = tf.equal(current_newTweet, zeor_tensor)
              aeqb_int = tf.to_int32(aeqb)

              dict_list_inconsistent = inconsistent(old, new)
              dict_list_consistent = consistent(old, new)

              firstTenOldSummary = tf.where( (tf.greater_equal(result2[:, 0], 0.5)) & (
                  tf.not_equal(tf.reduce_sum(aeqb_int), tf.reduce_sum(tf.ones_like(aeqb_int)))),
                                            dict_list_inconsistent['update_old_summary'],
                                            dict_list_consistent['update_old_summary'])
              laterNewTweet = tf.where((tf.greater_equal(result2[:, 0], 0.5)) & (
                  tf.not_equal(tf.reduce_sum(aeqb_int), tf.reduce_sum(tf.ones_like(aeqb_int)))),
                                       dict_list_inconsistent['update_new_tweet'],
                                       dict_list_consistent['update_new_tweet'])
              firstTenOldSummary_logist = tf.where((tf.greater_equal(result2[:, 0], 0.5)) & (
                  tf.not_equal(tf.reduce_sum(aeqb_int), tf.reduce_sum(tf.ones_like(aeqb_int)))),
                                                   dict_list_inconsistent['update_old_summary_logist'],
                                                   dict_list_consistent['update_old_summary_logist'])
              laterNewTweet_logist = tf.where((tf.greater_equal(result2[:, 0], 0.5)) & (
                  tf.not_equal(tf.reduce_sum(aeqb_int), tf.reduce_sum(tf.ones_like(aeqb_int)))),
                                              dict_list_inconsistent['update_new_tweet_logist'],
                                              dict_list_consistent['update_new_tweet_logist'])
              final_score = tf.where((tf.greater_equal(result2[:, 0], 0.5)) & (
                  tf.not_equal(tf.reduce_sum(aeqb_int), tf.reduce_sum(tf.ones_like(aeqb_int)))),
                                     dict_list_inconsistent['update_final_score'],
                                     dict_list_consistent['update_final_score'])
              summary_index = tf.where((tf.greater_equal(result2[:, 0], 0.5)) & (
                  tf.not_equal(tf.reduce_sum(aeqb_int), tf.reduce_sum(tf.ones_like(aeqb_int)))),
                                       dict_list_inconsistent['summary_index'], dict_list_consistent['summary_index'])

              flag += 1
      logists = final_score
      print(logists)
      print("inconsistent end")
      self.probs = tf.sigmoid(logists)
      print(self.probs)
      self.probs = self.probs * self._art_padding_mask  # （5，50）  The probability of each sentence being extracted
      self.avg_prob = tf.reduce_mean(tf.reduce_sum(self.probs, 1) / tf.reduce_sum(self._art_padding_mask,
                                                                                  1))
      tf.summary.scalar('avg_prob', self.avg_prob)

      ################################################
      # Calculate the loss                           #
      ################################################
      if self._graph_mode == 'compute_loss':
        with tf.variable_scope('loss'):
          losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logists, labels=self._target_batch) # (batch_size, max_art_len)
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


  def build_graph(self):    # selector graph
    """Add the placeholders, model, global step, train_op and summaries to the graph"""
    tf.logging.info('Building graph...')
    t0 = time.time()
    self._add_placeholders()    # Setting of placeholder variables
    # with tf.device("/gpu:3"):
    self._add_sent_selector()   # Specific process
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

