import os
import sys
import time
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
from tensorflow.contrib.layers import batch_norm
class HID(object):
    """A RNN for text classification."""

    def __init__(
            self, sequence_length, num_classes, vocab_size, lstm_hidden_size, fc_hidden_size,
            embedding_size, embedding_type, l2_reg_lambda=0.0, pretrained_embedding=None):
        self.sequence_length = sequence_length
        # Placeholders for input, output, dropout_prob and training_tag
        self.input_x_front = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_front")
        self.input_x_behind = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_behind")
        self._sent_padding_mask_front = tf.placeholder(tf.float32, [None, sequence_length], name='sent_padding_mask_front')
        self._sent_padding_mask_behind = tf.placeholder(tf.float32, [None, sequence_length],
                                                       name='sent_padding_mask_behind')
        self.front_lens = tf.placeholder(tf.int32, [None], name = "front_lens")
        self.behind_lens = tf.placeholder(tf.int32, [None], name = "front_lens")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.l2_reg_lambda = l2_reg_lambda

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

    def _add_encoder(self, encoder_inputs, seq_len, name):
        """Add a single-layer bidirectional GRU encoder to the graph.

        Args:
          encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
          seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

        Returns:
          encoder_outputs:
            A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
        """
        with tf.variable_scope(name):
            cell_fw = tf.contrib.rnn.GRUCell(self._hps.hidden_dim_selector)  # GRU  hidden_dim_selector = 200
            cell_bw = tf.contrib.rnn.GRUCell(self._hps.hidden_dim_selector)
            if self.dropout_keep_prob is not None:
                cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=self.dropout_keep_prob)
                cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=self.dropout_keep_prob)
            (encoder_outputs, outputs_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs,
                                                                               dtype=tf.float32,
                                                                               sequence_length=seq_len,
                                                                               swap_memory=True)
            encoder_outputs = tf.concat(axis=2,
                                        values=encoder_outputs)  # concatenate the forwards and backwards states，hidden维度是word embedding维度 * 2
            # outputs_state = tf.concat(axis=1, values=outputs_state)
        return encoder_outputs, outputs_state

    def _add_encoder_art(self, encoder_inputs,name):
        """Add a single-layer bidirectional GRU encoder to the graph.

        Args:
          encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
          seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

        Returns:
          encoder_outputs:
            A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
        """
        with tf.variable_scope(name):
            cell_fw = tf.contrib.rnn.GRUCell(self._hps.hidden_dim_selector)  # GRU  hidden_dim_selector = 200
            cell_bw = tf.contrib.rnn.GRUCell(self._hps.hidden_dim_selector)

            (encoder_outputs, outputs_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, encoder_inputs,
                                                                               dtype=tf.float32,
                                                                               swap_memory=True)
            encoder_outputs = tf.concat(axis=2,
                                        values=encoder_outputs)  # concatenate the forwards and backwards states，hidden维度是word embedding维度 * 2
            # outputs_state = tf.concat(axis=1, values=outputs_state)
        return encoder_outputs, outputs_state

    def _add_incosistent(self):
        def _linear(input_, output_size, scope="SimpleLinear"):
            """
            Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
            Args:
                input_: a tensor or a list of 2D, batch x n, Tensors.
                output_size: int, second dimension of W[i].
                scope: VariableScope for the created subgraph; defaults to "SimpleLinear".
            Returns:
                A 2D Tensor with shape [batch x output_size] equal to
                sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
            Raises:
                ValueError: if some of the arguments has unspecified or wrong shape.
            """

            shape = input_.get_shape().as_list()
            if len(shape) != 2:
                raise ValueError("Linear is expecting 2D arguments: {0}".format(str(shape)))
            if not shape[1]:
                raise ValueError("Linear expects shape[1] of arguments: {0}".format(str(shape)))
            input_size = shape[1]

            # Now the computation.
            with tf.variable_scope(scope):
                W = tf.get_variable("W", [input_size, output_size], dtype=input_.dtype)
                b = tf.get_variable("b", [output_size], dtype=input_.dtype)

            return tf.nn.xw_plus_b(input_, W, b)

        def _highway_layer(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu):
            """
            Highway Network (cf. http://arxiv.org/abs/1505.00387).
            t = sigmoid(Wy + b)
            z = t * g(Wy + b) + (1 - t) * y
            where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
            """

            for idx in range(num_layers):
                g = f(_linear(input_, size, scope=("highway_lin_{0}".format(idx))))
                t = tf.sigmoid(_linear(input_, size, scope=("highway_gate_{0}".format(idx))) + bias)
                output = t * g + (1. - t) * input_
                input_ = output

            return output

        hps = self._hps
        vsize = self._vocab.size() # size of the vocabulary
        # Embedding Layer
        with tf.name_scope("embedding"):
            # Use random generated the word vector by default
            # Can also be obtained through our own word vectors trained by our corpus
            if FLAGS.embedding:
                self.embedding = tf.constant(self.embedding_place, dtype=tf.float32, name="embedding")
                print("embedding load")
                print(self.embedding_place)
            else:
                self.embedding = tf.Variable(tf.random_uniform([vsize, hps.emb_dim], minval=-1.0, maxval=1.0,
                                                               dtype=tf.float32), trainable=True, name="embedding")
            self.embedded_sentence_front = tf.nn.embedding_lookup(self.embedding, self.input_x_front)
            self.embedded_sentence_behind = tf.nn.embedding_lookup(self.embedding, self.input_x_behind)

        # Bi-gru Layer
        with tf.name_scope("Bi-gru"):

            outputs_sentence_front, state_sentence_front = self._add_encoder(self.embedded_sentence_front,self.front_lens, name="sent_encoder_front")
            outputs_sentence_behind, state_sentence_behind = self._add_encoder(self.embedded_sentence_behind,self.behind_lens, name="sent_encoder_behind")
            sent_padding_mask_front = tf.reshape(self._sent_padding_mask_front, [-1, self.sequence_length, 1])
            sent_padding_mask_behind = tf.reshape(self._sent_padding_mask_behind, [-1, self.sequence_length, 1])
            sent_lens_float_front = tf.reduce_sum(sent_padding_mask_front, axis=1)
            sent_lens_float_behind = tf.reduce_sum(sent_padding_mask_behind, axis=1)
            self.sent_lens_float_front = tf.where(sent_lens_float_front > 0.0, sent_lens_float_front,
                                            tf.ones(sent_lens_float_front.get_shape().as_list()))
            art_enc_inputs_front = tf.reduce_sum(outputs_sentence_front * sent_padding_mask_front, axis=1) / self.sent_lens_float_front
            art_enc_outputs_front,art_outputs_state_front = self._add_encoder_art(art_enc_inputs_front, name='art_encoder_front')

            self.sent_lens_float_behind = tf.where(sent_lens_float_behind > 0.0, sent_lens_float_behind,
                                            tf.ones(sent_lens_float_behind.get_shape().as_list()))
            art_enc_inputs_behind = tf.reduce_sum(sent_lens_float_behind * sent_padding_mask_behind, axis=1) / self.sent_lens_float_behind
            art_enc_outputs_behind, art_outputs_state_behind = self._add_encoder_art(art_enc_inputs_behind, name='art_encoder_behind')

            self.sent_feats_front = tf.contrib.layers.fully_connected(art_enc_outputs_front, hps.hidden_dim_selector,
                                                           activation_fn=tf.tanh)  # (batch_size, max_art_len, hidden_dim)   # 句子的embedding

            self.sent_feats_behind = tf.contrib.layers.fully_connected(art_enc_outputs_behind, hps.hidden_dim_selector,
                                                           activation_fn=tf.tanh)  # (batch_size, max_art_len, hidden_dim)   # 句子的embedding

            self.gru_out_sub = tf.keras.layers.subtract([self.sent_feats_front, self.sent_feats_behind])


        # Fully Connected Layer
        with tf.name_scope("fc"):
            W = tf.Variable(tf.truncated_normal(shape=[hps.hidden_dim_selector, hps.hidden_dim_selector],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(value=0.1, shape=[hps.hidden_dim_selector], dtype=tf.float32), name="b")
            self.fc = tf.nn.xw_plus_b(self.gru_out_sub, W, b)

            # Batch Normalization Layer
            self.fc_bn = batch_norm(self.fc, is_training=self.is_training, trainable=True, updates_collections=None)

            # Apply nonlinearity
            self.fc_out = tf.nn.relu(self.fc_bn, name="relu")

        # Highway Layer
        with tf.name_scope("highway"):
            self.highway = _highway_layer(self.fc_out, self.fc_out.get_shape()[1], num_layers=1, bias=0)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.highway, self.dropout_keep_prob)

        # Final scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal(shape=[hps.hidden_dim_selector, 2],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(value=0.1, shape=[2], dtype=tf.float32), name="b")
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.softmax_scores = tf.nn.softmax(self.logits, name="softmax_scores")
            self.predictions = tf.argmax(self.logits, 2, name="predictions")
            self.topKPreds = tf.nn.top_k(self.softmax_scores, k=1, sorted=True, name="topKPreds")

        # Calculate mean cross-entropy loss, L2 loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            losses = tf.reduce_mean(losses, name="softmax_losses")
            l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
                                 name="l2_losses") * self.l2_reg_lambda
            self.loss = tf.add(losses, l2_losses, name="loss")

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # TODO: Reconsider the metrics calculation
        # Number of correct predictions
        with tf.name_scope("num_correct"):
            correct = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct, "float"), name="num_correct")

        # Calculate Fp
        with tf.name_scope("fp"):
            fp = tf.metrics.false_positives(labels=tf.argmax(self.input_y, 1), predictions=self.predictions)
            self.fp = tf.reduce_sum(tf.cast(fp, "float"), name="fp")

        # Calculate Fn
        with tf.name_scope("fn"):
            fn = tf.metrics.false_negatives(labels=tf.argmax(self.input_y, 1), predictions=self.predictions)
            self.fn = tf.reduce_sum(tf.cast(fn, "float"), name="fn")

        # Calculate Recall
        with tf.name_scope("recall"):
            self.recall = self.num_correct / (self.num_correct + self.fn)

        # Calculate Precision
        with tf.name_scope("precision"):
            self.precision = self.num_correct / (self.num_correct + self.fp)

        # Calculate F1
        with tf.name_scope("F1"):
            self.F1 = (2 * self.precision * self.recall) / (self.precision + self.recall)

        # Calculate AUC
        with tf.name_scope("AUC"):
            self.AUC = tf.metrics.auc(self.softmax_scores, self.input_y, name="AUC")
