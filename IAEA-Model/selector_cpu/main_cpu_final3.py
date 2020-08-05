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

"""This is the top-level file to train, evaluate or test your summarization model"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import time
import os
import tensorflow as tf
import numpy as np
from collections import namedtuple
import pprint

from selector_cpu.model import SentenceSelector
from selector_cpu.evaluate import SelectorEvaluator
import selector_cpu.run_selector as run_selector

from RLrewriter3.model import Rewriter
from RLrewriter3.decode import BeamSearchDecoder
import RLrewriter3.run_rewriter as run_rewriter

from end2end2.evaluate import End2EndEvaluator
from end2end2.model import SelectorRewriter
import end2end2.run_end2end as run_end2end

from data import Vocab
from batcher import Batcher
import util
import pdb

FLAGS = tf.app.flags.FLAGS

# Where to find data
tf.app.flags.DEFINE_string('data_path', '', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', '', 'Path expression to text vocabulary file.')

# Important settings
tf.app.flags.DEFINE_string('model', '', 'must be one of selector/rewriter/end2end')
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/evalall')
tf.app.flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')
tf.app.flags.DEFINE_integer('decode_after', 0, 'skip already decoded docs')
tf.app.flags.DEFINE_string('decode_from', 'train', 'Decode from train/eval model.')
# Where to save output
tf.app.flags.DEFINE_integer('max_train_iter', 10000, 'max iterations to train')
tf.app.flags.DEFINE_integer('save_model_every', 1000, 'save the model every N iterations')
tf.app.flags.DEFINE_integer('model_max_to_keep', 5, 'save latest N models')
tf.app.flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', '', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')

# For eval mode used in rewriter and end2end training 
# (This mode will do evaluation during training for choosing best model)
tf.app.flags.DEFINE_string('eval_method', '', 'loss or rouge (loss mode is to get the loss for one batch; rouge mode is to get rouge scores for the whole dataset)')
tf.app.flags.DEFINE_integer('start_eval_rouge', 300, 'for rouge mode, start evaluating rouge scores after this iteration')   #要至少30000

# For evalall mode or (eval mode with eval_method == 'rouge')
tf.app.flags.DEFINE_string('decode_method', '', 'greedy/beam')
tf.app.flags.DEFINE_boolean('load_best_eval_model', False, 'evalall mode only')
tf.app.flags.DEFINE_string('eval_ckpt_path', '', 'evalall mode only, checkpoint path for evalall mode')
tf.app.flags.DEFINE_boolean('save_pkl', False, 'whether to save the results as pickle files')
tf.app.flags.DEFINE_boolean('save_vis', False, 'whether to save the results for visualization')

# Load pretrained selector or rewriter
tf.app.flags.DEFINE_string('pretrained_selector_path', '', 'pretrained selector checkpoint path')
tf.app.flags.DEFINE_string('pretrained_rewriter_path', '', 'pretrained rewriter checkpoint path')

# For end2end training
tf.app.flags.DEFINE_float('selector_loss_wt', 5.0, 'weight of selector loss when end2end')
tf.app.flags.DEFINE_boolean('inconsistent_loss', True, 'whether to minimize inconsistent loss when end2end')
tf.app.flags.DEFINE_integer('inconsistent_topk', 3, 'choose top K word attention to compute inconsistent loss')

# Hyperparameters for both selector and rewriter
tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
tf.app.flags.DEFINE_integer('emb_dim', 100, 'dimension of word embeddings')   # 先不考虑用Glove的100维推特embedding,现在要改成100，原来是128，现在变成Glove了
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')
tf.app.flags.DEFINE_string('embedding', 'data/Glove/glove.twitter.27B.100d.txt', 'path to the pre-trained embedding file')

# Hyperparameters for selector only
tf.app.flags.DEFINE_integer('hidden_dim_selector', 200, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('max_art_len', 50, 'max timesteps of sentence-level encoder')   # init 50
tf.app.flags.DEFINE_integer('max_sent_len', 50, 'max timesteps of word-level encoder')
tf.app.flags.DEFINE_string('select_method', 'prob', 'prob/ratio/num')
tf.app.flags.DEFINE_float('thres', 0.4, 'threshold for selecting sentence')
tf.app.flags.DEFINE_integer('min_select_sent', 5, 'min sentences need to be selected')
tf.app.flags.DEFINE_integer('max_select_sent', 20, 'max sentences to be selected')
tf.app.flags.DEFINE_boolean('eval_gt_rouge', False, 'whether to evaluate ROUGE scores of ground-truth selected sentences')
tf.app.flags.DEFINE_boolean('is_training', False, 'whether dropout to run')
tf.app.flags.DEFINE_float('dropout_rate', 0.5, 'dropout value')

# Hyperparameters for rewriter only
tf.app.flags.DEFINE_integer('hidden_dim_rewriter', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('enc_hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('dec_hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('max_enc_steps', 600, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')
tf.app.flags.DEFINE_boolean('avoid_trigrams', True, 'Avoids trigram during decoding')
tf.app.flags.DEFINE_boolean('share_decoder_weights', False, 'Share output matrix projection with word embedding') # Eq 13. in https://arxiv.org/pdf/1705.04304.pdf
tf.app.flags.DEFINE_integer('gpu_num', 2, 'which gpu to use to train the model')
# Pointer-generator with Self-Critic policy gradient: https://arxiv.org/pdf/1705.04304.pdf   是rewriter用的
tf.app.flags.DEFINE_boolean('rl_training', False, 'Use policy-gradient training by collecting rewards at the end of sequence.')
tf.app.flags.DEFINE_boolean('self_critic', True, 'Uses greedy sentence reward as baseline.')
tf.app.flags.DEFINE_boolean('use_discounted_rewards', False, 'Whether to use discounted rewards.')
tf.app.flags.DEFINE_boolean('use_intermediate_rewards', False, 'Whether to use intermediate rewards.')
tf.app.flags.DEFINE_boolean('convert_to_reinforce_model', False, 'Convert a pointer model to a reinforce model. Turn this on and run in train mode. Your current training model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')
tf.app.flags.DEFINE_boolean('intradecoder', False, 'Use intradecoder attention or not')
tf.app.flags.DEFINE_boolean('use_temporal_attention', False, 'Whether to use temporal attention or not')
tf.app.flags.DEFINE_boolean('matrix_attention', False, 'Use matrix attention, Eq. 2 https://arxiv.org/pdf/1705.04304.pdf')
tf.app.flags.DEFINE_float('eta', 0, 'RL/MLE scaling factor, 1 means use RL loss, 0 means use MLE loss')
tf.app.flags.DEFINE_boolean('fixed_eta', False, 'Use fixed value for eta or adaptive based on global step')
tf.app.flags.DEFINE_float('gamma', 0.99, 'discount factor')
tf.app.flags.DEFINE_string('reward_function', 'rouge_l/f_score', 'either bleu or one of the rouge measures (rouge_1/f_score,rouge_2/f_score,rouge_l/f_score)')


# parameters of DDQN model
tf.app.flags.DEFINE_boolean('ac_training', False, 'Use Actor-Critic learning by DDQN.')
tf.app.flags.DEFINE_boolean('dqn_scheduled_sampling', False, 'Whether to use scheduled sampling to use estimates of dqn model vs the actual q-estimates values')
tf.app.flags.DEFINE_string('dqn_layers', '512,256,128', 'DQN dense hidden layer size, will create three dense layers with 512, 256, and 128 size')
tf.app.flags.DEFINE_integer('dqn_replay_buffer_size', 100000, 'Size of the replay buffer')
tf.app.flags.DEFINE_integer('dqn_batch_size', 100, 'Batch size for training the DDQN model')
tf.app.flags.DEFINE_integer('dqn_target_update', 10000, 'Update target Q network every 10000 steps')
tf.app.flags.DEFINE_integer('dqn_sleep_time', 2, 'Train DDQN model every 2 seconds')
tf.app.flags.DEFINE_integer('dqn_gpu_num', 1, 'GPU number to train the DDQN')
tf.app.flags.DEFINE_boolean('dueling_net', True, 'Whether to use Duelling Network to train the model') # https://arxiv.org/pdf/1511.06581.pdf
tf.app.flags.DEFINE_boolean('dqn_polyak_averaging', True, 'Whether to use polyak averaging to update the target network parameters')
tf.app.flags.DEFINE_boolean('calculate_true_q', False, "Whether to use true Q-values to train DQN or use DQN's estimates to train it")
tf.app.flags.DEFINE_boolean('dqn_pretrain', False, "Pretrain the DDQN network with fixed Actor model")
tf.app.flags.DEFINE_integer('dqn_pretrain_steps', 10000, 'Number of steps to pre-train the DDQN')

#scheduled sampling parameters, https://arxiv.org/pdf/1506.03099.pdf
# At each time step t and for each sequence in the batch, we get the input to next decoding step by either
#   (1) sampling from the final distribution at (t-1), or
#   (2) reading from input_decoder_embedding.
# We do (1) with probability sampling_probability and (2) with 1 - sampling_probability.
# Using sampling_probability=0.0 is equivalent to using only the ground truth data (no sampling).
# Using sampling_probability=1.0 is equivalent to doing inference by only relying on the sampled token generated at each decoding step
tf.app.flags.DEFINE_boolean('scheduled_sampling', False, 'whether to do scheduled sampling or not')
tf.app.flags.DEFINE_string('decay_function', 'linear','linear, exponential, inv_sigmoid') #### TODO: implement this
tf.app.flags.DEFINE_float('sampling_probability', 0, 'epsilon value for choosing ground-truth or model output')
tf.app.flags.DEFINE_boolean('fixed_sampling_probability', False, 'Whether to use fixed sampling probability or adaptive based on global step')
tf.app.flags.DEFINE_boolean('hard_argmax', True, 'Whether to use soft argmax or hard argmax')
tf.app.flags.DEFINE_boolean('greedy_scheduled_sampling', False, 'Whether to use greedy approach or sample for the output, if True it uses greedy')
tf.app.flags.DEFINE_boolean('E2EBackProp', False, 'Whether to use E2EBackProp algorithm to solve exposure bias')
tf.app.flags.DEFINE_float('alpha', 1, 'soft argmax argument')
tf.app.flags.DEFINE_integer('k', 1, 'number of samples')
tf.app.flags.DEFINE_boolean('scheduled_sampling_final_dist', True, 'Whether to use final distribution or vocab distribution for scheduled sampling')

# Coverage hyperparameters
tf.app.flags.DEFINE_boolean('coverage', False, 'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
tf.app.flags.DEFINE_float('cov_loss_wt', 1.0, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')
tf.app.flags.DEFINE_boolean('convert_to_coverage_model', False, 'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')

# Utility flags, for restoring and changing checkpoints
tf.app.flags.DEFINE_boolean('restore_best_model', False, 'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')

def main(unused_argv):
  if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
    raise Exception("Problem with flags: %s" % unused_argv)

  pp = pprint.PrettyPrinter()
  pp.pprint(FLAGS.__flags)

  tf.logging.set_verbosity(tf.logging.INFO) # choose what level of logging you want
  if FLAGS.model not in ['selector', 'rewriter', 'end2end']:
    raise ValueError("The 'model' flag must be one of selector/rewriter/end2end")
  if FLAGS.mode not in ['train', 'eval', 'evalall']:
    raise ValueError("The 'mode' flag must be one of train/eval/evalall")
  tf.logging.info('Starting %s in %s mode...' % (FLAGS.model, FLAGS.mode))    # 记录当前的选择

  # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
  FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.model, FLAGS.exp_name)   # 创建实验生成的模型生成的位置
  if not os.path.exists(FLAGS.log_root):
    if FLAGS.mode=="train":
      os.makedirs(FLAGS.log_root)
    else:
      raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))

  vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)   # create a vocabulary  # 频率越大的词 它的id越小了

  # If in evalall mode, set batch_size = 1 or beam_size
  # Reason: in evalall mode, we decode one example at a time.
  # For rewriter, on each step, we have beam_size-many hypotheses in the beam, 
  # so we need to make a batch of these hypotheses.
  if FLAGS.mode == 'evalall':
    if FLAGS.model == 'selector':
      FLAGS.batch_size = 1
    else:
      if FLAGS.decode_method == 'beam':
        FLAGS.batch_size = FLAGS.beam_size

  # If single_pass=True, check we're in evalall mode
  if FLAGS.single_pass and FLAGS.mode=='train':
    raise Exception("The single_pass flag should not be True in train mode")

  # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
  hparam_list = ['model', 'mode', 'eval_method', 'selector_loss_wt', 'inconsistent_loss', 'inconsistent_topk', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm', 'hidden_dim_selector', 'hidden_dim_rewriter','emb_dim',
                 'batch_size', 'max_art_len', 'max_sent_len', 'max_dec_steps', 'max_enc_steps', 'coverage', 'cov_loss_wt', 'eval_gt_rouge', 'decode_method',
                 'lr', 'gamma', 'eta', 'fixed_eta','reward_function', 'intradecoder', 'use_temporal_attention', 'rl_training', 'matrix_attention', 'pointer_gen',
                 'alpha', 'hard_argmax', 'greedy_scheduled_sampling','k','calculate_true_q'
                 ,'dqn_scheduled_sampling', 'dqn_sleep_time', 'E2EBackProp','gpu_num','enc_hidden_dim','dec_hidden_dim',
                 'scheduled_sampling', 'sampling_probability', 'fixed_sampling_probability','hard_argmax', 'greedy_scheduled_sampling','dqn_scheduled_sampling', 'dqn_sleep_time', 'E2EBackProp','ac_training'
                 ]
  hps_dict = {}
  for key,val in FLAGS.__flags.items(): # for each flag
    if key in hparam_list: # if it's in the list
      hps_dict[key] = val # add it to the dict
  # for val in FLAGS:
  #   if val in hparam_list:
  #     hps_dict[val] = FLAGS[val].value
  hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

  # Create a batcher object that will create minibatches of data
  batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)

  tf.set_random_seed(111) # a seed value for randomness
  vocab.LoadWordEmbedding(FLAGS.embedding, FLAGS.emb_dim)
  if FLAGS.model == 'selector':   # 选择句子
    print(hps.mode)
    if hps.mode == 'train':   # 训练阶段
      print ("creating model...")
      model = SentenceSelector(hps, vocab)    # 初始化训练器的东西
      run_selector.setup_training(model, batcher, vocab.getWordEmbedding())
    elif hps.mode == 'eval':   # 评估阶段
      model = SentenceSelector(hps, vocab)
      run_selector.run_eval(model, batcher, vocab.getWordEmbedding())
    elif hps.mode == 'evalall':     # 真正的评估，测试,得到rouge和output
      model = SentenceSelector(hps, vocab)
      evaluator = SelectorEvaluator(model, batcher, vocab)
      evaluator.evaluate()
  elif FLAGS.model == 'rewriter':
    if hps.mode == 'train':
      print ("creating model...")
      model = Rewriter(hps, vocab)
      run_rewriter.setup_training(model, batcher, vocab.getWordEmbedding())
    elif hps.mode == 'eval':
      model = Rewriter(hps, vocab)
      if FLAGS.eval_method == 'loss':
        vocab.LoadWordEmbedding(FLAGS.embedding, FLAGS.emb_dim)
        run_rewriter.run_eval(model, batcher, vocab.getWordEmbedding())
      elif FLAGS.eval_method == 'rouge':
        assert FLAGS.decode_method == 'greedy'
        decoder = BeamSearchDecoder(model, batcher, vocab)
        run_rewriter.run_eval_rouge(decoder)
    elif hps.mode == 'evalall':
      decode_model_hps = hps  # This will be the hyperparameters for the decoder model
      if FLAGS.decode_method == 'beam':
        decode_model_hps = hps._replace(max_dec_steps=1) # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries
      model = Rewriter(decode_model_hps, vocab)
      decoder = BeamSearchDecoder(model, batcher, vocab)
      decoder.evaluate() # decode indefinitely (unless single_pass=True, in which case deocde the dataset exactly once)
  elif FLAGS.model == 'end2end':    # 端对端的预测
    if hps.mode == 'train':   # 训练阶段
      print ("creating model...")
      select_model = SentenceSelector(hps, vocab)   # 抽取器模型初始化
      rewrite_model = Rewriter(hps, vocab)   # 生成器模型初始化
      end2end_model = SelectorRewriter(hps, select_model, rewrite_model)     # 合并模型初始化
      run_end2end.setup_training(end2end_model, batcher, vocab.getWordEmbedding())   # 训练设置，传入模型器和批次的数据
    elif hps.mode == 'eval':
      select_model = SentenceSelector(hps, vocab)
      rewrite_model = Rewriter(hps, vocab)
      end2end_model = SelectorRewriter(hps, select_model, rewrite_model)
      if FLAGS.eval_method == 'loss':
        run_end2end.run_eval(end2end_model, batcher, vocab.getWordEmbedding())
      elif FLAGS.eval_method == 'rouge':
        assert FLAGS.decode_method == 'greedy'
        evaluator = End2EndEvaluator(end2end_model, batcher, vocab)
        run_end2end.run_eval_rouge(evaluator)
    elif hps.mode == 'evalall':
      eval_model_hps = hps  # This will be the hyperparameters for the decoder model
      if FLAGS.decode_method == 'beam':
        eval_model_hps = hps._replace(max_dec_steps=1) # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries
      select_model = SentenceSelector(eval_model_hps, vocab)
      rewrite_model = Rewriter(eval_model_hps, vocab)
      end2end_model = SelectorRewriter(hps, select_model, rewrite_model)
      evaluator = End2EndEvaluator(end2end_model, batcher, vocab)
      evaluator.evaluate() # decode indefinitely (unless single_pass=True, in which case deocde the dataset exactly once)

if __name__ == '__main__':
  tf.app.run()
