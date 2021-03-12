#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""This is the top-level file to train, evaluate or test your summarization model"""

import sys
import time
import os
import tensorflow as tf
import numpy as np
import pickle as pk
import util
import pdb

FLAGS = tf.app.flags.FLAGS

def write_to_summary(value, tag_name, step, summary_writer):
  summary = tf.Summary()
  summary.value.add(tag=tag_name, simple_value=value)
  summary_writer.add_summary(summary, step)

def setup_training(model, batcher, word_vector):     # End-end training settings, batcher is specific data input
  """Does setup before starting training (run_training)"""
  train_dir = os.path.join(FLAGS.log_root, "train")    # Where to store experimental results
  if not os.path.exists(train_dir): os.makedirs(train_dir)

  # default_device = tf.device('/gpu:3')
  # with default_device:
  assert FLAGS.coverage, "Please run the end2end model with coverage mechanism."
  model.build_graph() # build the graph

  if FLAGS.pretrained_selector_path and FLAGS.pretrained_rewriter_path:    # If there is a pre-trained model
    params = tf.global_variables()    # Get all the variables
    selector_vars = [param for param in params if "SentSelector" in param.name and 'Adagrad' not in param.name]
    rewriter_vars = [param for param in params if "seq2seq" in param.name and 'Adagrad' not in param.name]
    uninitialized_vars = [param for param in params if param not in selector_vars and param not in rewriter_vars]
    selector_saver = tf.train.Saver(selector_vars)   # Extractor saver
    rewriter_saver = tf.train.Saver(rewriter_vars)   # Abstractor saver
    local_init_op = tf.variables_initializer(uninitialized_vars)   # Variables that are not Extractor or Abstractor
    all_saver = tf.train.Saver(max_to_keep=FLAGS.model_max_to_keep)
  else:
    saver = tf.train.Saver(max_to_keep=FLAGS.model_max_to_keep)

  if FLAGS.pretrained_selector_path and FLAGS.pretrained_rewriter_path:
    sv = tf.train.Supervisor(logdir=train_dir,
                         is_chief=True,
                         saver=None,
                         local_init_op=local_init_op,
                         summary_op=None,
                         save_summaries_secs=60, # save summaries for tensorboard every 60 secs
                         save_model_secs=0, # checkpoint every 60 secs
                         global_step=model.global_step,
                         init_feed_dict = {model._rewriter.embedding_place: word_vector} if FLAGS.embedding else None)   # ,model._selector.embedding_place: word_vector,
  else:
    sv = tf.train.Supervisor(logdir=train_dir,
                         is_chief=True,
                         saver=saver,
                         summary_op=None,
                         save_summaries_secs=60, # save summaries for tensorboard every 60 secs
                         save_model_secs=0, # checkpoint every 60 secs
                         global_step=model.global_step,
                         init_feed_dict = {model._rewriter.embedding_place: word_vector} if FLAGS.embedding else None)
  
  summary_writer = sv.summary_writer
  tf.logging.info("Preparing or waiting for session...")
  sess_context_manager = sv.prepare_or_wait_for_session(config=util.get_config())
  tf.logging.info("Created session.")

  try:
    if FLAGS.pretrained_selector_path and FLAGS.pretrained_rewriter_path:
      run_training(model, batcher, sess_context_manager, sv, summary_writer, word_vector, \
                   selector_saver, rewriter_saver, all_saver) # this is an infinite loop until interrupted
    else:
      run_training(model, batcher, sess_context_manager, sv, summary_writer, word_vector) # this is an infinite loop until interrupted
  except KeyboardInterrupt:
    tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
    sv.stop()


def run_training(model, batcher, sess_context_manager, sv, summary_writer, word_vector, \
                 selector_saver=None, rewriter_saver=None, all_saver=None):
  """Repeatedly runs training iterations, logging loss to screen and writing summaries"""
  tf.logging.info("starting run_training")
  train_step = 0
  ckpt_path = os.path.join(FLAGS.log_root, "train", "model.ckpt_cov")

  with sess_context_manager as sess:
    if FLAGS.pretrained_selector_path:    # Load the pre-trained model
      tf.logging.info('Loading selector model')
      _ = util.load_ckpt(selector_saver, sess, ckpt_path=FLAGS.pretrained_selector_path)
    if FLAGS.pretrained_rewriter_path:
      tf.logging.info('Loading rewriter model')
      _ = util.load_ckpt(rewriter_saver, sess, ckpt_path=FLAGS.pretrained_rewriter_path)

    for _ in range(FLAGS.max_train_iter): # repeats until interrupted
      batch = batcher.next_batch()

      tf.logging.info('running training step...')
      t0=time.time()
      results = model.run_train_step(sess, batch, word_vector, train_step)
      t1=time.time()
      tf.logging.info('seconds for training step: %.3f', t1-t0)

      loss = results['loss']
      tf.logging.info('rl_loss: %f', loss) # print the loss to screen
      train_step = results['global_step']

      if not np.isfinite(loss):
        raise Exception("Loss is not finite. Stopping.")

      tf.logging.info("reinforce_avg_logprobs: %f", results['reinforce_avg_logprobs'])

      if FLAGS.coverage:
        tf.logging.info("coverage_loss: %f", results['coverage_loss']) # print the coverage loss to screen
        tf.logging.info("reinforce_coverage_loss: %f", results['reinforce_coverage_loss'])

      if FLAGS.inconsistent_loss:
        tf.logging.info('inconsistent_loss: %f', results['inconsist_loss'])

      tf.logging.info("selector_loss: %f", results['selector_loss'])
      recall, ratio, _ = util.get_batch_ratio(batch.original_articles_sents, batch.original_extracts_ids, results['probs'])
      write_to_summary(ratio, 'SentSelector/select_ratio/recall=0.9', train_step, summary_writer)

      # get the summaries and iteration number so we can write summaries to tensorboard
      summaries = results['summaries'] # we will write these summaries to tensorboard using summary_writer
      summary_writer.add_summary(summaries, train_step) # write the summaries
      if train_step % 100 == 0: # flush the summary writer every so often
        summary_writer.flush()

      if train_step % FLAGS.save_model_every == 0:
        if FLAGS.pretrained_selector_path and FLAGS.pretrained_rewriter_path:
          all_saver.save(sess, ckpt_path, global_step=train_step)
        else:
          sv.saver.save(sess, ckpt_path, global_step=train_step)

      print ('Step: ', train_step)


def run_eval(model, batcher, word_vector):
  """Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far."""
  model.build_graph()   # build the graph
  saver = tf.train.Saver(max_to_keep=3) # we will keep 3 best checkpoints at a time
  sess = tf.Session(config=util.get_config())
  eval_dir = os.path.join(FLAGS.log_root, "eval_loss") # make a subdir of the root dir for eval data
  bestmodel_save_path = os.path.join(eval_dir, 'bestmodel') # this is where checkpoints of best models are saved
  summary_writer = tf.summary.FileWriter(eval_dir)
  running_avg_loss = 0 # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
  running_avg_ratio = 0 # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
  best_loss = None  # will hold the best loss achieved so far
  train_dir = os.path.join(FLAGS.log_root, "train")
  train_step = 0
  while True:
    ckpt_state = tf.train.get_checkpoint_state(train_dir)
    tf.logging.info('max_enc_steps: %d, max_dec_steps: %d', FLAGS.max_enc_steps, FLAGS.max_dec_steps)
    _ = util.load_ckpt(saver, sess) # load a new checkpoint
    batch = batcher.next_batch() # get the next batch

    # run eval on the batch
    t0=time.time()
    results = model.run_eval_step(sess, batch, word_vector, train_step)
    t1=time.time()
    tf.logging.info('seconds for batch: %.2f', t1-t0)

    # print the loss and coverage loss to screen
    loss = results['loss']
    tf.logging.info('rl_loss: %f', loss)
    train_step = results['global_step']

    tf.logging.info("reinforce_avg_logprobs: %f", results['reinforce_avg_logprobs'])

    if FLAGS.coverage:
      tf.logging.info("coverage_loss: %f", results['coverage_loss'])
      tf.logging.info("reinforce_coverage_loss: %f", results['reinforce_coverage_loss'])

    if FLAGS.inconsistent_loss:
      tf.logging.info('inconsistent_loss: %f', results['inconsist_loss'])

    tf.logging.info("selector_loss: %f", results['selector_loss'])
    recall, ratio, _ = util.get_batch_ratio(batch.original_articles_sents, batch.original_extracts_ids, results['probs'])
    write_to_summary(ratio, 'SentSelector/select_ratio/recall=0.9', train_step, summary_writer)

    # add summaries
    summaries = results['summaries']
    summary_writer.add_summary(summaries, train_step)

    # calculate running avg loss
    running_avg_loss = util.calc_running_avg_loss(np.asscalar(loss), running_avg_loss, summary_writer, train_step, 'running_avg_loss')
    running_avg_ratio = util.calc_running_avg_loss(ratio, running_avg_ratio, summary_writer, train_step, 'running_avg_ratio')

    # If running_avg_loss is best so far, save this checkpoint (early stopping).
    # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
    if best_loss is None or running_avg_loss < best_loss:
      tf.logging.info('Found new best model with %.3f running_avg_loss. Saving to %s', running_avg_loss, bestmodel_save_path)
      saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
      best_loss = running_avg_loss

    # flush the summary writer every so often
    if train_step % 100 == 0:
      summary_writer.flush()


def run_eval_rouge(evaluator):
  """Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far."""
  eval_dir = os.path.join(FLAGS.log_root, 'eval_rouge') # make a subdir of the root dir for eval data
  bestmodel_save_path = os.path.join(eval_dir, 'bestmodel') # this is where checkpoints of best models are saved
  summary_writer = tf.summary.FileWriter(eval_dir)

  best_rouge_file = os.path.join(eval_dir, 'best_rouge.pkl')
  if os.path.exists(best_rouge_file):
    best_rouges = pk.load(open(best_rouge_file, 'rb'))
    current_step = best_rouges['step']
    best_rouge1 = best_rouges['1']
    best_rouge2 = best_rouges['2']
    best_rougeL = best_rouges['l']
    tf.logging.info('previous best rouge1: %3f, rouge2: %3f, rougeL: %3f, step: %d', \
                    best_rouge1, best_rouge2, best_rougeL, current_step)
  else:
    current_step = None
    best_rouge1 = None  # will hold the best rouge1 achieved so far
    best_rouge2 = None  # will hold the best rouge2 achieved so far
    best_rougeL = None  # will hold the best rougeL achieved so far

  train_dir = os.path.join(FLAGS.log_root, "train")
  if FLAGS.coverage:
    ckpt_base_path = os.path.join(train_dir, "model.ckpt_cov")
  else:
    ckpt_base_path = os.path.join(train_dir, "model.ckpt")

  while True:
    if current_step is None:
      ckpt_state = tf.train.get_checkpoint_state(train_dir)
      if ckpt_state:
        step = os.path.basename(ckpt_state.model_checkpoint_path).split('-')[1]

        if int(step) < FLAGS.start_eval_rouge:
          tf.logging.info('Step = ' + str(step) + ' (smaller than start_eval_rouge, Sleeping for 10 secs...)')
          time.sleep(10)
          continue
        else:
          current_step = int(step)
          current_ckpt_path = ckpt_base_path + '-' + str(current_step)
      else:
        tf.logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", train_dir, 10)
        time.sleep(10)
        continue
    else:
      current_step += FLAGS.save_model_every
      if int(current_step) < FLAGS.start_eval_rouge:
        tf.logging.info('Step = ' + str(current_step) + ' (smaller than start_eval_rouge, Sleeping for 10 secs...)')
        time.sleep(10)
        continue
      current_ckpt_path = ckpt_base_path + '-' + str(current_step)
      evaluator.init_batcher()

    tf.logging.info('max_enc_steps: %d, max_dec_steps: %d', FLAGS.max_enc_steps, FLAGS.max_dec_steps)
    do_eval = evaluator.prepare_evaluate(ckpt_path=current_ckpt_path)
    if not do_eval:  # The checkpoint has already been evaluated. Evaluate next one.
      tf.logging.info('step %d checkpoint has already been evaluated, evaluate next checkpoint.', current_step)
      continue
    rouge_results, rouge_results_str = evaluator.evaluate()

    # print the loss and coverage loss to screen
    results_file = os.path.join(eval_dir, "ROUGE_results_all.txt")
    with open(results_file, "a") as f:
      f.write('Step: ' + str(current_step))
      f.write(rouge_results_str + '\n')

    # add summaries
    write_to_summary(rouge_results['1'], 'rouge_results/rouge1', current_step, summary_writer)
    write_to_summary(rouge_results['2'], 'rouge_results/rouge2', current_step, summary_writer)
    write_to_summary(rouge_results['l'], 'rouge_results/rougeL', current_step, summary_writer)

    # If running_avg_loss is best so far, save this checkpoint (early stopping).
    # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
    better_metric = 0
    if best_rouge1 is None or rouge_results['1'] >= best_rouge1:
      best_rouge1 = rouge_results['1']
      better_metric += 1
    if best_rouge2 is None or rouge_results['2'] >= best_rouge2:
      best_rouge2 = rouge_results['2']
      better_metric += 1
    if best_rougeL is None or rouge_results['l'] >= best_rougeL:
      best_rougeL = rouge_results['l']
      better_metric += 1

    if better_metric >= 2:
      tf.logging.info('Found new best model with rouge1 %f, rouge2 %f, rougeL %f. Saving to %s', rouge_results['1'], rouge_results['2'], rouge_results['l'], bestmodel_save_path)
      evaluator._saver.save(evaluator._sess, bestmodel_save_path, global_step=current_step, latest_filename='checkpoint_best')
      rouge_results['step'] = current_step
      with open(best_rouge_file, 'wb') as f:
        pk.dump(rouge_results, f)

    # flush the summary writer every so often
    if current_step % 100 == 0:
      summary_writer.flush()
