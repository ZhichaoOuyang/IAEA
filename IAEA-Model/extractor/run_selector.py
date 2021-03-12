#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""This is the top-level file to train, evaluate or test your summarization model"""
import os
import sys
import time
import tensorflow as tf
import numpy as np
import util
import pdb


FLAGS = tf.app.flags.FLAGS

def write_to_summary(value, tag_name, step, summary_writer):
  summary = tf.Summary()
  summary.value.add(tag=tag_name, simple_value=value)
  summary_writer.add_summary(summary, step)

def setup_training(model, batcher, word_vector):
  """Does setup before starting training (run_training)"""
  train_dir = os.path.join(FLAGS.log_root, "train")    # Save the path of the model pre-trained by the selector
  if not os.path.exists(train_dir): os.makedirs(train_dir)

  # default_device = tf.device('/gpu:3')
  # with default_device:
  model.build_graph() # build the graph
  params = tf.global_variables()
  if FLAGS.pretrained_selector_path: # cross entropy loss eval best model or train model
    params = tf.global_variables()
    # do not load global step and adagrad states (since the pretrained model may come from best eval model and the eval model do not have adagrad states)
    selector_vars = [param for param in params if "SentSelector" in param.name and 'Adagrad' not in param.name]
    uninitialized_vars = [param for param in params if param not in selector_vars]
    pretrained_saver = tf.train.Saver(selector_vars)
    local_init_op = tf.variables_initializer(uninitialized_vars)
    saver = tf.train.Saver(max_to_keep=FLAGS.model_max_to_keep)
  else:    # 没有预训练的model
    saver = tf.train.Saver(max_to_keep=FLAGS.model_max_to_keep)
  
  if FLAGS.pretrained_selector_path:  # 有预训练模型
    sv = tf.train.Supervisor(logdir=train_dir,
                         is_chief=True,
                         saver=None,
                         local_init_op=local_init_op,
                         summary_op=None,
                         save_summaries_secs=60, # save summaries for tensorboard every 60 secs
                         save_model_secs=0, # do not save checkpoint
                         global_step=model.global_step )  # add)
  else:
    sv = tf.train.Supervisor(logdir=train_dir,
                         is_chief=True,
                         saver=saver,
                         summary_op=None,
                         save_summaries_secs=60, # save summaries for tensorboard every 60 secs
                         save_model_secs=0, # do not save checkpoint
                         global_step=model.global_step)  # add    )

  summary_writer = sv.summary_writer
  tf.logging.info("Preparing or waiting for session...")
  sess_context_manager = sv.prepare_or_wait_for_session(config=util.get_config())
  tf.logging.info("Created session.")

  try:
    if FLAGS.pretrained_selector_path:
      run_training(model, batcher, sess_context_manager, sv, summary_writer, \
                   pretrained_saver, saver) # this is an infinite loop until interrupted
    else:
      run_training(model, batcher, sess_context_manager, sv, summary_writer) # this is an infinite loop until interrupted
  except KeyboardInterrupt:
    tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
    sv.stop()


def run_training(model, batcher, sess_context_manager, sv, summary_writer,
                 pretrained_saver=None, saver=None):
  """Repeatedly runs training iterations, logging loss to screen and writing summaries"""
  tf.logging.info("starting run_training")
  ckpt_path = os.path.join(FLAGS.log_root, "train", "model.ckpt")

  with sess_context_manager as sess:
    if FLAGS.pretrained_selector_path:
      tf.logging.info('Loading pretrained selector model')
      _ = util.load_ckpt(pretrained_saver, sess, ckpt_path=FLAGS.pretrained_selector_path)

    for _ in range(FLAGS.max_train_iter): # repeats until interrupted
      batch = batcher.next_batch()   #

      tf.logging.info('running training step...')
      t0=time.time()
      results = model.run_train_step(sess, batch)
      print("run train step finish")
      t1=time.time()
      tf.logging.info('seconds for training step: %.3f', t1-t0)

      loss = results['loss']
      tf.logging.info('loss: %f', loss) # print the loss to screen

      if not np.isfinite(loss):
        raise Exception("Loss is not finite. Stopping.")

      train_step = results['global_step'] # we need this to update our running average loss

      recall, ratio, _ = util.get_batch_ratio(batch.original_articles_sents, \
                                              batch.original_extracts_ids, results['probs'])
      write_to_summary(ratio, 'SentSelector/select_ratio/recall=0.9', train_step, summary_writer)
      
      # get the summaries and iteration number so we can write summaries to tensorboard
      summaries = results['summaries'] # we will write these summaries to tensorboard using summary_writer
      summary_writer.add_summary(summaries, train_step) # write the summaries
      if train_step % 100 == 0:   # flush the summary writer every so often
        summary_writer.flush()

      if train_step % FLAGS.save_model_every == 0:
        if FLAGS.pretrained_selector_path:
          saver.save(sess, ckpt_path, global_step=train_step)
        else:
          sv.saver.save(sess, ckpt_path, global_step=train_step)

      print ('Step: ', train_step)


def run_eval(model, batcher, word_vector):
  """Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far."""
  model.build_graph() # build the graph
  saver = tf.train.Saver(max_to_keep=3) # we will keep 3 best checkpoints at a time
  sess = tf.Session(config=util.get_config())
  if FLAGS.embedding:
    sess.run(tf.global_variables_initializer()) #, feed_dict={model.embedding_place: word_vector}
  eval_dir = os.path.join(FLAGS.log_root, "eval") # make a subdir of the root dir for eval data
  bestmodel_save_path = os.path.join(eval_dir, 'bestmodel') # this is where checkpoints of best models are saved
  summary_writer = tf.summary.FileWriter(eval_dir)   # Specify a file to save the graph

  running_avg_ratio = 0 # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
  best_ratio = None  # will hold the best loss achieved so far
  train_dir = os.path.join(FLAGS.log_root, "train")

  while True:
    ckpt_state = tf.train.get_checkpoint_state(train_dir)   # get the info of checkpoint file
    
    #tf.logging.info('max_enc_steps: %d, max_dec_steps: %d', FLAGS.max_enc_steps, FLAGS.max_dec_steps)
    _ = util.load_ckpt(saver, sess) # load a new checkpoint
    batch = batcher.next_batch() # get the next batch

    # run eval on the batch
    t0=time.time()
    results = model.run_eval_step(sess, batch)
    t1=time.time()
    tf.logging.info('seconds for batch: %.2f', t1-t0)

    # print the loss and coverage loss to screen
    loss = results['loss']
    tf.logging.info('loss: %f', loss)
    train_step = results['global_step']

    recall, ratio, _ = util.get_batch_ratio(batch.original_articles_sents, \
                                            batch.original_extracts_ids, results['probs'])
    write_to_summary(ratio, 'SentSelector/select_ratio/recall=0.9', train_step, summary_writer)
    
    # add summaries
    summaries = results['summaries']
    summary_writer.add_summary(summaries, train_step)

    # calculate running avg loss
    running_avg_ratio = util.calc_running_avg_loss(ratio, running_avg_ratio, summary_writer, train_step, 'running_avg_ratio')
    print("run_avg_ratio: ", running_avg_ratio)
    tf.log("run_avg_ratio: ", running_avg_ratio)
    # If running_avg_loss is best so far, save this checkpoint (early stopping).
    # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
    if best_ratio is None or running_avg_ratio < best_ratio:
      tf.logging.info('Found new best model with %.3f running_avg_ratio. Saving to %s', running_avg_ratio, bestmodel_save_path)
      saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
      best_ratio = running_avg_ratio


    # flush the summary writer every so often
    if train_step % 100 == 0:
      summary_writer.flush()

