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

import sys
import time
import os
from replay_buffer import ReplayBuffer
import tensorflow as tf
import numpy as np
import pickle as pk
import util
import pdb
from glob import glob

FLAGS = tf.app.flags.FLAGS

def write_to_summary(value, tag_name, step, summary_writer):
  summary = tf.Summary()
  summary.value.add(tag=tag_name, simple_value=value)
  summary_writer.add_summary(summary, step)


def calc_running_avg_loss(loss, running_avg_loss, step,summary_writer, decay=0.99):
  """Calculate the running average loss via exponential decay.
  This is used to implement early stopping w.r.t. a more smooth loss curve than the raw loss curve.

  Args:
    loss: loss on the most recent eval step
    running_avg_loss: running_avg_loss so far
    summary_writer: FileWriter object to write for tensorboard
    step: training iteration step
    decay: rate of exponential decay, a float between 0 and 1. Larger is smoother.

  Returns:
    running_avg_loss: new running average loss
  """
  if running_avg_loss == 0:  # on the first iteration just take the loss
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 12)  # clip
  loss_sum = tf.Summary()
  tag_name = 'running_avg_loss/decay=%f' % (decay)
  loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
  summary_writer.add_summary(loss_sum, step)
  tf.logging.info('running_avg_loss: %f', running_avg_loss)
  return running_avg_loss



def write_to_summary(value, tag_name, step, summary_writer):
  summary = tf.Summary()
  summary.value.add(tag=tag_name, simple_value=value)
  summary_writer.add_summary(summary, step)


def restore_best_model():
  """Load bestmodel file from eval directory, add variables for adagrad, and save to train directory"""
  tf.logging.info("Restoring bestmodel for training...")
  # Initialize all vars in the model
  sess = tf.Session(config=util.get_config())
  print("Initializing all variables...")
  sess.run(tf.initialize_all_variables())

  # Restore the best model from eval dir
  saver = tf.train.Saver([v for v in tf.all_variables() if "Adagrad" not in v.name])
  print("Restoring all non-adagrad variables from best model in eval dir...")
  curr_ckpt = util.load_ckpt(saver, sess, "eval")
  print("Restored %s." % curr_ckpt)

  # Save this model to train dir and quit
  new_model_name = curr_ckpt.split("/")[-1].replace("bestmodel", "model")
  new_fname = os.path.join(FLAGS.log_root, "train", new_model_name)
  print("Saving model to %s..." % (new_fname))
  new_saver = tf.train.Saver() # this saver saves all variables that now exist, including Adagrad variables
  new_saver.save(sess, new_fname)
  print("Saved.")
  exit()

def restore_best_eval_model():
  # load best evaluation loss so far
  best_loss = None
  best_step = None
  # goes through all event files and select the best loss achieved and return it
  event_files = sorted(glob('{}/eval/events*'.format(FLAGS.log_root)))
  for ef in event_files:
    try:
      for e in tf.train.summary_iterator(ef):
        for v in e.summary.value:
          step = e.step
          if 'running_avg_loss/decay' in v.tag:
            running_avg_loss = v.simple_value
            if best_loss is None or running_avg_loss < best_loss:
              best_loss = running_avg_loss
              best_step = step
    except:
      continue
  tf.logging.info('resotring best loss from the current logs: {}\tstep: {}'.format(best_loss, best_step))
  return best_loss


def convert_to_coverage_model(model,word_vector):
  """Load non-coverage checkpoint, add initialized extra variables for coverage, and save as new checkpoint"""
  tf.logging.info("converting non-coverage model to coverage model..")

  # initialize an entire coverage model from scratch
  sess = tf.Session(config=util.get_config())
  print ("initializing everything...")
  sess.run(tf.global_variables_initializer(), feed_dict={model.embedding_place: word_vector})

  # load all non-coverage weights from checkpoint
  saver = tf.train.Saver([v for v in tf.global_variables() if "coverage" not in v.name and "Adagrad" not in v.name])
  print ("restoring non-coverage variables...")
  curr_ckpt = util.load_ckpt(saver, sess)
  print ("restored.")

  # save this model and quit
  ckpt_path = os.path.join(FLAGS.log_root, "train", "model.ckpt_cov")
  step = curr_ckpt.split('-')[1]
  new_fname = ckpt_path + '-' + step + '-init'
  print ("saving model to %s..." % (new_fname))
  new_saver = tf.train.Saver() # this one will save all variables that now exist
  new_saver.save(sess, new_fname)
  print ("saved.")
  exit()

def convert_to_reinforce_model(model,word_vector):
  """Load non-reinforce checkpoint, add initialized extra variables for reinforce, and save as new checkpoint"""
  tf.logging.info("converting non-reinforce model to reinforce model..")

  # initialize an entire reinforce model from scratch
  sess = tf.Session(config=util.get_config())
  print("initializing everything...")
  sess.run(tf.global_variables_initializer(), feed_dict={model.embedding_place: word_vector})

  # load all non-reinforce weights from checkpoint
  saver = tf.train.Saver([v for v in tf.global_variables() if "reinforce" not in v.name and "Adagrad" not in v.name])
  print("restoring non-reinforce variables...")
  curr_ckpt = util.load_ckpt(saver, sess)
  print("restored.")

  # save this model and quit
  ckpt_path = os.path.join(FLAGS.log_root, "train", "model.ckpt_rl")
  step = curr_ckpt.split('-')[1]
  new_fname = ckpt_path + '-' + step + '-init'
  # new_fname = curr_ckpt + '_rl_init'
  print("saving model to %s..." % (new_fname))
  new_saver = tf.train.Saver() # this one will save all variables that now exist
  new_saver.save(sess, new_fname)
  print("saved.")
  exit()


def setup_training(model, batcher, word_vector):
  """Does setup before starting training (run_training)"""
  train_dir = os.path.join(FLAGS.log_root, "train")
  if not os.path.exists(train_dir): os.makedirs(train_dir)

  # default_device = tf.device('/gpu:3')
  # with default_device:
  model.build_graph() # build the graph
  if FLAGS.convert_to_reinforce_model:
    assert (FLAGS.rl_training ), "To convert your pointer model to a reinforce model, run with convert_to_reinforce_model=True and either rl_training=True or ac_training=True"
    convert_to_reinforce_model(model, word_vector)
  if FLAGS.convert_to_coverage_model:
    print("zhuan################################################################################")
    assert FLAGS.coverage, "To convert your non-coverage model to a coverage model, run with convert_to_coverage_model=True and coverage=True"
    convert_to_coverage_model(model, word_vector)
  if FLAGS.restore_best_model:
    restore_best_model()
  saver = tf.train.Saver(max_to_keep=FLAGS.model_max_to_keep) # only keep 1 checkpoint at a time

  # Loads pre-trained word-embedding. By default the mocpkdel learns the embedding.
  # if FLAGS.embedding:
  #   model._vocab.LoadWordEmbedding(FLAGS.embedding, FLAGS.emb_dim)
  #   word_vector = model._vocab.getWordEmbedding()
  #   print("embedding ",word_vector.shape)
  if FLAGS.embedding:
    print("embedding ", word_vector.shape)
  sv = tf.train.Supervisor(logdir=train_dir,
                     is_chief=True,
                     saver=saver,
                     summary_op=None,
                     save_summaries_secs=60, # save summaries for tensorboard every 60 secs
                     save_model_secs=0, # checkpoint every 60 secs
                     global_step=model.global_step,
                     init_feed_dict = {model.embedding_place: word_vector} if FLAGS.embedding else None   # add
                     )
  summary_writer = sv.summary_writer
  tf.logging.info("Preparing or waiting for session...")
  sess_context_manager = sv.prepare_or_wait_for_session(config=util.get_config())
  tf.logging.info("Created session.")

  try:
    run_training(model, batcher, sess_context_manager, sv, summary_writer) # this is an infinite loop until interrupted
  except KeyboardInterrupt:
    tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
    sv.stop()

def run_training(model, batcher, sess_context_manager, sv, summary_writer):
  """Repeatedly runs training iterations, logging loss to screen and writing summaries"""
  tf.logging.info("Starting run_training")

  # if FLAGS.debug:  # start the tensorflow debugger
  #   sess = tf_debug.LocalCLIDebugWrapperSession(sess_context_manager)
  #   sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

  train_step = 0
  # starting the main thread
  tf.logging.info('Starting Seq2Seq run_training...')
  if FLAGS.coverage:
    ckpt_path = os.path.join(FLAGS.log_root, "train", "model.ckpt_cov")
  else:
    ckpt_path = os.path.join(FLAGS.log_root, "train", "model.ckpt")
  with sess_context_manager as sess:
    for _ in range(FLAGS.max_train_iter):  # repeats until interrupted
      batch = batcher.next_batch()
      t0 = time.time()
      results = model.run_train_steps(sess, batch, train_step)
      t1 = time.time()
      # get the summaries and iteration number so we can write summaries to tensorboard
      summaries = results['summaries']  # we will write these summaries to tensorboard using summary_writer
      train_step = results['global_step']  # we need this to update our running average loss
      tf.logging.info('seconds for training step {}: {}'.format(train_step, t1 - t0))

      printer_helper = {}
      printer_helper['pgen_loss'] = results['pgen_loss']
      if FLAGS.coverage:
        printer_helper['coverage_loss'] = results['coverage_loss']
        if FLAGS.rl_training or FLAGS.ac_training:
          printer_helper['rl_cov_total_loss'] = results['reinforce_cov_total_loss']
        printer_helper['pointer_cov_total_loss'] = results['pointer_cov_total_loss']
      if FLAGS.rl_training or FLAGS.ac_training:
        printer_helper['shared_loss'] = results['shared_loss']
        printer_helper['rl_loss'] = results['rl_loss']
        printer_helper['rl_avg_logprobs'] = results['rl_avg_logprobs']
      if FLAGS.rl_training:
        printer_helper['sampled_r'] = np.mean(results['sampled_sentence_r_values'])
        printer_helper['greedy_r'] = np.mean(results['greedy_sentence_r_values'])
        printer_helper['r_diff'] = printer_helper['greedy_r'] - printer_helper['sampled_r']


      for (k, v) in printer_helper.items():
        if not np.isfinite(v):
          raise Exception("{} is not finite. Stopping.".format(k))
        tf.logging.info('{}: {}\t'.format(k, v))
      tf.logging.info('-------------------------------------------')

      summary_writer.add_summary(summaries, train_step)  # write the summaries
      if train_step % 100 == 0:  # flush the summary writer every so often
        summary_writer.flush()
      if train_step % FLAGS.save_model_every == 0:
        sv.saver.save(sess, ckpt_path, global_step=train_step)
      print('Step: ' , train_step)




def run_eval(model, batcher,word_vector):
  """Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far."""
  model.build_graph()  # build the graph
  saver = tf.train.Saver(max_to_keep=3)  # we will keep 3 best checkpoints at a time
  sess = tf.Session(config=util.get_config())

  if FLAGS.embedding:
    sess.run(tf.global_variables_initializer(), feed_dict={model.embedding_place: word_vector})
  eval_dir = os.path.join(FLAGS.log_root, "eval")  # make a subdir of the root dir for eval data
  bestmodel_save_path = os.path.join(eval_dir, 'bestmodel')  # this is where checkpoints of best models are saved
  summary_writer = tf.summary.FileWriter(eval_dir)
  running_avg_loss = 0  # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
  best_loss = restore_best_eval_model()  # will hold the best loss achieved so far
  train_step = 0

  while True:
    _ = util.load_ckpt(saver, sess)  # load a new checkpoint
    processed_batch = 0
    avg_losses = []
    # evaluate for 100 * batch_size before comparing the loss
    # we do this due to memory constraint, best to run eval on different machines with large batch size
    # while processed_batch < 100 * FLAGS.batch_size:
    processed_batch += FLAGS.batch_size
    batch = batcher.next_batch()  # get the next batch
    tf.logging.info('run eval step on seq2seq model.')
    t0 = time.time()
    results = model.run_eval_step(sess, batch, train_step)
    t1 = time.time()

    tf.logging.info('experiment: {}'.format(FLAGS.exp_name))
    tf.logging.info('processed_batch: {}, seconds for batch: {}'.format(processed_batch, t1 - t0))

    printer_helper = {}
    loss = printer_helper['pgen_loss'] = results['pgen_loss']
    if FLAGS.coverage:
      printer_helper['coverage_loss'] = results['coverage_loss']
      if FLAGS.rl_training or FLAGS.ac_training:
        printer_helper['rl_cov_total_loss'] = results['reinforce_cov_total_loss']
      loss = printer_helper['pointer_cov_total_loss'] = results['pointer_cov_total_loss']
    if FLAGS.rl_training or FLAGS.ac_training:
      printer_helper['shared_loss'] = results['shared_loss']
      printer_helper['rl_loss'] = results['rl_loss']
      printer_helper['rl_avg_logprobs'] = results['rl_avg_logprobs']
    if FLAGS.rl_training:
      printer_helper['sampled_r'] = np.mean(results['sampled_sentence_r_values'])
      printer_helper['greedy_r'] = np.mean(results['greedy_sentence_r_values'])
      printer_helper['r_diff'] = printer_helper['greedy_r'] - printer_helper['sampled_r']

    for (k, v) in printer_helper.items():
      if not np.isfinite(v):
        raise Exception("{} is not finite. Stopping.".format(k))
      tf.logging.info('{}: {}\t'.format(k, v))

      # add summaries
      summaries = results['summaries']
      train_step = results['global_step']
      print(train_step)
      summary_writer.add_summary(summaries, train_step)

      # calculate running avg loss
      avg_losses.append(calc_running_avg_loss(np.asscalar(loss), running_avg_loss, train_step, summary_writer))
      tf.logging.info('-------------------------------------------')

    running_avg_loss = np.mean(avg_losses)
    tf.logging.info('==========================================')
    tf.logging.info('best_loss: {}\trunning_avg_loss: {}\t'.format(best_loss, running_avg_loss))
    tf.logging.info('==========================================')

    # If running_avg_loss is best so far, save this checkpoint (early stopping).
    # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
    if best_loss is None or running_avg_loss < best_loss:
      print(train_step)
      tf.logging.info('Found new best model with %.3f running_avg_loss. Saving to %s', running_avg_loss,
                      bestmodel_save_path)
      saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
      best_loss = running_avg_loss

    # flush the summary writer every so often
    if train_step % 100 == 0:
      summary_writer.flush()
    # time.sleep(600) # run eval every 10 minute

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
