#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""This file contains code to run beam search decoding"""

import tensorflow as tf
import numpy as np
import data

FLAGS = tf.app.flags.FLAGS

class Hypothesis(object):
  """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

  def __init__(self, tokens, log_probs, state, decoder_output, encoder_mask, attn_dists_norescale, attn_dists, p_gens, context_vector, coverage):
    """Hypothesis constructor.

    Args:
      tokens: List of integers. The ids of the tokens that form the summary so far.
      log_probs: List, same length as tokens, of floats, giving the log probabilities of the tokens so far.
      state: Current state of the decoder, a LSTMStateTuple.
      attn_dists: List, same length as tokens, of numpy arrays with shape (attn_length). These are the attention distributions so far.
      p_gens: List, same length as tokens, of floats, or None if not using pointer-generator model. The values of the generation probability so far.
      coverage: Numpy array of shape (attn_length), or None if not using coverage. The current coverage vector.
    """
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.decoder_output = decoder_output
    self.encoder_mask = encoder_mask
    self.attn_dists_norescale = attn_dists_norescale
    self.attn_dists = attn_dists
    self.p_gens = p_gens
    self.context_vector = context_vector
    self.coverage = coverage

  def extend(self, token, log_prob, state, decoder_output, encoder_mask, attn_dist_norescale, attn_dist, p_gen, context_vector, coverage):
    """Return a NEW hypothesis, extended with the information from the latest step of beam search.

    Args:
      token: Integer. Latest token produced by beam search.
      log_prob: Float. Log prob of the latest token.
      state: Current decoder state, a LSTMStateTuple.
      attn_dist: Attention distribution from latest step. Numpy array shape (attn_length).
      p_gen: Generation probability on latest step. Float.
      coverage: Latest coverage vector. Numpy array shape (attn_length), or None if not using coverage.
    Returns:
      New Hypothesis for next step.
    """
    return Hypothesis(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      decoder_output=self.decoder_output + [decoder_output] if decoder_output is not None else [],
                      encoder_mask=self.encoder_mask + [encoder_mask] if encoder_mask is not None else [],
                      attn_dists_norescale = self.attn_dists_norescale + [attn_dist_norescale],
                      attn_dists = self.attn_dists + [attn_dist],
                      p_gens = self.p_gens + [p_gen],
                      context_vector = context_vector,
                      coverage = coverage)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def log_prob(self):
    # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
    return sum(self.log_probs)

  @property
  def avg_log_prob(self):
    # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
    return self.log_prob / len(self.tokens)


def run_beam_search(sess, model, vocab, batch):
  """Performs beam search decoding on the given example.

  Args:
    sess: a tf.Session
    model: a seq2seq model
    vocab: Vocabulary object
    batch: Batch object that is the same example repeated across the batch

  Returns:
    best_hyp: Hypothesis object; the best hypothesis found by beam search.
  """
  # Run the encoder to get the encoder hidden states and decoder initial state
  output = model._selector.run_eval_step(sess, batch, probs_only=True)
  selector_probs = output['probs']
  enc_states, dec_in_state = model._rewriter.run_encoder(sess, batch)
  # dec_in_state is a LSTMStateTuple
  # enc_states has shape [batch_size, <=max_enc_steps, 2*hidden_dim].

  # Initialize beam_size-many hyptheses
  hyps = [Hypothesis(tokens=[vocab.word2id(data.START_DECODING)],   # first token is the start token
                     log_probs=[0.0],   # first probability is 1.0 (start token)
                     state=dec_in_state,
                     decoder_output=[np.zeros([FLAGS.dec_hidden_dim])],
                     encoder_mask=[np.zeros([batch.enc_batch.shape[1]])],
                     attn_dists_norescale=[],
                     attn_dists=[],
                     p_gens=[],
                     context_vector=np.zeros([enc_states.shape[2]]),
                     coverage=np.zeros([batch.enc_batch.shape[1]]) # zero vector of length attention_length
                     ) for _ in range(FLAGS.beam_size)]
  results = [] # this will contain finished hypotheses (those that have emitted the [STOP] token)

  steps = 0
  while steps < FLAGS.max_dec_steps and len(results) < FLAGS.beam_size:
    latest_tokens = [h.latest_token for h in hyps] # latest token produced by each hypothesis
    latest_tokens = [t if t in range(vocab.size()) else vocab.word2id(data.UNKNOWN_TOKEN) for t in latest_tokens] # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings
    states = [h.state for h in hyps] # list of current decoder states of the hypotheses
    prev_coverage = [h.coverage for h in hyps] # list of coverage vectors (or None)
    prev_context = [h.context_vector for h in hyps]
    decoder_outputs = np.array([h.decoder_output for h in hyps]).swapaxes(0, 1) # shape (?, batch_size, dec_hidden_dim)
    encoder_es = np.array([h.encoder_mask for h in hyps]).swapaxes(0, 1)  # shape (?, batch_size, enc_hidden_dim)
    # Run one step of the decoder to get the new info
    (topk_ids, topk_log_probs, new_states, attn_dists_norescale, attn_dists, new_context, p_gens, new_coverage,decoder_output, encoder_e) = \
                        model._rewriter.decode_onestep(sess=sess,
                                             batch=batch,
                                             latest_tokens=latest_tokens,
                                             enc_states=enc_states,
                                             dec_init_states=states,
                                             prev_coverage=prev_coverage,
                                             prev_decoder_outputs=decoder_outputs if FLAGS.intradecoder else tf.stack(
                                               [], axis=0),
                                             prev_encoder_es=encoder_es if FLAGS.use_temporal_attention else tf.stack(
                                               [], axis=0),
                                             selector_probs=selector_probs)

    # Extend each hypothesis and collect them all in all_hyps
    all_hyps = []
    num_orig_hyps = 1 if steps == 0 else len(hyps) # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
    for i in range(num_orig_hyps):
      h, new_state, attn_dist_norescale, attn_dist, new_context_i, p_gen, new_coverage_i = hyps[i], new_states[i], attn_dists_norescale[i], attn_dists[i], new_context[i], p_gens[i], new_coverage[i]  # take the ith hypothesis and new decoder state info
      decoder_output_i = None
      encoder_mask_i = None
      if FLAGS.intradecoder:
        decoder_output_i = decoder_output[i]
      if FLAGS.use_temporal_attention:
        encoder_mask_i = encoder_e[i]
      for j in range(FLAGS.beam_size * 2):  # for each of the top 2*beam_size hyps:
        # Extend the ith hypothesis with the jth option
        new_hyp = h.extend(token=topk_ids[i, j],
                           log_prob=topk_log_probs[i, j],
                           state=new_state,
                           decoder_output=decoder_output_i,
                           encoder_mask=encoder_mask_i,
                           attn_dist_norescale=attn_dist_norescale,
                           attn_dist=attn_dist,
                           p_gen=p_gen,
                           context_vector=new_context_i,
                           coverage=new_coverage_i)
        all_hyps.append(new_hyp)

    # Filter and collect any hypotheses that have produced the end token.
    hyps = [] # will contain hypotheses for the next step
    for h in sort_hyps(all_hyps): # in order of most likely h
      if h.latest_token == vocab.word2id(data.STOP_DECODING): # if stop token is reached...
        # If this hypothesis is sufficiently long, put in results. Otherwise discard.
        if steps >= FLAGS.min_dec_steps:
          results.append(h)
      else: # hasn't reached stop token, so continue to extend this hypothesis
        hyps.append(h)
      if len(hyps) == FLAGS.beam_size or len(results) == FLAGS.beam_size:
        # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop.
        break

    steps += 1

  # At this point, either we've got beam_size results, or we've reached maximum decoder steps

  if len(results)==0: # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
    results = hyps

  # Sort hypotheses by average log probability
  hyps_sorted = sort_hyps(results)

  # Return the hypothesis with highest average log prob
  return hyps_sorted[0]

def sort_hyps(hyps):
  """Return a list of Hypothesis objects, sorted by descending average log probability"""
  return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)
