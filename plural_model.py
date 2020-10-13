from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import math
import json
import threading
import numpy as np
import tensorflow as tf
import h5py

import util

class PluralModel(object):
  def __init__(self, config):
    self.config = config
    self.context_embeddings = util.EmbeddingDictionary(config["context_embeddings"])
    self.head_embeddings = util.EmbeddingDictionary(config["head_embeddings"], maybe_cache=self.context_embeddings)
    self.char_embedding_size = config["char_embedding_size"]
    self.char_dict = util.load_char_dict(config["char_vocab_path"])
    if config["lm_path"]:
      self.lm_file = h5py.File(self.config["lm_path"], "r")
    else:
      self.lm_file = None
    self.lm_layers = self.config["lm_layers"]
    self.lm_size = self.config["lm_size"]
    self.eval_data = None # Load eval data lazily.

    input_props = []
    input_props.append((tf.float32, [None, None, self.context_embeddings.size])) # Context embeddings.
    input_props.append((tf.float32, [None, None, self.head_embeddings.size])) # Head embeddings.
    input_props.append((tf.float32, [None, None, self.lm_size, self.lm_layers])) # LM embeddings.
    input_props.append((tf.int32, [None, None, None])) # Character indices.
    input_props.append((tf.int32, [None])) # Text lengths.
    input_props.append((tf.bool, [])) # Is training.
    input_props.append((tf.int32, [None])) # Gold starts.
    input_props.append((tf.int32, [None])) # Gold ends.
    input_props.append((tf.int32, [None,None])) # antecedents
    input_props.append((tf.int32, [None])) #antecedent_len
    input_props.append((tf.int32,[None])) #anaphors
    input_props.append((tf.bool,[None,None])) # gold labels

    self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
    dtypes, shapes = zip(*input_props)
    queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
    self.enqueue_op = queue.enqueue(self.queue_input_tensors)
    self.input_tensors = queue.dequeue()

    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    self.reset_global_step = tf.assign(self.global_step, 0)

    self.predictions, self.loss = self.get_predictions_and_loss(*self.input_tensors)
    learning_rate = tf.train.exponential_decay(self.config["learning_rate"], self.global_step,
                                               self.config["decay_frequency"], self.config["decay_rate"], staircase=True)
    trainable_params = tf.trainable_variables()
    gradients = tf.gradients(self.loss, trainable_params)
    gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_gradient_norm"])
    optimizers = {
      "adam" : tf.train.AdamOptimizer,
      "sgd" : tf.train.GradientDescentOptimizer
    }
    optimizer = optimizers[self.config["optimizer"]](learning_rate)
    self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)


  def start_enqueue_thread(self, session):
    train_examples = []
    auxiliary_train_examples = []
    use_teacher_annealing = self.config["use_teacher_annealing"]
    total_plural_doc, use_plural_doc = 0,0
    total_2nd_doc,use_2nd_doc = 0,0
    auxiliary_train_type = self.config["auxiliary_train_type"] #[plural, coref, bridging]

    if self.config["train_path"]: #the main train_path can only be the plural
      for line in open(self.config["train_path"]):
        doc = json.loads(line)
        doc['mode'] = 'plural'
        total_plural_doc+=1
        if len(doc['plurals']) > 0:
          use_plural_doc+=1
          train_examples.append(doc)

    if self.config["auxiliary_train_path"]:
      for line in open(self.config["auxiliary_train_path"]):
        doc = json.loads(line)
        doc['mode'] = auxiliary_train_type
        total_2nd_doc += 1
        if (auxiliary_train_type == 'plural' and len(doc['plurals']) > 0) or \
              (auxiliary_train_type == 'bridging' and len(doc['bridgings'])>0) or \
              auxiliary_train_type == 'coref':
          use_2nd_doc += 1
          auxiliary_train_examples.append(doc)


    print('Find %d plural train documents use %d.' %(total_plural_doc,use_plural_doc))
    if self.config["auxiliary_train_path"]:
      print('Find %d %s train documents use %d from auxiliary train path.' %(total_2nd_doc,auxiliary_train_type,use_2nd_doc))

    if not train_examples:
      train_examples = auxiliary_train_examples
      auxiliary_train_examples = []

    def _enqueue_loop():
      max_step = self.config["max_step"]
      curr_step = 0
      train_ind = 0
      train_2nd_ind = 0

      random.shuffle(train_examples)
      if auxiliary_train_examples:
        random.shuffle(auxiliary_train_examples)
      while True:
        main_train_ratio = 1.0 if not auxiliary_train_examples else \
          (0.5 if not use_teacher_annealing else curr_step*1.0/max_step)

        if random.random() <= main_train_ratio:
          example = train_examples[train_ind]
          train_ind+=1
          if train_ind >= len(train_examples):
            random.shuffle(train_examples)
            train_ind = 0
        else:
          example = auxiliary_train_examples[train_2nd_ind]
          train_2nd_ind+=1
          if train_2nd_ind >= len(auxiliary_train_examples):
            random.shuffle(auxiliary_train_examples)
            train_2nd_ind = 0

        tensorized_example = self.tensorize_example(example, is_training=True)
        feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
        session.run(self.enqueue_op, feed_dict=feed_dict)
        curr_step+=1
    enqueue_thread = threading.Thread(target=_enqueue_loop)
    enqueue_thread.daemon = True
    enqueue_thread.start()

  def restore(self, session,model_file_name = 'model.max.ckpt'):
    # Don't try to restore unused variables from the TF-Hub ELMo module.
    vars_to_restore = [v for v in tf.global_variables() if "module/" not in v.name]
    saver = tf.train.Saver(vars_to_restore)
    checkpoint_path = os.path.join(self.config["log_dir"], model_file_name)
    print("Restoring from {}".format(checkpoint_path))
    session.run(tf.global_variables_initializer())
    saver.restore(session, checkpoint_path)

  def load_lm_embeddings(self, doc_key):
    if self.lm_file is None:
      return np.zeros([0, 0, self.lm_size, self.lm_layers])
    file_key = doc_key.replace("/", ":")
    group = self.lm_file[file_key]
    num_sentences = len(list(group.keys()))
    sentences = [group[str(i)][...] for i in range(num_sentences)]
    lm_emb = np.zeros([num_sentences, max(s.shape[0] for s in sentences), self.lm_size, self.lm_layers])
    for i, s in enumerate(sentences):
      lm_emb[i, :s.shape[0], :, :] = s
    return lm_emb

  def tensorize_example(self, example, is_training):
    clusters = example["clusters"]
    doc_mode = 'plural' if not 'mode' in example else example['mode']
    gold_mentions = [tuple(m) for m in example['mentions']]
    gold_starts = np.array([s for s,_ in gold_mentions])
    gold_ends = np.array([e for _,e in gold_mentions])
    cluster_ids = [-1] * len(gold_mentions)
    for cluster_id, cluster in enumerate(clusters):
      for mid in cluster:
        cluster_ids[mid] = cluster_id

    sentences = example["sentences"]
    num_words = sum(len(s) for s in sentences)

    max_sentence_length = max(len(s) for s in sentences)
    max_word_length = max(max(max(len(w) for w in s) for s in sentences), max(self.config["filter_widths"]))
    text_len = np.array([len(s) for s in sentences])
    tokens = [[""] * max_sentence_length for _ in sentences]
    context_word_emb = np.zeros([len(sentences), max_sentence_length, self.context_embeddings.size])
    head_word_emb = np.zeros([len(sentences), max_sentence_length, self.head_embeddings.size])
    char_index = np.zeros([len(sentences), max_sentence_length, max_word_length])
    for i, sentence in enumerate(sentences):
      for j, word in enumerate(sentence):
        tokens[i][j] = word
        context_word_emb[i, j] = self.context_embeddings[word]
        head_word_emb[i, j] = self.head_embeddings[word]
        char_index[i, j, :len(word)] = [self.char_dict[c] for c in word]
    tokens = np.array(tokens)

    doc_key = example["doc_key"]

    plural_bridging_cluster_pairs = {} #for bridging and plural
    if doc_mode == 'plural':
      for mid, aid in example['plurals']:
        msid = cluster_ids[mid]
        asid = cluster_ids[aid]
        if not msid in plural_bridging_cluster_pairs:
          plural_bridging_cluster_pairs[msid] = set()
          plural_bridging_cluster_pairs[msid].add(asid)
    elif doc_mode == 'bridging':
      for mid, aid, is_inv in example['bridgings']:
        if is_inv:#for element-of-inv the antecedent is equavalent to plural
          msid = cluster_ids[mid]
          asid = cluster_ids[aid]
        else:
          msid = cluster_ids[aid]
          asid = cluster_ids[mid]
        if not msid in plural_bridging_cluster_pairs:
          plural_bridging_cluster_pairs[msid] = set()
          plural_bridging_cluster_pairs[msid].add(asid)


    num_anaphors = sum(len(cl) for cid, cl in enumerate(clusters) if cid in plural_bridging_cluster_pairs)
    negative_example_rate = 1 if is_training else 0
    if doc_mode != 'coref' and is_training and self.config["ntimes_negative_examples"] > 0:
      negative_example_rate = max(0.2, num_anaphors*self.config["ntimes_negative_examples"]/(len(gold_mentions)-num_anaphors))

    lm_emb = self.load_lm_embeddings(doc_key)
    max_training_words = self.config["max_training_words"] if doc_mode=='plural' else 1000  #for coref/bridging 1000 words maximum

    if is_training and num_words > max_training_words:
      context_word_emb, head_word_emb, lm_emb, char_index, text_len, is_training, gold_starts, gold_ends,np_cluster_ids\
        = self.truncate_example(context_word_emb, head_word_emb, lm_emb, char_index, text_len, is_training, gold_starts, gold_ends, np.array(cluster_ids),max_training_words)
      cluster_ids = [id.item() for id in np_cluster_ids]

    num_mention = gold_starts.shape[0]
    anaphors = np.arange(num_mention)
    num_top_antecedents = min(num_mention, self.config['max_top_antecedents'])
    antecedents = np.zeros([num_mention, num_top_antecedents])
    gold_labels = np.zeros([num_mention, num_top_antecedents], dtype=np.bool)
    antecedents_len = np.zeros([num_mention])
    us_mask = np.ones([num_mention],np.bool)

    for i in xrange(num_mention):
      antecedents_len[i] = min(i, num_top_antecedents)
      cid = cluster_ids[i]
      if doc_mode == 'coref':
        for j in xrange(min(i, num_top_antecedents)):
          ant = i - j - 1
          antecedents[i, j] = ant
          if cluster_ids[i] == cluster_ids[ant]:
            gold_labels[i, j] = True
      elif doc_mode == 'plural':
        us_mask[i] = cid in plural_bridging_cluster_pairs or random.random() < negative_example_rate
        if not us_mask[i]:
          continue
        for j in xrange(min(i, num_top_antecedents)):
          ant = i - j - 1
          antecedents[i, j] = ant
          if cid in plural_bridging_cluster_pairs and cluster_ids[ant] in plural_bridging_cluster_pairs[cid]:
            gold_labels[i, j] = True
      elif doc_mode == 'bridging':
        # for bridging we seach antecedent from both direction as
        # we switch the antecedent and anaphors for element-of relations
        # for element-of-inv we do not switch the position
        us_mask[i] = cid in plural_bridging_cluster_pairs or random.random() < negative_example_rate
        if not us_mask[i]:
          continue
        nbefore = min(i, num_top_antecedents // 2)
        mafter = min(num_mention - i - 1, num_top_antecedents // 2)
        antecedents_len[i] = nbefore + mafter
        for j in xrange(nbefore):
          ant = i - j - 1
          antecedents[i, j] = ant
          if cid in plural_bridging_cluster_pairs and cluster_ids[ant] in plural_bridging_cluster_pairs[cid]:
            gold_labels[i, j] = True
        for j in xrange(mafter):
          ant = i + j + 1
          antecedents[i, j + nbefore] = ant
          if cid in plural_bridging_cluster_pairs and cluster_ids[ant] in plural_bridging_cluster_pairs[cid]:
            gold_labels[i, j + nbefore] = True

    anaphors = anaphors[us_mask]
    antecedents = antecedents[us_mask]
    antecedents_len = antecedents_len[us_mask]
    gold_labels = gold_labels[us_mask]

    return (context_word_emb, head_word_emb, lm_emb, char_index, text_len, is_training, gold_starts, gold_ends, antecedents,antecedents_len,anaphors,gold_labels)


  def truncate_example(self, context_word_emb, head_word_emb, lm_emb, char_index, text_len, is_training, gold_starts, gold_ends, cluster_ids,max_training_words):
    num_words = sum(text_len)
    assert num_words > max_training_words
    num_sentences = context_word_emb.shape[0]
    max_training_sentences = num_sentences


    while num_words > max_training_words:
      max_training_sentences -= 1
      sentence_offset = random.randint(0, num_sentences - max_training_sentences)
      word_offset = text_len[:sentence_offset].sum()
      num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()


    context_word_emb = context_word_emb[sentence_offset:sentence_offset + max_training_sentences, :, :]
    head_word_emb = head_word_emb[sentence_offset:sentence_offset + max_training_sentences, :, :]
    lm_emb = lm_emb[sentence_offset:sentence_offset + max_training_sentences, :, :, :]
    char_index = char_index[sentence_offset:sentence_offset + max_training_sentences, :, :]
    text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

    gold_spans = np.logical_and(gold_starts >= word_offset, gold_ends < word_offset + num_words)
    gold_starts = gold_starts[gold_spans] - word_offset
    gold_ends = gold_ends[gold_spans] - word_offset
    cluster_ids = cluster_ids[gold_spans]

    return context_word_emb, head_word_emb, lm_emb, char_index, text_len, is_training, gold_starts, gold_ends, cluster_ids


  def get_dropout(self, dropout_rate, is_training):
    return 1 - (tf.to_float(is_training) * dropout_rate)



  def get_predictions_and_loss(self, context_word_emb, head_word_emb, lm_emb, char_index, text_len, is_training, gold_starts, gold_ends, antecedents,antecedents_len,anaphors, gold_labels):
    self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
    self.lexical_dropout = self.get_dropout(self.config["lexical_dropout_rate"], is_training)
    self.lstm_dropout = self.get_dropout(self.config["lstm_dropout_rate"], is_training)

    num_sentences = tf.shape(context_word_emb)[0]
    max_sentence_length = tf.shape(context_word_emb)[1]

    context_emb_list = [context_word_emb]
    head_emb_list = [head_word_emb]

    if self.config["char_embedding_size"] > 0:
      char_emb = tf.gather(tf.get_variable("char_embeddings", [len(self.char_dict), self.config["char_embedding_size"]]), char_index) # [num_sentences, max_sentence_length, max_word_length, emb]
      flattened_char_emb = tf.reshape(char_emb, [num_sentences * max_sentence_length, util.shape(char_emb, 2), util.shape(char_emb, 3)]) # [num_sentences * max_sentence_length, max_word_length, emb]
      flattened_aggregated_char_emb = util.cnn(flattened_char_emb, self.config["filter_widths"], self.config["filter_size"]) # [num_sentences * max_sentence_length, emb]
      aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb, [num_sentences, max_sentence_length, util.shape(flattened_aggregated_char_emb, 1)]) # [num_sentences, max_sentence_length, emb]
      context_emb_list.append(aggregated_char_emb)
      head_emb_list.append(aggregated_char_emb)

    if self.lm_file:
      lm_emb_size = util.shape(lm_emb, 2)
      lm_num_layers = util.shape(lm_emb, 3)
      with tf.variable_scope("lm_aggregation"):
        self.lm_weights = tf.nn.softmax(tf.get_variable("lm_scores", [lm_num_layers], initializer=tf.constant_initializer(0.0)))
        self.lm_scaling = tf.get_variable("lm_scaling", [], initializer=tf.constant_initializer(1.0))
      flattened_lm_emb = tf.reshape(lm_emb, [num_sentences * max_sentence_length * lm_emb_size, lm_num_layers])
      flattened_aggregated_lm_emb = tf.matmul(flattened_lm_emb, tf.expand_dims(self.lm_weights, 1)) # [num_sentences * max_sentence_length * emb, 1]
      aggregated_lm_emb = tf.reshape(flattened_aggregated_lm_emb, [num_sentences, max_sentence_length, lm_emb_size])
      aggregated_lm_emb *= self.lm_scaling
      context_emb_list.append(aggregated_lm_emb)

    context_emb = tf.concat(context_emb_list, 2) # [num_sentences, max_sentence_length, emb]
    head_emb = tf.concat(head_emb_list, 2) # [num_sentences, max_sentence_length, emb]
    context_emb = tf.nn.dropout(context_emb, self.lexical_dropout) # [num_sentences, max_sentence_length, emb]
    head_emb = tf.nn.dropout(head_emb, self.lexical_dropout) # [num_sentences, max_sentence_length, emb]

    text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length) # [num_sentence, max_sentence_length]

    context_outputs = self.lstm_contextualize(context_emb, text_len, text_len_mask) # [num_words, emb]

    flattened_head_emb = self.flatten_emb_by_sentence(head_emb, text_len_mask) # [num_words]


    mention_emb = self.get_span_emb(flattened_head_emb, context_outputs, gold_starts, gold_ends)


    k=util.shape(antecedents,0)
    c = util.shape(antecedents,1)

    anaphor_emb = tf.gather(mention_emb,anaphors) #[k,emb]
    antecedent_emb = tf.gather(mention_emb, antecedents) # [k, c, emb]


    pair_emb = self.get_pair_embeddings(anaphor_emb, antecedents, antecedent_emb) # [k, c,emb]


    with tf.variable_scope("plural_scores"):
      plural_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [k, c, 1]
      plural_scores = tf.squeeze(plural_scores, 2) # [k, c]
      plural_scores = plural_scores + tf.log(tf.sequence_mask(antecedents_len,c,tf.float32))

    dummy_scores = tf.zeros([k, 1])
    dummy_labels = tf.logical_not(tf.reduce_any(gold_labels, 1, keepdims=True))  # [k, 1]

    plural_scores_with_dummy = tf.concat([dummy_scores, plural_scores], 1)
    gold_labels_with_dummy = tf.concat([dummy_labels, gold_labels], 1)

    loss = self.softmax_loss(plural_scores_with_dummy,gold_labels_with_dummy)
    loss = tf.reduce_sum(loss)

    return [plural_scores,antecedents_len,anaphors], loss

  def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends):
    span_emb_list = []

    span_start_emb = tf.gather(context_outputs, span_starts) # [k, emb]
    span_emb_list.append(span_start_emb)

    span_end_emb = tf.gather(context_outputs, span_ends) # [k, emb]
    span_emb_list.append(span_end_emb)

    span_width = tf.minimum(1 + span_ends - span_starts,self.config["max_span_width"]) # [k]

    if self.config["use_features"]:
      span_width_index = span_width - 1 # [k]
      span_width_emb = tf.gather(tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]]), span_width_index) # [k, emb]
      span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
      span_emb_list.append(span_width_emb)

    if self.config["model_heads"]:
      span_indices = tf.expand_dims(tf.range(self.config["max_span_width"]), 0) + tf.expand_dims(span_starts, 1) # [k, max_span_width]
      span_indices = tf.minimum(util.shape(context_outputs, 0) - 1, span_indices) # [k, max_span_width]
      span_text_emb = tf.gather(head_emb, span_indices) # [k, max_span_width, emb]
      with tf.variable_scope("head_scores"):
        self.head_scores = util.projection(context_outputs, 1) # [num_words, 1]
      span_head_scores = tf.gather(self.head_scores, span_indices) # [k, max_span_width, 1]
      span_mask = tf.expand_dims(tf.sequence_mask(span_width, self.config["max_span_width"], dtype=tf.float32), 2) # [k, max_span_width, 1]
      span_head_scores += tf.log(span_mask) # [k, max_span_width, 1]
      span_attention = tf.nn.softmax(span_head_scores, 1) # [k, max_span_width, 1]
      span_head_emb = tf.reduce_sum(span_attention * span_text_emb, 1) # [k, emb]
      span_emb_list.append(span_head_emb)

    span_emb = tf.concat(span_emb_list, 1) # [k, emb]
    return span_emb # [k, emb]


  def softmax_loss(self, antecedent_scores, antecedent_labels):
    gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels)) # [k, max_ant + 1]
    marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1]) # [k]
    log_norm = tf.reduce_logsumexp(antecedent_scores, [1]) # [k]
    return log_norm - marginalized_gold_scores # [k]

  def bucket_distance(self, distances):
    """
    Places the given values (designed for distances) into 10 semi-logscale buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
    """
    logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances))/math.log(2))) + 3
    use_identity = tf.to_int32(distances <= 4)
    combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
    return tf.clip_by_value(combined_idx, 0, 9)

  def get_pair_embeddings(self, mention_emb, antecedents, antecedent_emb):
    k = util.shape(mention_emb, 0)
    c = util.shape(antecedents, 1)

    feature_emb_list = []
    antecedent_offsets = tf.tile(tf.expand_dims(tf.range(c) + 1, 0), [k, 1]) # [k, c]


    if self.config["use_features"]:
      antecedent_distance_buckets = self.bucket_distance(antecedent_offsets) # [k, c]
      antecedent_distance_emb = tf.gather(tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]]), antecedent_distance_buckets) # [k, c]
      feature_emb_list.append(antecedent_distance_emb)

    feature_emb = tf.concat(feature_emb_list, 2) # [k, c, emb]
    feature_emb = tf.nn.dropout(feature_emb, self.dropout) # [k, c, emb]

    target_emb = tf.expand_dims(mention_emb, 1) # [k, 1, emb]
    similarity_emb = antecedent_emb * target_emb # [k, c, emb]
    target_emb = tf.tile(target_emb, [1, c, 1]) # [k, c, emb]

    pair_emb = tf.concat([target_emb, antecedent_emb, similarity_emb, feature_emb], 2) # [k, c, emb]


    return pair_emb


  def flatten_emb_by_sentence(self, emb, text_len_mask):
    num_sentences = tf.shape(emb)[0]
    max_sentence_length = tf.shape(emb)[1]

    emb_rank = len(emb.get_shape())
    if emb_rank  == 2:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
    elif emb_rank == 3:
      flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
    else:
      raise ValueError("Unsupported rank: {}".format(emb_rank))
    return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))

  def lstm_contextualize(self, text_emb, text_len, text_len_mask):
    num_sentences = tf.shape(text_emb)[0]

    current_inputs = text_emb # [num_sentences, max_sentence_length, emb]

    for layer in xrange(self.config["contextualization_layers"]):
      with tf.variable_scope("layer_{}".format(layer)):
        with tf.variable_scope("fw_cell"):
          cell_fw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences, self.lstm_dropout)
        with tf.variable_scope("bw_cell"):
          cell_bw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences, self.lstm_dropout)
        state_fw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [num_sentences, 1]), tf.tile(cell_fw.initial_state.h, [num_sentences, 1]))
        state_bw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [num_sentences, 1]), tf.tile(cell_bw.initial_state.h, [num_sentences, 1]))

        (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=cell_fw,
          cell_bw=cell_bw,
          inputs=current_inputs,
          sequence_length=text_len,
          initial_state_fw=state_fw,
          initial_state_bw=state_bw)

        text_outputs = tf.concat([fw_outputs, bw_outputs], 2) # [num_sentences, max_sentence_length, emb]
        text_outputs = tf.nn.dropout(text_outputs, self.lstm_dropout)
        if layer > 0:
          highway_gates = tf.sigmoid(util.projection(text_outputs, util.shape(text_outputs, 2))) # [num_sentences, max_sentence_length, emb]
          text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs
        current_inputs = text_outputs

    return self.flatten_emb_by_sentence(text_outputs, text_len_mask)

  def get_plural_pairs(self, plural_scores,antecedent_len,anaphors, cluster_ids, plural_mentions):
    plural_pairs = []
    plural_maps = {}
    for i, mid in enumerate(anaphors):
      if not mid in plural_mentions:
        continue
      score_list = []
      ant_cid_list = set()
      for j in xrange(antecedent_len[i]):
        ant = mid - j -1
        score_list.append((plural_scores[i,j].item(), ant))
      score_list = sorted(score_list,reverse=True)
      for s, ant in score_list:
        ant_cid = cluster_ids[ant]
        if len(ant_cid_list) >=5:
          break
        elif ant_cid in ant_cid_list:
          continue
        elif s > 0:
          ant_cid_list.add(ant_cid)
        elif len(ant_cid_list) < 2:
          ant_cid_list.add(ant_cid)
      plural_maps[mid] = ant_cid_list
      for cid in ant_cid_list:
        plural_pairs.append((mid, cid))
    return set(plural_pairs),plural_maps



  def load_eval_data(self):
    if self.eval_data is None:
      def load_line(line):
        example = json.loads(line)
        return self.tensorize_example(example, is_training=False), example
      with open(self.config["eval_path"]) as f:
        self.eval_data = [load_line(l) for l in f.readlines()]
      print("Loaded {} eval examples.".format(len(self.eval_data)))


  def evaluate(self, session):
    self.load_eval_data()

    tp,fn,fp = 0,0,0
    emp,emt = 0,0

    for example_num, (tensorized_example, example) in enumerate(self.eval_data):
      feed_dict = {i:t for i,t in zip(self.input_tensors, tensorized_example)}
      plural_scores,antecedent_len,anaphors = session.run(self.predictions, feed_dict=feed_dict)

      num_of_mention = len(example['mentions'])
      clusters = example['clusters']
      cluster_ids = [-1] * num_of_mention
      for cid, cl in enumerate(clusters):
        for m in cl:
          cluster_ids[m] = cid

      plural_mentions = set(mid for mid,_ in example['plurals'])
      gold_plural_maps = {}
      for mid, ant in example['plurals']:
        if mid not in gold_plural_maps:
          gold_plural_maps[mid] = set()
        gold_plural_maps[mid].add(cluster_ids[ant])

      gold_plurals = set((mid, cluster_ids[ant]) for mid, ant in example['plurals'])
      pred_plurals, pred_plural_maps = self.get_plural_pairs(plural_scores,antecedent_len,anaphors,cluster_ids,plural_mentions)

      tp += len(gold_plurals & pred_plurals)
      fn += len(gold_plurals - pred_plurals)
      fp += len(pred_plurals - gold_plurals)


      emt += len(gold_plural_maps)
      for mid in gold_plural_maps:
        if len(gold_plural_maps[mid]) == len(pred_plural_maps[mid])==len(gold_plural_maps[mid] & pred_plural_maps[mid]):
          emp+=1


      if example_num % 10 == 0:
        print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))

    recall = 0.0 if tp == 0 else float(tp)/(tp+fn)
    precision = 0.0 if tp == 0 else float(tp)/(tp+fp)
    f1 = 0.0 if precision == 0.0 else 2.0*recall*precision/(recall+precision)

    print("Plural F1: {:.2f}%".format(f1*100))
    print("Plural recall: {:.2f}%".format(recall*100))
    print("Plural precision: {:.2f}%".format(precision*100))
    print("Plural exact match accrucy: {:.2f}%".format(emp*100.0/emt))

    summary_dict = {}
    summary_dict["Plural F1"] = f1
    summary_dict["Plural recall"] = recall
    summary_dict["Plural precision"] = precision

    return util.make_summary(summary_dict), f1*100
