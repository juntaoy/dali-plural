#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import plural_model as pm
import util

if __name__ == "__main__":
  config = util.initialize_from_env()
  config['eval_path'] = config['test_path']

  model = pm.PluralModel(config)
  with tf.Session() as session:
    model.restore(session)
    model.evaluate(session)
