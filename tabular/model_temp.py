# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by: Zhengying Liu, Isabelle Guyon

"""An example of code submission for the AutoDL challenge.

It implements 3 compulsory methods ('__init__', 'train' and 'test') and
an attribute 'done_training' for indicating if the model will not proceed more
training due to convergence or limited time budget.

To create a valid submission, zip model.py together with other necessary files
such as Python modules/packages, pre-trained weights. The final zip file should
not exceed 300MB.
"""

from sklearn.linear_model import LinearRegression
import logging
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('./AutoDL_scoring_program/')
from tools import log
# from models import *
import time
import lightgbm as lgb
from sklearn.model_selection import train_test_split
# from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe, fmin
import score


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Flatten, Conv1D
from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.advanced_activations import PReLU
from keras import optimizers
from keras.regularizers import l2

from CONSTANT import *
from CONSTANT import MAX_VALID_PERCLASS_SAMPLE

np.random.seed(42)
import random

class Model(object):
  """Fully connected neural network with no hidden layer."""

  def __init__(self, metadata):
    """
    Args:
      metadata: an AutoDLMetadata object. Its definition can be found in
          AutoDL_ingestion_program/dataset.py
    """
    self.START = True
    self.done_training = False

    self.metadata = metadata
    self.output_dim = self.metadata.get_output_size()
    
    # Change to True if you want to show device info at each operation
    log_device_placement = False
    
    # Classifier using model_fn (see below)
    self.classifier = None

    # Attributes for preprocessing
    self.default_image_size = (112,112)
    self.default_num_frames = 10
    self.default_shuffle_buffer = 100

    # Attributes for managing time budget
    # Cumulated number of training steps
    self.birthday = time.time()
    self.train_begin_times = []
    self.test_begin_times = []
    self.li_steps_to_train = []
    self.li_cycle_length = []
    self.li_estimated_time = []
    self.time_estimator = LinearRegression()
    # Critical number for early stopping
    # Depends on number of classes (output_dim)
    # see the function self.choose_to_stop_early() below for more details
    self.num_epochs_we_want_to_train = 70
    self._model = None
    
    self.model = None
    
    
  
  def train(self, dataset, remaining_time_budget=None):
    
    X,y = dataset
    

    BATCH_SIZE = 400
    EPOCHS = 1000
    PATIENCE = 50
    
    print(X.shape)
    
    # X = X.reshape((-1, X.shape[1],1))
    # y = y.reshape((y.shape[0],self.output_dim))
    
    print(X.shape)
    
    print(y)
    
    #from xgboost import XGBClassifier
    #from sklearn.multiclass import OneVsRestClassifier
    #from sklearn.metrics import accuracy_score
    
    #clf_multilabel = OneVsRestClassifier(XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100))
    #fit=clf_multilabel.fit(X, y)
    #print(clf_multilabel.score(X,y))
    
    model = Sequential()
    model.add(Dense(128, input_dim=X.shape[1], init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.9))
    

    
    model.add(Dense(64, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.8))

    model.add(Dense(self.output_dim, init='he_normal', activation='softmax'))
    
    
    
    #model = Sequential()
    #model.add(Conv1D(8, kernel_size=3, strides=1, padding='same', input_shape=(X.shape[1],1)))
    #model.add(BatchNormalization())
    #model.add(Conv1D(8, kernel_size=3, strides=1, padding='same'))
    #model.add(Conv1D(16, kernel_size=3, strides=1, padding='same'))
    #model.add(BatchNormalization())
    #model.add(Conv1D(16, kernel_size=3, strides=1, padding='same'))
    #model.add(Conv1D(32, kernel_size=3, strides=1, padding='same'))
    #model.add(BatchNormalization())
    #model.add(Conv1D(32, kernel_size=3, strides=1, padding='same'))
    #model.add(Conv1D(32, kernel_size=3, strides=1, padding='same'))
    #model.add(Conv1D(64, kernel_size=3, strides=1, padding='same'))
    #model.add(Activation('tanh'))
    #model.add(Flatten())
    #model.add(Dropout(0.5))
    
    #model.add(Dense(512,kernel_initializer='he_normal', activation='relu', W_regularizer=l2(0.01)))
    #model.add(Dense(128,kernel_initializer='he_normal', activation='relu', W_regularizer=l2(0.01)))
    #model.add(Dense(self.output_dim, kernel_initializer='normal', activation='softmax'))

    print(model.summary())
    
    callbacks_list = [
        # keras.callbacks.ModelCheckpoint(
            # filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
            # monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE)
    ]
    
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    

    history = model.fit(X,
                        y,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks=callbacks_list,
                        validation_split=0.3,
                        verbose=1)
    self.model = model 
    
    
  def test(self, dataset, remaining_time_budget=None):
    """Test this algorithm on the tensorflow |dataset|.

    Args:
      Same as that of `train` method, except that the `labels` will be empty.
    Returns:
      predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim).
          here `sample_count` is the number of examples in this dataset as test
          set and `output_dim` is the number of labels to be predicted. The
          values should be binary or in the interval [0,1].
    """

    test_begin = time.time()
    self.test_begin_times.append(test_begin)
    logger.info("Begin testing...")
    
    # dataset = dataset.reshape((-1, dataset.shape[1],1))
    y_pred = self.model.predict(dataset)
    
    # Start testing (i.e. making prediction on test set)
    
    
    test_end = time.time()
    # Update some variables for time management
    test_duration = test_end - test_begin
    logger.info("[+] Successfully made one prediction. {:.2f} sec used. "\
              .format(test_duration) +\
              "Duration used for test: {:2f}".format(test_duration))
    return y_pred

  ##############################################################################
  #### Above 3 methods (__init__, train, test) should always be implemented ####
  ##############################################################################

  # Model functions that contain info on neural network architectures
  # Several model functions are to be implemented, for different domains


 
    
  def _hyperopt(self, X, y, params):
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=1)
    # train_data = lgb.Dataset(X_train, label=y_train)
    # valid_data = lgb.Dataset(X_val, label=y_val)
    
    train_data = lgb.Dataset(X, label=y)
     
    space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.5)),
        "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
        "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
        "feature_fraction": hp.quniform("feature_fraction", 0.8, 1.0, 0.1),
        "reg_alpha": hp.uniform("reg_alpha", 0, 2),
        "reg_lambda": hp.uniform("reg_lambda", 0, 2),
        "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
    }
  
    def objective(hyperparams):
        # model = lgb.cv({**params, **hyperparams}, train_data, 300, valid_data, early_stopping_rounds=30, verbose_eval=0)
        
        model = lgb.cv({**params, **hyperparams},train_data, 300, nfold=5, early_stopping_rounds=30, verbose_eval=0)
        
        # score = model.best_score["valid_0"][params["metric"]]
        
        score = min(model['multi_logloss-mean'])
        
        return {'loss': -score, 'status': STATUS_OK}
  
    trials = Trials()
    best = fmin(fn=objective, space=space, trials=trials,
                algo=tpe.suggest, max_evals=2, verbose=1,
                rstate=np.random.RandomState(1))
  
    hyperparams = space_eval(space, best)
    # log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")
    return hyperparams
    
    

def sigmoid_cross_entropy_with_logits(labels=None, logits=None):
  """Re-implementation of this function:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

  Let z = labels, x = logits, then return the sigmoid cross entropy
    max(x, 0) - x * z + log(1 + exp(-abs(x)))
  (Then sum over all classes.)
  """
  labels = tf.cast(labels, dtype=tf.float32)
  relu_logits = tf.nn.relu(logits)
  exp_logits = tf.exp(- tf.abs(logits))
  sigmoid_logits = tf.log(1 + exp_logits)
  element_wise_xent = relu_logits - labels * logits + sigmoid_logits
  return element_wise_xent

def get_num_entries(tensor):
  """Return number of entries for a TensorFlow tensor.

  Args:
    tensor: a tf.Tensor or tf.SparseTensor object of shape
        (batch_size, sequence_size, row_count, col_count[, num_channels])
  Returns:
    num_entries: number of entries of each example, which is equal to
        sequence_size * row_count * col_count [* num_channels]
  """
  tensor_shape = tensor.shape
  assert(len(tensor_shape) > 1)
  num_entries  = 1
  for i in tensor_shape[1:]:
    num_entries *= int(i)
  return num_entries

def crop_time_axis(tensor_4d, num_frames, begin_index=None):
  """Given a 4-D tensor, take a slice of length `num_frames` on its time axis.

  Args:
    tensor_4d: A Tensor of shape
        [sequence_size, row_count, col_count, num_channels]
    num_frames: An integer representing the resulted chunk (sequence) length
    begin_index: The index of the beginning of the chunk. If `None`, chosen
      randomly.
  Returns:
    A Tensor of sequence length `num_frames`, which is a chunk of `tensor_4d`.
  """
  # pad sequence if not long enough
  pad_size = tf.maximum(num_frames - tf.shape(tensor_4d)[0], 0)
  padded_tensor = tf.pad(tensor_4d, ((0, pad_size), (0, 0), (0, 0), (0, 0)))

  # If not given, randomly choose the beginning index of frames
  if not begin_index:
    maxval = tf.shape(padded_tensor)[0] - num_frames + 1
    begin_index = tf.random.uniform([1],
                                    minval=0,
                                    maxval=maxval,
                                    dtype=tf.int32)
    begin_index = tf.stack([begin_index[0], 0, 0, 0], name='begin_index')

  sliced_tensor = tf.slice(padded_tensor,
                           begin=begin_index,
                           size=[num_frames, -1, -1, -1])

  return sliced_tensor

def resize_space_axes(tensor_4d, new_row_count, new_col_count):
  """Given a 4-D tensor, resize space axes to have target size.

  Args:
    tensor_4d: A Tensor of shape
        [sequence_size, row_count, col_count, num_channels].
    new_row_count: An integer indicating the target row count.
    new_col_count: An integer indicating the target column count.
  Returns:
    A Tensor of shape [sequence_size, target_row_count, target_col_count].
  """
  resized_images = tf.image.resize_images(tensor_4d,
                                          size=(new_row_count, new_col_count))
  return resized_images





def get_logger(verbosity_level):
  """Set logging format to something like:
       2019-04-25 12:52:51,924 INFO model.py: <message>
  """
  logger = logging.getLogger(__file__)
  logging_level = getattr(logging, verbosity_level)
  logger.setLevel(logging_level)
  formatter = logging.Formatter(
    fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
  stdout_handler = logging.StreamHandler(sys.stdout)
  stdout_handler.setLevel(logging_level)
  stdout_handler.setFormatter(formatter)
  stderr_handler = logging.StreamHandler(sys.stderr)
  stderr_handler.setLevel(logging.WARNING)
  stderr_handler.setFormatter(formatter)
  logger.addHandler(stdout_handler)
  logger.addHandler(stderr_handler)
  logger.propagate = False
  return logger

logger = get_logger('INFO')
