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
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
# from mlxtend.classifier import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import logging
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('./AutoDL_scoring_program/')
import time
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe, fmin
import score

np.random.seed(42)


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
    
    self.hyperparams = None
    self.first_hyper = 1
    self.sclf = None

    self.first_round_auc = [0]
    self.first_round_classifier = []
    
    self.second_classifier = None
    self.start = True
    self.loop_num = 0
    
    self.biaozhi = False
    
    self.hyper_loop = 0

  def train(self, dataset, remaining_time_budget=None):
    
    if remaining_time_budget < 1050:
            self.done_training = True
    
    
    print("----------------------------------------------------------------------start")
    
    if(self.loop_num==0):
        self.train_begin_times.append(time.time())
        if len(self.train_begin_times) >= 2:
          cycle_length = self.train_begin_times[-1] - self.train_begin_times[-2]
          self.li_cycle_length.append(cycle_length)
          
        train_start = time.time()
        X, y = dataset
        
        X_train, X_val, y_temp_, y_temp = train_test_split(X, y, test_size=0.1, random_state=0)
        
        params = {"objective": "multiclass", "metric": "multi_logloss", "verbosity": -1, "seed": 1, "num_threads": 4, 'num_class':y.shape[1]}
        
        y_val=[list(x).index(max(x)) for x in y_temp]
        y_train=[list(x).index(max(x)) for x in y_temp_]
          
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)
        
        self.classifier = lgb.train({**params}, train_data, 1, valid_data, init_model=self.classifier, early_stopping_rounds=1, verbose_eval=1, keep_training_booster=True)
        
        
        train_end = time.time()
        print(train_end - train_start)
        
        self.loop_num = self.loop_num + 1
        
  
    
    elif(self.start == True):
        self.train_begin_times.append(time.time())
        if len(self.train_begin_times) >= 2:
          cycle_length = self.train_begin_times[-1] - self.train_begin_times[-2]
          self.li_cycle_length.append(cycle_length)
          
        train_start = time.time()
        X, y = dataset
        
        X_train, X_val, y_temp_, y_temp = train_test_split(X, y, test_size=0.1, random_state=0)
        
        params = {"objective": "multiclass", "metric": "multi_logloss", "verbosity": -1, "seed": 1, "num_threads": 4, 'num_class':y.shape[1]}
        
        y_val=[list(x).index(max(x)) for x in y_temp]
        y_train=[list(x).index(max(x)) for x in y_temp_]
          
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)
        
        self.classifier = lgb.train({**params}, train_data, 5, valid_data, init_model=self.classifier, early_stopping_rounds=5, verbose_eval=1, keep_training_booster=True)
        self.first_round_classifier.append(self.classifier)
        
        train_end = time.time()
        print(train_end - train_start)
        
        train_duration = train_end - train_start
        
        
        y_pred1 = self.classifier.predict(X_val)
        auc_score = score.autodl_auc(y_temp,y_pred1)
        
        if self.first_round_auc[-1] < auc_score or len(self.first_round_auc) < 5:
            self.first_round_auc.append(auc_score)
        else:
            self.start = False
        print("auc----------------------------------------------")
        print(auc_score)
        print(self.first_round_auc)
        
        self.loop_num = self.loop_num + 1
        
        # self.START = False
        logger.info("{:.2f} sec used. ".format(train_duration) +\
                "Total time used for training + test: {:.2f} sec. ".format(sum(self.li_cycle_length)))
    else:
  
        self.train_begin_times.append(time.time())
        if len(self.train_begin_times) >= 2:
          cycle_length = self.train_begin_times[-1] - self.train_begin_times[-2]
          self.li_cycle_length.append(cycle_length)
          
        train_start = time.time()
        X, y = dataset
        
        params = {"objective": "multiclass", "metric": "multi_logloss", "verbosity": -1, "seed": 1, "num_threads": 4, 'num_class':y.shape[1]}
          
              
        if self.first_hyper == 1:
            self.hyperparams = self._hyperopt(X, y, params)
            self.first_hyper = self.first_hyper + 1
            
        X_train, X_val, y_temp_, y_temp = train_test_split(X, y, test_size=0.2, random_state=0)
        
        y_val=[list(x).index(max(x)) for x in y_temp]
        y_train=[list(x).index(max(x)) for x in y_temp_]
          
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val)
          
        
        self.second_classifier = lgb.train({**params, **self.hyperparams}, train_data, 300, valid_data, init_model=self.second_classifier, 
                                            early_stopping_rounds=30,verbose_eval=50, keep_training_booster=True)
        print("--------------endtrain")
        train_end = time.time()
        print(train_end - train_start)
        # self.first_round_classifier.append(self.classifier)
        y_pred1 = self.second_classifier.predict(X_val)
        auc_score = score.autodl_auc(y_temp,y_pred1)
        print("auc------------------------------------------------")
        print(auc_score)
        
        if(auc_score > self.first_round_auc[-1]):
            self.classifier = self.second_classifier
            #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!llll")
            # self.first_round_auc.append(auc_score)
            #self.second_classifier = self.classifier
            self.first_round_classifier.append(self.second_classifier)
            
        else:
            print("!!!!!!!&&&&&&&&&&&&&&&&&&&&&&&&")
            self.classifier = self.first_round_classifier[-1]
            
        
        # Update for time budget managing
        train_duration = train_end - train_start
       
        self.hyper_loop = self.hyper_loop + 1
       
        logger.info("{:.2f} sec used. ".format(train_duration) +\
                "Total time used for training + test: {:.2f} sec. ".format(sum(self.li_cycle_length)))
        
        
        
        
        
        
                    
                
        

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
    
    if self.sclf != None:
        y_pred = self.sclf.predict(dataset)
        y_pred = y_pred.reshape(-1,self.output_dim)
    # Start testing (i.e. making prediction on test set)
    else:
        y_pred = self.classifier.predict(dataset)
    
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
    
    # train_data = lgb.Dataset(X, label=y)
    y_train=[list(x).index(max(x)) for x in y]
    
    train_data = lgb.Dataset(X, label=y_train)
     
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
        
        model = lgb.cv({**params, **hyperparams},train_data, 500, nfold=5, early_stopping_rounds=30, verbose_eval=0)
        
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
