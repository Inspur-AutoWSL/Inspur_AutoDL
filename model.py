"""Combine all winner solutions in previous challenges (AutoCV, AutoCV2,
AutoNLP and AutoSpeech).
"""

import logging
import numpy as np
import os
import sys
import tensorflow as tf
import time
os.system("pip install hyperopt")
here = os.path.dirname(os.path.abspath(__file__))
model_dirs = ['',                       # current directory
              'AutoSpeech_Combine/PASA_NJU',
              'AutoNLP/upwind_flys',    # AutoNLP 2nd place winner
              'tabular',
              'AutoCV/image',
              'AutoCV2/video'
             ]         # simple NN model
for model_dir in model_dirs:
  sys.path.append(os.path.join(here, model_dir))

from AutoNLP.upwind_flys.model import Model as AutoNLPModel
from AutoSpeech_Combine.PASA_NJU.model import Model as AutoSpeechModel
from tabular.model import Model as TabularModel
from tabular.model_task import Model as TabulartaskModel
from AutoCV.image.model import Model as AutoCVimageModel
from AutoCV2.video.model import Model as AutoCVvideoModel
DOMAIN_TO_MODEL = {
                   'text': AutoNLPModel,
                    'speech': AutoSpeechModel,
                    'tabular': TabularModel,
                    'tabular_task': TabulartaskModel,
                    'image': AutoCVimageModel,
                    'video': AutoCVvideoModel}

class Model():
  """A model that combine all winner solutions. Using domain inferring and
  apply winner solution in the corresponding domain."""

  def __init__(self, metadata):
    """
    Args:
      metadata: an AutoDLMetadata object. Its definition can be found in
          AutoDL_ingestion_program/dataset.py
    """
    self.done_training = False
    self.metadata = metadata
    self.domain = infer_domain(metadata)
    logger.info("The inferred domain of current dataset is: {}."\
                .format(self.domain))
    self.domain_metadata = get_domain_metadata(metadata, self.domain)
    DomainModel = DOMAIN_TO_MODEL[self.domain]
    
    print(DomainModel)
    
    self.domain_model = DomainModel(self.domain_metadata)
    
    self.loop_num = 0
    self.num = 20000000
    self.X = []
    self.Y = []
    self.sess = None
    self.start = True
    self.multi_label = False
    
  def train(self, dataset, remaining_time_budget=None):
    """Train method of domain-specific model."""
    # Convert training dataset to necessary format and
    # store as self.domain_dataset_train
    self.set_domain_dataset(dataset, is_training=True)

    # Train the model
    self.domain_model.train(self.domain_dataset_train, remaining_time_budget=remaining_time_budget)

    # Update self.done_training
    self.done_training = self.domain_model.done_training

  def test(self, dataset, remaining_time_budget=None):
    """Test method of domain-specific model."""
    # Convert test dataset to necessary format and
    # store as self.domain_dataset_test
    self.set_domain_dataset(dataset, is_training=False)

    # As the original metadata doesn't contain number of test examples, we
    # need to add this information
    if self.domain in ['text', 'speech'] and\
       (not self.domain_metadata['test_num'] >= 0):
      self.domain_metadata['test_num'] = len(self.X_test)

    # Make predictions
    Y_pred = self.domain_model.test(self.domain_dataset_test,
                                    remaining_time_budget=remaining_time_budget)

    # Update self.done_training
    self.done_training = self.domain_model.done_training

    return Y_pred

  ##############################################################################
  #### Above 3 methods (__init__, train, test) should always be implemented ####
  ##############################################################################

  def to_numpy_text(self, dataset, is_training):
    """Given the TF dataset received by `train` or `test` method, compute two
    lists of NumPy arrays: `X_train`, `Y_train` for `train` and `X_test`,
    `Y_test` for `test`. Although `Y_test` will always be an
    all-zero matrix, since the test labels are not revealed in `dataset`.
    The computed two lists will by memorized as object attribute:
      self.X_train
      self.Y_train
    or
      self.X_test
      self.Y_test
    according to `is_training`.
    WARNING: since this method will load all data in memory, it's possible to
      cause Out Of Memory (OOM) error, especially for large datasets (e.g.
      video/image datasets).
    Args:
      dataset: a `tf.data.Dataset` object, received by the method `self.train`
        or `self.test`.
      is_training: boolean, indicates whether it concerns the training set.
    Returns:
      two lists of NumPy arrays, for features and labels respectively. If the
        examples all have the same shape, they can be further converted to
        NumPy arrays by:
          X = np.array(X)
          Y = np.array(Y)
        And in this case, `X` will be of shape
          [num_examples, sequence_size, row_count, col_count, num_channels]
        and `Y` will be of shape
          [num_examples, num_classes]
    """
    if is_training:
      subset = 'train'
      if(self.loop_num<1):
        attr_X = 'X_{}'.format(subset)
        attr_Y = 'Y_{}'.format(subset)
        # Only iterate the TF dataset when it's not done yet
        
        if self.start:
          self.total_num = self.domain_metadata['train_num']
          
          dataset = dataset.shuffle(self.total_num)
          # dataset = dataset.padded_batch(4, padded_shapes=([-1, 1, 1, 1], [-1]))
          # dataset = dataset.prefetch(50)
          
          iterator = dataset.make_one_shot_iterator()
          self.next_element = iterator.get_next()
          self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        i = 0
        
        label_set = set()
        
        
        while True:
          try:
            # self.sess.run(self.train_op)
            example, labels = self.sess.run(self.next_element)
            self.X.append(example)
            self.Y.append(labels)
            
              
            #print("multi---task-----------------------------")
            # print(np.array(X).shape)
            for i in range(len(self.Y)):
                if sum(self.Y[i]) > 1:
                    self.multi_label = True
            
            
            if(self.multi_label==True):
                while True:
                    try:
                      print("multi_----------------------------------------------")
                      example, labels = self.sess.run(self.next_element)
                      self.X.append(example)
                      self.Y.append(labels)
                      # print(len(self.X))
                    except tf.errors.OutOfRangeError:
                      break 
                break      
            
            #print("over-------------------")
            
            
            # print(self.Y)
            # print(len(self.Y))
            label_true = np.argmax(labels)
            label_set.add(label_true)

            # self.total_num = self.domain_metadata['train_num']
            if(len(label_set)==self.domain_metadata['class_num']) and len(self.Y) > 1200:
                break
              
            # print(len(label_set))
            
            
            # if i==int(self.domain_metadata['train_num']/3):
                # break
            # print("-----------------")
            # print(i)
            i = i + 1
          except tf.errors.OutOfRangeError:
            break
        setattr(self, attr_X, self.X)
        setattr(self, attr_Y, self.Y)
        X = getattr(self, attr_X)
        Y = getattr(self, attr_Y)
        self.loop_num = self.loop_num + 1
        self.start = False
      elif(self.loop_num < 2):
        attr_X = 'X_{}'.format(subset)
        attr_Y = 'Y_{}'.format(subset)
        # Only iterate the TF dataset when it's not done yet

        while True:
          try:
            # self.sess.run(self.train_op)
            example, labels = self.sess.run(self.next_element)
            self.X.append(example)
            self.Y.append(labels)
            if len(self.Y) > self.total_num * 0.3:
              break
          except tf.errors.OutOfRangeError:
            break
        setattr(self, attr_X, self.X)
        setattr(self, attr_Y, self.Y)
        X = getattr(self, attr_X)
        Y = getattr(self, attr_Y)
        print(len(self.X))
        self.loop_num = self.loop_num + 1
      else:
        attr_X = 'X_{}'.format(subset)
        attr_Y = 'Y_{}'.format(subset)
        # Only iterate the TF dataset when it's not done yet
        while True:
          try:
            # self.sess.run(self.train_op)
            example, labels = self.sess.run(self.next_element)
            self.X.append(example)
            self.Y.append(labels)
            # print(len(self.X))
          except tf.errors.OutOfRangeError:
            break
        setattr(self, attr_X, self.X)
        setattr(self, attr_Y, self.Y)
        X = getattr(self, attr_X)
        Y = getattr(self, attr_Y)
        self.loop_num = self.loop_num + 1
        
        print("data over------------------------------")
        print(len(self.X))
  
    else:
      subset = 'test'
      
      attr_X = 'X_{}'.format(subset)
      attr_Y = 'Y_{}'.format(subset)
  
      # Only iterate the TF dataset when it's not done yet
      if not (hasattr(self, attr_X) and hasattr(self, attr_Y)):
        # dataset = dataset.padded_batch(4, padded_shapes=([-1, 1, 1, 1], [-1]))
        # dataset = dataset.prefetch(50)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        X = []
        Y = []
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
          while True:
            try:
              example, labels = sess.run(next_element)
              X.append(example)
              Y.append(labels)
            except tf.errors.OutOfRangeError:
              break
        setattr(self, attr_X, X)
        setattr(self, attr_Y, Y)
      X = getattr(self, attr_X)
      Y = getattr(self, attr_Y)
    
    
    
    return X, Y

  def to_numpy_speech(self, dataset, is_training):
    """Given the TF dataset received by `train` or `test` method, compute two
    lists of NumPy arrays: `X_train`, `Y_train` for `train` and `X_test`,
    `Y_test` for `test`. Although `Y_test` will always be an
    all-zero matrix, since the test labels are not revealed in `dataset`.
    The computed two lists will by memorized as object attribute:
      self.X_train
      self.Y_train
    or
      self.X_test
      self.Y_test
    according to `is_training`.
    WARNING: since this method will load all data in memory, it's possible to
      cause Out Of Memory (OOM) error, especially for large datasets (e.g.
      video/image datasets).
    Args:
      dataset: a `tf.data.Dataset` object, received by the method `self.train`
        or `self.test`.
      is_training: boolean, indicates whether it concerns the training set.
    Returns:
      two lists of NumPy arrays, for features and labels respectively. If the
        examples all have the same shape, they can be further converted to
        NumPy arrays by:
          X = np.array(X)
          Y = np.array(Y)
        And in this case, `X` will be of shape
          [num_examples, sequence_size, row_count, col_count, num_channels]
        and `Y` will be of shape
          [num_examples, num_classes]
    """
    
    if is_training:
      subset = 'train'
    else:
      subset = 'test'
    attr_X = 'X_{}'.format(subset)
    attr_Y = 'Y_{}'.format(subset)

    # Only iterate the TF dataset when it's not done yet
    if not (hasattr(self, attr_X) and hasattr(self, attr_Y)):
      iterator = dataset.make_one_shot_iterator()
      next_element = iterator.get_next()
      X = []
      Y = []
      with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        while True:
          try:
            example, labels = sess.run(next_element)
            X.append(example)
            Y.append(labels)
          except tf.errors.OutOfRangeError:
            break
      setattr(self, attr_X, X)
      setattr(self, attr_Y, Y)
    X = getattr(self, attr_X)
    Y = getattr(self, attr_Y)
    return X, Y


  def to_numpy(self, dataset, is_training):
    """Given the TF dataset received by `train` or `test` method, compute two
    lists of NumPy arrays: `X_train`, `Y_train` for `train` and `X_test`,
    `Y_test` for `test`. Although `Y_test` will always be an
    all-zero matrix, since the test labels are not revealed in `dataset`.
    The computed two lists will by memorized as object attribute:
      self.X_train
      self.Y_train
    or
      self.X_test
      self.Y_test
    according to `is_training`.
    WARNING: since this method will load all data in memory, it's possible to
      cause Out Of Memory (OOM) error, especially for large datasets (e.g.
      video/image datasets).
    Args:
      dataset: a `tf.data.Dataset` object, received by the method `self.train`
        or `self.test`.
      is_training: boolean, indicates whether it concerns the training set.
    Returns:
      two lists of NumPy arrays, for features and labels respectively. If the
        examples all have the same shape, they can be further converted to
        NumPy arrays by:
          X = np.array(X)
          Y = np.array(Y)
        And in this case, `X` will be of shape
          [num_examples, sequence_size, row_count, col_count, num_channels]
        and `Y` will be of shape
          [num_examples, num_classes]
    """
    
    
    if is_training:
      subset = 'train'
      
      if(self.loop_num<1):
        attr_X = 'X_{}'.format(subset)
        attr_Y = 'Y_{}'.format(subset)
    
      
        print("-----------train---process----time------------------------------------------------------")
        
        start = time.time()
        print(time.time()-start)
        
        if self.start:
          sample_count = self.domain_metadata.size()
          dataset = dataset.batch(int(sample_count))
          iterator = dataset.make_one_shot_iterator()
          self.next_element = iterator.get_next()
        
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        
        while True:
          try:
            example, labels = self.sess.run(self.next_element)
            self.X.append(example)
            self.Y.append(labels)
            break
          except tf.errors.OutOfRangeError:
            break
        setattr(self, attr_X, self.X)
        setattr(self, attr_Y, self.Y)
        X = getattr(self, attr_X)
        Y = getattr(self, attr_Y)
        
        
        self.loop_num = self.loop_num + 1
        self.start = False
        
        print(np.array(X).shape)
        for i in range(len(Y[0])):
         
          if sum(Y[0][i]) > 1:
            print("multi----------------------------------------------------")
            DomainModel = DOMAIN_TO_MODEL['tabular_task']
            self.domain_model = DomainModel(self.domain_metadata)
            break
          else:
            continue
        
        
      else:
        attr_X = 'X_{}'.format(subset)
        attr_Y = 'Y_{}'.format(subset)
        # Only iterate the TF dataset when it's not done yet
        while True:
          try:
            example, labels = self.sess.run(self.next_element)
            self.X.append(example)
            self.Y.append(labels)
            # print(len(self.X))
          except tf.errors.OutOfRangeError:
            break
        setattr(self, attr_X, self.X)
        setattr(self, attr_Y, self.Y)
        X = getattr(self, attr_X)
        Y = getattr(self, attr_Y)
        self.loop_num = self.loop_num + 1
    else:
      subset = 'test'
      
      attr_X = 'X_{}'.format(subset)
      attr_Y = 'Y_{}'.format(subset)
    
      # Only iterate the TF dataset when it's not done yet
      if not (hasattr(self, attr_X) and hasattr(self, attr_Y)):
        
        print("------------------------------test----process---time------------------------------------")
        
        start = time.time()
        

        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        X = []
        Y = []
        
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
          while True:
            try:
              example, labels = sess.run(next_element)
              X.append(example)
              Y.append(labels)
            except tf.errors.OutOfRangeError:
              break
        
        
        print("---")
        print(time.time()-start)
        
    
    
        
        print(np.array(X).shape)
        
        setattr(self, attr_X, X)
        setattr(self, attr_Y, Y)
        
        print(time.time()-start)
        
    X = getattr(self, attr_X)
    Y = getattr(self, attr_Y)
    
    return X, Y

  def set_domain_dataset(self, dataset, is_training=True):
    """Recover the dataset in corresponding competition format (esp. AutoNLP
    and AutoSpeech) and set corresponding attributes:
      self.domain_dataset_train
      self.domain_dataset_test
    according to `is_training`.
    """
    if is_training:
      subset = 'train'
    
      attr_dataset = 'domain_dataset_{}'.format(subset)
      if self.domain == 'text':
        cn = 3
      elif self.domain == 'speech':
        cn = 1
      else:
        cn = 1
      if self.loop_num < cn:
        logger.info("Begin recovering dataset format in the original " +
                    "competition for the subset: {}...".format(subset))
        if self.domain == 'text':
         # Get X, Y as lists of NumPy array
          X, Y = self.to_numpy_text(dataset, is_training=is_training)

          # Retrieve vocabulary (token to index map) from metadata and construct
          # the inverse map
          vocabulary = self.metadata.get_channel_to_index_map()
          index_to_token = [None] * len(vocabulary)
          for token in vocabulary:
            index = vocabulary[token]
            index_to_token[index] = token
  
          # Get separator depending on whether the dataset is in Chinese
          if is_chinese(self.metadata):
            sep = ''
          else:
            sep = ' '
  
          # Construct the corpus
          corpus = []
          for x in X: # each x in X is a list of indices (but as float)
            tokens = [index_to_token[int(i)] for i in x]
            document = sep.join(tokens)
            corpus.append(document)

          # lab = []
          # for x in X:  # each x in X is a list of indices (but as float)
          #   # print(x)
          #   for xj in x:
          #     tokens = [index_to_token[int(i)] for i in xj]
          #     document = sep.join(tokens)
          #     corpus.append(document)
          # for y in Y:
          #   for i in range(len(y)):
          #     lab.extend(np.split(y, len(y))[i])
          # Construct the dataset for training or test
          if is_training:
            labels = np.array(Y)
            domain_dataset = corpus, labels
          else:
            domain_dataset = corpus
  
          # Set the attribute
          setattr(self, attr_dataset, domain_dataset)
  
        elif self.domain == 'speech':
          
          X, Y = self.to_numpy_speech(dataset, is_training=is_training)
          # Get X, Y as lists of NumPy array
          X = [np.squeeze(x) for x in X]

          # Construct the dataset for training or test
          if is_training:
            labels = np.array(Y)
            domain_dataset = X, labels
          else:
            domain_dataset = X
  
          # Set the attribute
          setattr(self, attr_dataset, domain_dataset)

        elif self.domain == 'tabular':
          print("-------------------------------------------------------tabular")
          start_time = time.time()
          X, Y = self.to_numpy(dataset, is_training=is_training)
          
          print("------------")
          print(time.time()-start_time)
          
          # X = [np.squeeze(x) for x in X]
          
          if is_training:
            
            X = [np.squeeze(x) for x in X]
            X = X[0]
            Y = [np.squeeze(x) for x in Y]
            # Y = np.concatenate(Y,axis=0)
            Y = Y[0]
            print("-------------kkkkkkkkkkkkkk")
            print(len(X))
            
            
            labels = np.array(Y)
            domain_dataset = X, labels
          else:
          
            X = [np.squeeze(x) for x in X]
            
            
            X = np.concatenate(X,axis=0)
            domain_dataset = X
            
          setattr(self, attr_dataset, domain_dataset)

        elif self.domain in ['image', 'video']:
          setattr(self, attr_dataset, dataset)
        else:
          raise ValueError("The domain {} doesn't exist.".format(self.domain))
    else:
      subset = 'test'
      
      attr_dataset = 'domain_dataset_{}'.format(subset)
  
      if not hasattr(self, attr_dataset):
        logger.info("Begin recovering dataset format in the original " +
                    "competition for the subset: {}...".format(subset))
        if self.domain == 'text':
          # Get X, Y as lists of NumPy array
          X, Y = self.to_numpy_text(dataset, is_training=is_training)
  
          # Retrieve vocabulary (token to index map) from metadata and construct
          # the inverse map
          vocabulary = self.metadata.get_channel_to_index_map()
          index_to_token = [None] * len(vocabulary)
          for token in vocabulary:
            index = vocabulary[token]
            index_to_token[index] = token
  
          # Get separator depending on whether the dataset is in Chinese
          if is_chinese(self.metadata):
            sep = ''
          else:
            sep = ' '
  
          # Construct the corpus
          corpus = []
          for x in X: # each x in X is a list of indices (but as float)
            tokens = [index_to_token[int(i)] for i in x]
            document = sep.join(tokens)
            corpus.append(document)

          # lab = []
          # for x in X:  # each x in X is a list of indices (but as float)
          #   # print(x)
          #   for xj in x:
          #     tokens = [index_to_token[int(i)] for i in xj]
          #     document = sep.join(tokens)
          #     corpus.append(document)
          # for y in Y:
          #   for i in range(len(y)):
          #     lab.extend(np.split(y, len(y))[i])
          # Construct the dataset for training or test
          if is_training:
            labels = np.array(Y)
            domain_dataset = corpus, labels
          else:
            domain_dataset = corpus
  
          # Set the attribute
          setattr(self, attr_dataset, domain_dataset)
  
        elif self.domain == 'speech':
          X, Y = self.to_numpy_speech(dataset, is_training=is_training)
          # Get X, Y as lists of NumPy array
          X = [np.squeeze(x) for x in X]

          # Construct the dataset for training or test
          if is_training:
            labels = np.array(Y)
            domain_dataset = X, labels
          else:
            domain_dataset = X
  
          # Set the attribute
          setattr(self, attr_dataset, domain_dataset)

        elif self.domain == 'tabular':
          X, Y = self.to_numpy(dataset, is_training=is_training)
          # Get X, Y as lists of NumPy array
          X = [np.squeeze(x) for x in X]
          
          # Construct the dataset for training or test
          if is_training:
            labels = np.array(Y)
            domain_dataset = X, labels
          else:
            domain_dataset = np.array(X)
            
          setattr(self, attr_dataset, domain_dataset)

        elif self.domain in ['image', 'video']:
          setattr(self, attr_dataset, dataset)
        else:
          raise ValueError("The domain {} doesn't exist.".format(self.domain))

def infer_domain(metadata):
  """Infer the domain from the shape of the 4-D tensor.

  Args:
    metadata: an AutoDLMetadata object.
  """
  row_count, col_count = metadata.get_matrix_size(0)
  sequence_size = metadata.get_sequence_size()
  channel_to_index_map = metadata.get_channel_to_index_map()
  domain = None
  if sequence_size == 1:
    if row_count == 1 or col_count == 1:
      domain = "tabular"
    else:
      domain = "image"
  else:
    if row_count == 1 and col_count == 1:
      if len(channel_to_index_map) > 0:
        domain = "text"
      else:
        domain = "speech"
    else:
      domain = "video"
  return domain


def is_chinese(metadata):
  """Judge if the dataset is a Chinese NLP dataset. The current criterion is if
  each word in the vocabulary contains one single character, because when the
  documents are in Chinese, we tokenize each character when formatting the
  dataset.

  Args:
    metadata: an AutoDLMetadata object.
  """
  domain = infer_domain(metadata)
  if domain != 'text':
    return False
  for i, token in enumerate(metadata.get_channel_to_index_map()):
    if len(token) != 1:
      return False
    if i >= 100:
      break
  return True


def get_domain_metadata(metadata, domain, is_training=True):
  """Recover the metadata in corresponding competitions, esp. AutoNLP
  and AutoSpeech.

  Args:
    metadata: an AutoDLMetadata object.
    domain: str, can be one of 'image', 'video', 'text', 'speech' or 'tabular'.
  """
  if domain == 'text':
    # Fetch metadata info from `metadata`
    class_num = metadata.get_output_size()
    num_examples = metadata.size()
    language = 'ZH' if is_chinese(metadata) else 'EN'
    time_budget = 1200 # WARNING: Hard-coded

    # Create domain metadata
    domain_metadata = {}
    domain_metadata['class_num'] = class_num
    if is_training:
      domain_metadata['train_num'] = num_examples
      domain_metadata['test_num'] = -1
    else:
      domain_metadata['train_num'] = -1
      domain_metadata['test_num'] = num_examples
    domain_metadata['language'] = language
    domain_metadata['time_budget'] = time_budget

    return domain_metadata
  elif domain == 'speech':
    # Fetch metadata info from `metadata`
    class_num = metadata.get_output_size()
    num_examples = metadata.size()

    # WARNING: hard-coded properties
    file_format = 'wav'
    sample_rate = 16000

    # Create domain metadata
    domain_metadata = {}
    domain_metadata['class_num'] = class_num
    if is_training:
      domain_metadata['train_num'] = num_examples
      domain_metadata['test_num'] = -1
    else:
      domain_metadata['train_num'] = -1
      domain_metadata['test_num'] = num_examples
    domain_metadata['file_format'] = file_format
    domain_metadata['sample_rate'] = sample_rate

    return domain_metadata
  else:
    return metadata


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
