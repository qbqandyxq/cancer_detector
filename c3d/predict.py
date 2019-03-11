# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring




import os.path
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import c3d_model
import numpy as np
import pandas as pd
#dataset_dir = '/home/qbq/Documents/data/UCF-101/'

# Basic model parameters as external flags.
flags = tf.app.flags
flags.DEFINE_integer('gpu_num',
		     1,
		     'the number of gpu')
flags.DEFINE_string('dataset_dir',
		     '/home/lab/dataset/cancer_dataset/test_dataset/',
		     'the path of test dataset')
flags.DEFINE_integer('batch_size', 1 , 'Batch size.')
flags.DEFINE_string('checkpoint', './output_train2proc/c3d_ucf_model-4999',
                    "the path to checkpoint saved dir")
flags.DEFINE_string('TEST_LIST_PATH','/home/lab/dataset/cancer_dataset/test_label.csv',
                    "the path to test.list dir")
flags.DEFINE_string('csv_output','./20190310output.csv','the path to output csv file')
FLAGS = flags.FLAGS


test_num = input_data.get_test_num(FLAGS.TEST_LIST_PATH)

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
  labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
  return images_placeholder, labels_placeholder

def _variable_on_cpu(name, shape, initializer):
  #with tf.device('/cpu:%d' % cpu_id):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var) * wd
    tf.add_to_collection('losses', weight_decay)
  return var

def run_test():
  model_name = FLAGS.checkpoint
  test_list_file = FLAGS.TEST_LIST_PATH
  num_test_videos = len(pd.read_csv(test_list_file))
  print(("Number of test videos={}".format(num_test_videos)))

  # Get the sets of images and labels for training, validation, and
  images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size * FLAGS.gpu_num)
  with tf.variable_scope('var_name') as var_scope:
    weights = {
            'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 1, 64], 0.04, 0.00),
            'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
            'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
            'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
            'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
            'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
            
            'wc6a': _variable_with_weight_decay('wc6a', [3, 3, 3, 512, 256], 0.0005, 0.00),
            'wc6b': _variable_with_weight_decay('wc6b', [3, 3, 3, 256, 128], 0.0005, 0.00),
            'wc7a': _variable_with_weight_decay('wc7a', [3, 3, 3, 128, 64], 0.0005, 0.00),
            'wc7b': _variable_with_weight_decay('wc7b', [3, 3, 3, 64, 32], 0.0005, 0.00),
            'wc8a': _variable_with_weight_decay('wc8a', [3, 3, 3, 32, 16], 0.0005, 0.00),
            'wc8b': _variable_with_weight_decay('wc8b', [3, 3, 3, 16, 8], 0.0005, 0.00),
            'wd1': _variable_with_weight_decay('wd1', [128, 64], 0.0005,0.00),
            'out': _variable_with_weight_decay('wout', [128, c3d_model.NUM_CLASSES], 0.0005, 0.00)
            
#'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
            #'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
            #'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.04, 0.005)
            }
    biases = {
            'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
            'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
            'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
            'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
            'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
            'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
            'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
            'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),

            'bc6a': _variable_with_weight_decay('bc6a', [256], 0.04, 0.000),
            'bc6b': _variable_with_weight_decay('bc6b', [128], 0.04, 0.000),
            'bc7a': _variable_with_weight_decay('bc7a', [64], 0.04, 0.000),
            'bc7b': _variable_with_weight_decay('bc7b', [32], 0.04, 0.000),
            'bc8a': _variable_with_weight_decay('bc8a', [16], 0.04, 0.000),
            'bc8b': _variable_with_weight_decay('bc8b', [8], 0.04, 0.000),
            #'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
            #'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
            'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.04, 0.0),
            }
  logits = []
  for gpu_index in range(0, FLAGS.gpu_num):
    with tf.device('/gpu:%d' % gpu_index):
      logit = c3d_model.inference_c3d(images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:,:,:,:], 0.6, FLAGS.batch_size, weights, biases)
      logits.append(logit)
  logits = tf.concat(logits,0)
  norm_score = tf.nn.softmax(logits)
  
  saver = tf.train.Saver()
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  init = tf.global_variables_initializer()
  sess.run(init)
  # Create a saver for writing training checkpoints.
  saver.restore(sess, model_name)
  # And then after everything is built, start the training loop.
  bufsize = 4
  #write_file = open("./output/work.txt", "w+", bufsize)
  next_start_pos = 0
  all_steps = int((num_test_videos - 1) / (FLAGS.batch_size * FLAGS.gpu_num) + 1)
  loss=[]
  accuracy_epoch = 0
  accuracy_out = 0
  start_time = time.time()
  idlist_ = []
  retlist_ = []
  df = pd.read_csv(test_list_file)
  for step in range(all_steps):
    test_images, test_labels, next_start_pos, dirnames_, valid_len = \
            input_data.read_clip_and_label(
                    FLAGS.dataset_dir,
                    test_list_file,
                    FLAGS.batch_size * FLAGS.gpu_num,
                    start_pos = next_start_pos,
                    crop_size = c3d_model.CROP_SIZE,
                    shuffle = False,
                    )
    print("=====", step)
    predict_score = norm_score.eval(
            session = sess,
            feed_dict = {images_placeholder: test_images}
            )
    
    for i in range(0, valid_len):
      true_label = test_labels[i]
      top1_predicted_label = np.argmax(predict_score[i])
      # Write results: true label, class prob for true label, predicted label, class prob for predicted label
      if true_label == top1_predicted_label:
        accuracy_out += 1 
      #loss.append(true_label[0]-top1_predicted_label)
      idlist_.append(dirnames_[0].split('/')[-1])
      retlist_.append(top1_predicted_label)
      #print(true_label, predict_score, top1_predicted_label)
      #write_file.write('{}, {}, {}, {}\n'.format(
      #        true_label[0],
      #        predict_score[i][true_label],
      #        top1_predicted_label,
      #        predict_score[i][top1_predicted_label]))
      #accuracy_epoch += accuracy_out
  #write_file.close()
  end_time = time.time()
  csvfile = FLAGS.csv_output
  df=pd.DataFrame({'id':idlist_,'ret':retlist_})
  df.to_csv(csvfile, index=False, sep=',')
  print("done")
  #print("loss is:", np.mean(loss), "total time for test: ", (end_time - start_time))
  #print(('Test accuracy is %.5f' % (accuracy_out / (test_num))))
  #print("test num", test_num)
def main(_):
  run_test()

if __name__ == '__main__':
  tf.app.run()
