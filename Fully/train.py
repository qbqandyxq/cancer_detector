import os
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
# import input_data
import input_process_data
import c3d_model
import math
import numpy as np

# Basic model parameters as external flags.
flags = tf.app.flags
#gpu_num = 1
flags.DEFINE_integer('gpu_num',
		     1,
		     'the number of gpu')
flags.DEFINE_float('learning_rate', 
                   1e-5, 
                   'Initial learning rate.')
flags.DEFINE_float('learning_rate_fine', 
                    1e-4, 
                   'finetune learning rate')
flags.DEFINE_integer('max_steps', 
                    5000, 
                    'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 
                     16, #16
                    'Batch size.')
flags.DEFINE_string('train_list', 
                    './list/train_label.csv',
                    'the train list')
flags.DEFINE_string('test_list', 
                    './list/test1.list',
                    'the test list while training')
flags.DEFINE_boolean('use_pretrained_model',
                    False,
                    'set true to load pre train model.')
#MOVING_AVERAGE_DECAY = 0.9995
flags.DEFINE_float('MOVING_AVERAGE_DECAY',
		     0.9995,
		     'weight decay')
flags.DEFINE_string('model_save_dir',
		    './output/',
		      'The model will be saved after the train process.')
#model_save_dir = './output_step1_lr5_2'
#dataset_dir = '/home/lab/Desktop/mwx/C3D-tensorflow-master/UCF101'
flags.DEFINE_string('dataset_dir',
		    '/home/lab/Desktop/wsm/cancer_detector',
		    'the path of the dataset')
FLAGS = flags.FLAGS

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

def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def tower_loss(name_scope, logit, labels):
  cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logit))
  #cross_entropy_mean=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=tf.one_hot(logit,2)))
  tf.summary.scalar(
                  name_scope + '_cross_entropy',
                  cross_entropy_mean
                  )
  weight_decay_loss = tf.get_collection('weightdecay_losses')
  tf.summary.scalar(name_scope + '_weight_decay_loss', tf.reduce_mean(weight_decay_loss) )

  # Calculate the total loss for the current tower.
  total_loss = cross_entropy_mean + weight_decay_loss 
  tf.summary.scalar(name_scope + '_total_loss', tf.reduce_mean(total_loss) )
  return total_loss

def tower_acc(logit, labels):
  correct_pred = tf.equal(tf.argmax(logit, 1), labels)
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  return accuracy

def f1score(labels, logit): #calculate F1score
  predicted = tf.argmax(logit, 1)
  TP = tf.count_nonzero(predicted * labels)
  TN = tf.count_nonzero((predicted - 1) * (labels - 1))
  FP = tf.count_nonzero(predicted * (labels - 1))
  FN = tf.count_nonzero((predicted - 1) * labels)
  
  tf_precision = TP/( TP + FP )
  tf_recall = TP/( TP + FN )
  tf_f1_score = 2 * tf_precision * tf_recall / (tf_precision + tf_recall)
  
  TP_ = tf.count_nonzero((predicted-1) * (labels-1))
  TN_ = tf.count_nonzero((predicted) * (labels))
  FP_ = tf.count_nonzero((predicted-1) * (labels))
  FN_ = tf.count_nonzero((predicted) * (labels-1))
  
  tf_precision_ = TP_/( TP_ + FP_ )
  tf_recall_ = TP_/( TP_ + FN_ )
  tf_f1_score_ = 2 * tf_precision_ * tf_recall_ / (tf_precision_ + tf_recall_)
  
  tf_f1_score = (tf_f1_score + tf_f1_score_)/2
  return tf_precision, tf_recall, tf_f1_score

def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, wd):
  var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var)*wd
    tf.add_to_collection('weightdecay_losses', weight_decay)
  return var

def run_training():
  # Get the sets of images and labels for training, validation, and
  # Tell TensorFlow that the model will be built into the default Graph.

  # Create model directory
  if not os.path.exists(FLAGS.model_save_dir):
      os.makedirs(FLAGS.model_save_dir)
  use_pretrained_model = FLAGS.use_pretrained_model 
  model_filename = "./sports1m_finetuning_ucf101.model"

  with tf.Graph().as_default():
    global_step = tf.get_variable(
                    'global_step',
                    [],
                    initializer=tf.constant_initializer(0),
                    trainable=False
                    )
    images_placeholder, labels_placeholder = placeholder_inputs(
                    FLAGS.batch_size * FLAGS.gpu_num
                    )
    tower_grads1 = []
    tower_grads2 = []
    logits = []
    opt_stable = tf.train.AdamOptimizer(FLAGS.learning_rate)
    opt_finetuning = tf.train.AdamOptimizer(FLAGS.learning_rate_fine)
    with tf.variable_scope('var_name') as var_scope:
      weights = {
              'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 1, 64], 0.0005),
              'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
              'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
              'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005),
              'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
              'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
              'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
              'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
              #'wc6a': _variable_with_weight_decay('wc6a', [3, 3, 3, 512, 256], 0.0005),
              #'wc6b': _variable_with_weight_decay('wc6b', [3, 3, 3, 256, 128], 0.0005),
              #'wc7a': _variable_with_weight_decay('wc7a', [3, 3, 3, 128, 64], 0.0005),
              #'wc7b': _variable_with_weight_decay('wc7b', [3, 3, 3, 64, 32], 0.0005),
              #'wc8a': _variable_with_weight_decay('wc8a', [3, 3, 3, 32, 16], 0.0005),
              #'wc8b': _variable_with_weight_decay('wc8b', [3, 3, 3, 16, 8], 0.0005),
              #'wd1': _variable_with_weight_decay('wd1', [128, 64], 0.0005),
              #'out': _variable_with_weight_decay('wout', [128, c3d_model.NUM_CLASSES], 0.0005)
              'wd1': _variable_with_weight_decay('wd1', [32768, 4096], 0.0005),
              'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.0005),
              'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.0005)
              }
      biases = {
              'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
              'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
              'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
              'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000),
              'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
              'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000),
              'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
              'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
              #'bc6a': _variable_with_weight_decay('bc6a', [256], 0.000),
              #'bc6b': _variable_with_weight_decay('bc6b', [128], 0.000),
              #'bc7a': _variable_with_weight_decay('bc7a', [64], 0.000),
              #'bc7b': _variable_with_weight_decay('bc7b', [32], 0.000),
              #'bc8a': _variable_with_weight_decay('bc8a', [16], 0.000),
              #'bc8b': _variable_with_weight_decay('bc8b', [8], 0.000),
              'bd1': _variable_with_weight_decay('bd1', [4096], 0.000),
              'bd2': _variable_with_weight_decay('bd2', [4096], 0.000),
              'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.000),
              }
    for gpu_index in range(0, FLAGS.gpu_num):
      with tf.device('/gpu:%d' % gpu_index):
        
        varlist2 = [ weights['out'],biases['out'] ]
        varlist1 = list( set(list(weights.values()) + list(biases.values())) - set(varlist2) )
        logit = c3d_model.inference_c3d(
                        images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:,:,:,:],
                        0.5,
                        FLAGS.batch_size,
                        weights,
                        biases
                        )
        loss_name_scope = ('gpud_%d_loss' % gpu_index)
        loss = tower_loss(
                        loss_name_scope,
                        logit,
                        labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size])

        grads1 = opt_stable.compute_gradients(loss, varlist1)
        grads2 = opt_finetuning.compute_gradients(loss, varlist2)
        

        tower_grads1.append(grads1)
        tower_grads2.append(grads2)
        
        logits.append(logit)
    
    #grad = opt_stable.minimize(loss)
    logits = tf.concat(logits,0)
    accuracy = tower_acc(logits, labels_placeholder)
    tf_precision, tf_recall, f1_score = f1score(labels_placeholder, logits)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('f1_score', f1_score)
    tf.summary.scalar('tf_precision', tf_precision)
    tf.summary.scalar('tf_recall', tf_recall)
    grads1 = average_gradients(tower_grads1)
    grads2 = average_gradients(tower_grads2)
    apply_gradient_op1 = opt_stable.apply_gradients(grads1)
    apply_gradient_op2 = opt_finetuning.apply_gradients(grads2, global_step=global_step)
    
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.MOVING_AVERAGE_DECAY)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    train_op = tf.group(apply_gradient_op1, apply_gradient_op2, variables_averages_op)
    #train_op = tf.group(grad, variables_averages_op)
    null_op = tf.no_op()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(list(weights.values()) + list(biases.values()))
    init = tf.global_variables_initializer()

    # Create a session for running Ops on the Graph.
    sess = tf.Session(
                    config=tf.ConfigProto(allow_soft_placement=True)
                    )
    sess.run(init)
    if os.path.isfile(model_filename) and use_pretrained_model:
      print("==========================load pre train model=================================")
      saver.restore(sess, model_filename)

    # Create summary writter
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(FLAGS.model_save_dir,'logs/train'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(FLAGS.model_save_dir,'logs/test'), sess.graph)
    for step in range(FLAGS.max_steps):
      start_time = time.time()
      train_images, train_labels, _, _, _ = input_process_data.read_clip_and_label(
                      FLAGS.dataset_dir,
                      filename=FLAGS.train_list,
                      batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                      num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
                      crop_size=c3d_model.CROP_SIZE,
                      shuffle=True
                      )
      _, loss_ = sess.run([train_op, loss], feed_dict={
                      images_placeholder: train_images,
                      labels_placeholder: train_labels,
                      })
      duration = time.time() - start_time
      #if (step) % 10 ==0:
      print(('Step %d: %.3f sec' % (step, duration)), "loss is:", np.mean(loss_))

      # Save a checkpoint and evaluate the model periodically.
      if (step) % 10 == 0 or (step + 1) == FLAGS.max_steps:
        saver.save(sess, os.path.join(FLAGS.model_save_dir, 'c3d_model'), global_step=step)
        print('Training Data Eval:')
        summary, acc, p, r,f1 = sess.run(
                        [merged, accuracy,tf_precision,tf_recall, f1_score],
                        feed_dict={images_placeholder: train_images,
                            labels_placeholder: train_labels
                            })
        print(("accuracy: " + "{:.5f}".format(acc)))
        print(("f1_score: {:.5f}".format(f1)))  
        train_writer.add_summary(summary, step)
        print('Validation Data Eval:')
        val_images, val_labels, _, _, _ = input_process_data.read_clip_and_label(
                        FLAGS.dataset_dir,
                        filename=FLAGS.test_list,
                        batch_size=FLAGS.batch_size * FLAGS.gpu_num,
                        num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
                        crop_size=c3d_model.CROP_SIZE,
                        shuffle=True
                        )
        summary, acc, p,r,f1 = sess.run(
                        [merged, accuracy, tf_precision,tf_recall,f1_score],
                        feed_dict={
                                        images_placeholder: val_images,
                                        labels_placeholder: val_labels
                                        })
        print(("accuracy: " + "{:.5f}".format(acc)))
        print(("f1_score:{:.5f}".format(f1)))
        test_writer.add_summary(summary, step)
  print("done")

def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()
