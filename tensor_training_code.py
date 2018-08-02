from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

import os
import matplotlib.image as mpimage

tf.logging.set_verbosity(tf.logging.INFO)




def eval_confusion_matrix(labels, predictions):
    '''
        This function is used to calculate confusion matrix
        :param labels: the actual labels
        :param predictions: The predicted values by the model
        :return: confusion matrix
    '''
    with tf.variable_scope("eval_confusion_matrix"):
        con_matrix = tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=2)

        con_matrix_sum = tf.Variable(tf.zeros(shape=(2,2), dtype=tf.int32),
                                            trainable=False,
                                            name="confusion_matrix_result",
                                            collections=[tf.GraphKeys.LOCAL_VARIABLES])


        update_op = tf.assign_add(con_matrix_sum, con_matrix)

        return tf.convert_to_tensor(con_matrix_sum), update_op


def cnn_model_fn(features, labels, mode):
  '''
      The actual CNN model code
      :param features: extracted features
      :param labels: the labels
      :param mode:Training or Testing mode
  '''
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 64, 48, 3])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=16,
      kernel_size=[2, 2],
      activation=tf.nn.relu,
      padding="same")

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)


  # Batch Normalization Layer
  bn = tf.layers.batch_normalization(
      pool1,
      axis=-1,
      momentum=0.99,
      epsilon=0.001,
      center=True,
      scale=True,
      beta_initializer=tf.zeros_initializer(),
      gamma_initializer=tf.ones_initializer(),
      moving_mean_initializer=tf.zeros_initializer(),
      moving_variance_initializer=tf.ones_initializer(),
      beta_regularizer=None,
      gamma_regularizer=None,
      training=True,
      trainable=True,
      name=None,
      reuse=None,
      renorm=False,
      renorm_clipping=None,
      renorm_momentum=0.99,
  )

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=bn,
      filters=32,
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 16 * 12 * 32])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Dropout layer
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"]),"conv_matrix": eval_confusion_matrix(labels, predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def load_data(img_dir):
    '''
        This method loads images and converts them to tensors.
        :param img_dir: The directory where images are present
        :return:tensor containing image data
    '''
    n= np.array([mpimage.imread(os.path.join(img_dir, img))[:,:,:].flatten() for img in os.listdir(img_dir) if img.endswith(".jpg")])
    labels=[]
    for img in os.listdir(img_dir):
        name=img.strip()
        name=name.split("-")
        if name[1]=="v.jpg":
            labels.append(0)
        else:
            labels.append(1)

    return np.asarray(n,dtype=np.float32),np.asarray(labels,dtype=np.float32)



def main():
  # Load training and eval data
  print("loading data..... (this may take some time!)")

  data,labels=load_data("D:\\RIT\\631\\Project\\data_project\\resizing_folder\\mixed_train\\") # insert path to training data
  test_data, test_labels = load_data("D:\\RIT\\631\\Project\\data_project\\resizing_folder\\mixed_test\\") # insert path to testing data


  print("loading data completed.....")
  print("Depending upon the underlying hardware and installed packages, multiple warnings might be shown below.....")
  print("--------------------------\n\n")

  # Creating the Estimator
  classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="D:\\RIT\\631\\Project\\data_project\\Tensorflow_metadata_v6") # insert path to checkpoint files folder (type any valid location you like)

  # Setting up the logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Training the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": data},
      y=labels,
      batch_size=50,
      num_epochs=100,
      shuffle=True)
  classifier.train(
      input_fn=train_input_fn,
      hooks=[logging_hook])

  # Evaluating the model and printing the results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_data},
      y=test_labels,
      num_epochs=1,
      shuffle=True)
  eval_results = classifier.evaluate(input_fn=eval_input_fn)
  print("\n\n")
  print("Displaying performance")
  print("----------RESULTS----------------")
  print("--------------------------\n\n")
  print("Models Testing accuracy:\t"+str(round(eval_results['accuracy']*100,2))+"%")
  print("-----")
  print("Confusion Matrix:\n")
  print(eval_results['conv_matrix'])
  print("--------------------------")
  print("-------------END-------------")


main()