from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(tf.cast(features["x"], tf.float32), [-1, 154, 100, 2])
  print(input_layer)

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[1, 5],
      padding="same",
      activation=tf.nn.relu)
  print(conv1)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, 2], strides=[1,2])
  print(pool1)

  # Convolutional Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=8,
      kernel_size=[1, 5],
      padding="same",
      activation=tf.nn.relu)
  print(conv2)

  # Pooling Layer #2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 5], strides=[1,5])
  print(pool2)

  # Convolutional Layer #3
  conv3 = tf.layers.conv2d(
	  inputs=pool2,
	  filters=2,
	  kernel_size=[154, 5],
	  padding="same",
	  activation=tf.nn.relu)
  print(conv3)

  # Pooling Layer #3
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[1, 2], strides=[1, 2])
  print(pool3)

  # Dense Layer
  pool3_flat = tf.reshape(pool3, [-1, 154 * 5 * 2])
  print(pool3_flat)

  dense = tf.layers.dense(inputs=pool3_flat, units=512, activation=tf.nn.relu)
  print(dense)

  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
  print(dropout)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=154)
  print(logits)

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
  multiclass_labels = tf.reshape(tf.cast(labels, tf.int32), [-1, 154])
  loss = tf.losses.sigmoid_cross_entropy(
      multi_class_labels=multiclass_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"]),
      "precision": tf.metrics.precision(labels=labels, predictions=predictions["classes"]),
      "mean_per_class_accuracy": tf.metrics.mean_per_class_accuracy(labels=labels, predictions=predictions["classes"]),
      "auc": tf.metrics.auc(labels=labels, predictions=predictions["probabilities"]),
      }

  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



def main():
  tf.app.run()