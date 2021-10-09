import os

from tensorflow.python.framework.constant_op import constant
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

#  Initialization of Tensors
x = tf.constant(4)
print(x)

x = tf.constant(4, shape=(1,1), dtype=tf.float32)
print(x)

x = constant([[1,2,3], [4,5,6]])
print(x)

x = tf.ones((3,3))
print(x)

x = tf.zeros((2,3))
print(x)
# Mathematical Operations

# Indexing

# Reshaping

