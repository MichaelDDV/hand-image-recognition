# Here's an outline for data structures we plan to use.
# The core of this project will involve creating a convolutional neural network using TensorFlow and Keras.
# This convolutional neural network is going to handle the image classification, with layers representing (conceptually
# and literally) steps in the process. numpy and pandas are supporting modules here - numpy will allow us to work with
# large scale matrix multiplication efficiently and easily, and pandas will let us store and access those matrices
# with a high degree of ease as well as control.
# tensorflow_hub is something we will potentially use to take an already-trained model and retrain it with our own
# pictures.


import numpy
import pandas
import tensorflow as tf
import keras
import tensorflow_hub

# -----------------------
# Data structure samples:
# -----------------------


# Sample code for loading in a model
with tf.gfile.FastGFile("sample/path/.pb","rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

# Of course we'll need a TensorFlow Session
with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('tensor_name')

# Sample code for creating and compiling and fitting a model using Keras.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(sample_train_images, sample_train_labels, epochs=5)