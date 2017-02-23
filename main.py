from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

# A placeholder, a representation of a value that we'll supply when we ask tensorflow to run a computation.
# In this case, the placeholder represents a 32 bit float object, of n inputs, and a vector of size 784 (28bit*28bit)
x = tf.placeholder(tf.float32, [None,784])

# Variable: A modifiable tensor.
# Generally, model variable 
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Runs softmax on the matrix multiplication of x with weights with bias.
weightsWithBias = tf.matmul(x,W) + b
y = tf.nn.softmax(weightsWithBias)

# Implement the cost function
# Initialize a placeholder for the correct labels.
y_ = tf.placeholder(tf.float32, [None, 10])

# Calclate cross entropy for cost.
# (Why does this seem to produce more accurate results? About 2-3%?)
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Better to use this cross entropy than calculating from scratch. Tutorial said "numerically unstable". Seemed to be inherent to the algorithm itself.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=weightsWithBias, labels=y_)

# Train the model using gradient descent.
alpha = 0.5
train_step = tf.train.GradientDescentOptimizer(alpha).minimize(cross_entropy)

# Start a Tensorflow session. Tensorflow sessions connects to C/C++ instances to more efficiently perform calculations.
sess = tf.InteractiveSession()

# Initialize the created variables (W,b)
tf.global_variables_initializer().run()

# Actually train the model.
# Use Stochastic gradient descent. Randomly select batches of 100 samples from the training data, and perform gradient descent based on loss function results.
# Doing fully batched gradient descent is taxing on the system, so we use randomly selected batches to yield similar results.

for _ in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Evaluate the performance of the model

# The argmax function allows us to check the index of the highest entry of a tensor along an axis. 
# In this case, the index of the highest value of each axis is simply the classification.
# This line of code displays a list of booleans where the predicition was correct.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# Cast to float32, and then calculate the percentage.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Check the accuracy of the test data.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
