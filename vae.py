import tensorflow as tf
import numpy as np

class vae():
	def __init__(self):
		self.inp = tf.placeholder(dtype = tf.int64)
		self.inp_normalized = tf.cast(self.inp, dtype = tf.float32) / 255.0
		self.sampled_standard_normal_input = tf.placeholder(dtype = tf.float32)
		self.training = tf.placeholder(dtype = tf.float32)

		self.wconv1 = tf.get_variable(name = 'wconv1', dtype = tf.float32, shape = [5, 5, 1, 16], initializer = tf.contrib.layers.xavier_initializer())
		self.bconv1 = tf.get_variable(name = 'bconv1', dtype = tf.float32, shape = [16], initializer = tf.initializers.constant(1.0))
		self.conv1 = tf.nn.lrn(tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(self.inp_normalized, self.wconv1, strides = [1, 1, 1, 1], padding ='SAME') + self.bconv1), ksize = [1, 3, 3, 1], strides = [1, 1, 1, 1], padding = 'SAME'))

		self.wconv2 = tf.get_variable(name = 'wconv2', dtype = tf.float32, shape = [5, 5, 16, 16], initializer = tf.contrib.layers.xavier_initializer())
		self.bconv2 = tf.get_variable(name = 'bconv2', dtype = tf.float32, shape = [16], initializer = tf.initializers.constant(1.0))
		self.conv2 = tf.nn.lrn(tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(self.conv1, self.wconv2, strides = [1, 1, 1, 1], padding = 'SAME') + self.bconv2), ksize = [1, 3, 3, 1], strides = [1, 1, 1 ,1], padding = 'SAME'))

		self.reshaped_conv2 = tf.reshape(self.conv2, shape = [-1, 28 * 28 * 16])

		#learn the mean
		self.mean_w1 = tf.get_variable(name = 'mean_w1', dtype = tf.float32, shape = [28 * 28 * 16, 500], initializer = tf.contrib.layers.xavier_initializer())
		self.mean_b1 = tf.get_variable(name = 'mean_b1', dtype = tf.float32, shape = [500], initializer = tf.initializers.constant(1.0))
		self.mean1 = tf.nn.relu(tf.matmul(self.reshaped_conv2, self.mean_w1) + self.mean_b1)

		self.mean_w2 = tf.get_variable(name = 'mean_w2', dtype = tf.float32, shape = [500, 10], initializer = tf.contrib.layers.xavier_initializer())
		self.mean_b2 = tf.get_variable(name = 'mean_b2', dtype = tf.float32, shape = [10], initializer = tf.initializers.constant(0.0))
		self.mean = tf.matmul(self.mean1, self.mean_w2) + self.mean_b2
		#done with mean

		#learn the standard deviation
		self.stddev_w1 = tf.get_variable(name = 'stddev_w1', dtype = tf.float32, shape = [28 * 28 * 16, 500], initializer = tf.contrib.layers.xavier_initializer())
		self.stddev_b1 = tf.get_variable(name = 'stddev_b1', dtype = tf.float32, shape = [500], initializer = tf.initializers.constant(1.0))
		self.stddev1 = tf.nn.relu(tf.matmul(self.reshaped_conv2, self.stddev_w1) + self.stddev_b1)

		self.stddev_w2 = tf.get_variable(name = 'stddev_w2', dtype = tf.float32, shape = [500, 10], initializer = tf.contrib.layers.xavier_initializer())
		self.stddev_b2 = tf.get_variable(name = 'stddev_b2', dtype = tf.float32, shape = [10], initializer = tf.initializers.constant(0.0))
		self.stddev = tf.nn.relu(tf.matmul(self.stddev1, self.stddev_w2) + self.stddev_b2)
		#done with the stddev

		self.sampled_latent_variable = self.training * (self.sampled_standard_normal_input * self.stddev + self.mean) + self.sampled_standard_normal_input * (1 - self.training)

		self.decoder_wfc1 = tf.get_variable(name = 'decoder_wfc1', dtype = tf.float32, shape = [10, 500], initializer = tf.contrib.layers.xavier_initializer())
		self.decoder_bfc1 = tf.get_variable(name = 'decoder_bfc1', dtype = tf.float32, shape = [500], initializer = tf.initializers.constant(0.0))
		self.decoder_fcout1 = tf.nn.relu(tf.matmul(self.sampled_latent_variable, self.decoder_wfc1) + self.decoder_bfc1)

		self.decoder_wfc2 = tf.get_variable(name = 'decoder_wfc2', dtype = tf.float32, shape = [500, 28 * 28 * 16], initializer = tf.contrib.layers.xavier_initializer())
		self.decoder_bfc2 = tf.get_variable(name = 'decoder_bfc2', dtype = tf.float32, shape = [28 * 28 * 16], initializer = tf.initializers.constant(1.0))
		self.decoder_fcout2 = tf.nn.relu(tf.matmul(self.decoder_fcout1, self.decoder_wfc2) + self.decoder_bfc2)

		self.decoder_convolutional_layers_input = tf.reshape(self.decoder_fcout2, shape = [-1, 28, 28, 16])

		self.decoder_wconv1 = tf.get_variable(name = 'decoder_wconv1', dtype = tf.float32, shape = [5, 5, 16, 16], initializer = tf.contrib.layers.xavier_initializer())
		self.decoder_bconv1 = tf.get_variable(name = 'decoder_bconv1', dtype = tf.float32, shape = [16], initializer = tf.contrib.layers.xavier_initializer())
		self.decoder_conv1 = tf.nn.relu(tf.nn.conv2d(self.decoder_convolutional_layers_input, self.decoder_wconv1, strides = [1, 1, 1, 1], padding = 'SAME') + self.decoder_bconv1)

		self.decoder_wconv2 = tf.get_variable(name = 'decoder_wconv2', dtype = tf.float32, shape = [5, 5, 16, 1], initializer = tf.contrib.layers.xavier_initializer())
		self.decoder_bconv2 = tf.get_variable(name = 'decoder_bconv2', dtype = tf.float32, shape = [1], initializer = tf.contrib.layers.xavier_initializer())
		self.decoder_conv2 = tf.nn.conv2d(self.decoder_conv1, self.decoder_wconv2, strides = [1, 1, 1, 1], padding = 'SAME') + self.decoder_bconv2

		self.output = tf.nn.sigmoid(self.decoder_conv2)

		self.final_output = tf.cast(self.output * 255.0, dtype = tf.int64)

		#loss
		
		#self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.inp_normalized, logits = self.decoder_conv2)
		#self.reconstruction_loss = tf.reduce_mean(self.cross_entropy)
		self.reconstruction_loss = tf.losses.mean_squared_error(labels = self.inp_normalized, predictions = self.output)
		self.KL_loss = 0.5 * (tf.reduce_mean(tf.reduce_sum(self.stddev) + tf.reduce_sum(self.mean ** 2) - 10 - tf.log(1e-8 + tf.reduce_prod(self.stddev))))
		self.total_loss = self.reconstruction_loss + self.KL_loss
		#trainer
		self.train = tf.train.AdamOptimizer(0.00001).minimize(self.total_loss)
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def train_vae(self, batch):
		n = len(batch)
		sampled_standard_normal_values = (np.random.multivariate_normal(np.zeros([10]), np.identity(10), n))
		self.sess.run(self.train, feed_dict = {self.inp : batch, self.training : 1, self.sampled_standard_normal_input : sampled_standard_normal_values})

	def find_loss(self, batch):
		n = len(batch)
		sampled_standard_normal_values = (np.random.multivariate_normal(np.zeros([10]), np.identity(10), n))
		return self.sess.run(self.total_loss, feed_dict = {self.inp : batch, self.training : 0, self.sampled_standard_normal_input : sampled_standard_normal_values})

	def get_samples(self, n):
		sampled_standard_normal_values = (np.random.multivariate_normal(np.zeros([10]), np.identity(10), n))
		return self.sess.run(self.final_output, feed_dict = {self.training : 0, self.sampled_standard_normal_input : sampled_standard_normal_values, self.inp : np.zeros([n, 28, 28, 1])})
