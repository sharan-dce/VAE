import tensorflow as tf
import numpy as np
from display import display
import vae
import matplotlib.pyplot as plt

epochs = 100
batch_size = 2

mnist = tf.keras.datasets.mnist

#remove this [ : 10] to run through the whole dataset after testing
x_train = mnist.load_data()[0][0]

x_train = x_train.reshape([-1, 28, 28, 1])

VAE = vae.vae()

for epoch in range(epochs):
	for i in range((len(x_train) + batch_size - 1) // batch_size):
		L = i * batch_size
		'''
		############################
		batch = x_train[L : L + batch_size]
		n = len(batch)
		sampled_standard_normal_values = (np.random.multivariate_normal(np.zeros([10]), np.identity(10), n))
		#print (np.shape(VAE.sess.run(VAE.reconstruction_loss, feed_dict = {VAE.inp : batch, VAE.training : 0, VAE.sampled_standard_normal_input : sampled_standard_normal_values})))

		print ((VAE.sess.run(VAE.conv1, feed_dict = {VAE.inp : batch, VAE.training : 0, VAE.sampled_standard_normal_input : sampled_standard_normal_values})))
		#############################
		'''
		VAE.train_vae(x_train[L : L + batch_size])

	loss = 0.0
	for i in range((len(x_train) + batch_size - 1) // batch_size):
		L = i * batch_size
		batch = x_train[L : L + batch_size]
		loss += VAE.find_loss(batch)
	np.random.shuffle(x_train)
	print ('Epoch {0:7d}, Loss{1:8.2f}'.format(epoch, loss))
	display(VAE.get_samples(225))
