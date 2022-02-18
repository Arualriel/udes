import numpy as np
import tensorflow as tf
import constants as cst
from abc import ABC, abstractmethod
import time

#************************************************************
#************************************************************
#************************************************************
class UDE(ABC):
	def __init__(self):
		# General Parameters
		self.learning_rate = cst.LEARNING_RATE
		self.training_steps = cst.TRAINING_STEPS
		self.train_mode = cst.TRAIN_MODE
		
		# Network Parameters & Creation
		self.display_step = cst.DISPLAY_STEP
		self.neurons_input = cst.NEURONS_INPUT
		self.neurons_output = cst.NEURONS_OUTPUT
		self.layers = cst.LAYERS
		self._create_network()

		# Train & Plot Points
		self.train_points = cst.TRAIN_POINTS
		self.plot_points = cst.PLOT_POINTS

		# Domain & Main Function
		self.x_vec = self._get_domain(self.train_points)
		self.NN_vec = self._multilayer_perceptron(self.x_vec)
		self.fx_vec = self._f(self.x_vec,self.NN_vec)
		
		# Infinitesimal Small Number
		self.inf_s = np.sqrt(np.finfo(np.float32).eps)

	#--------------------------------------------------------
	""" This function defines the network architecture. """
	def _create_network(self):
		self.weights = {}
		self.biases = {}
		prev_layer_size = self.neurons_input

		# Create Layers & Biases
		for i in range(len(self.layers)):
			self.weights['h' + str(i)] = tf.Variable(tf.random.normal([prev_layer_size, self.layers[i]]))
			self.biases['b' + str(i)] = tf.Variable(tf.random.normal([self.layers[i]]))
			prev_layer_size = self.layers[i]

		self.weights['out'] = tf.Variable(tf.random.normal([prev_layer_size, self.neurons_output]))
		self.biases['out'] = tf.Variable(tf.random.normal([self.neurons_output]))

		"""
		weights = {
			'h1': tf.Variable(tf.random.normal([1, 32])),
			'h2': tf.Variable(tf.random.normal([32, 32])),
			'out': tf.Variable(tf.random.normal([32, 1]))
		}
		biases = {
			'b1': tf.Variable(tf.random.normal([32])),
			'b2': tf.Variable(tf.random.normal([32])),
			'out': tf.Variable(tf.random.normal([1]))
		}
		"""

		# Select Optimizer
		self.optimizer = self._select_optimizer()

	#--------------------------------------------------------
	""" Selects the optimizer to be used for the network. """
	def _select_optimizer(self):
		if cst.OPTIMIZER == 'SGD':
			return tf.optimizers.SGD(self.learning_rate)
		elif cst.OPTIMIZER == 'ADAM':
			return tf.optimizers.Adam(self.learning_rate)
		else:
			raise ValueError("[ERROR]: The selected OPTIMIZER is not yet considered.")

	#--------------------------------------------------------
	"""
	A simple multilayer perceptron with the same activation function for every layer.
	The output layer uses a SIGMOID activation because the original problem domain is [0,1].
	"""
	def _multilayer_perceptron(self, x):
		prev_layer = np.array([[[x]]], dtype='float32')
		
		for i in range(len(self.layers)):
			l = tf.add(tf.matmul(prev_layer, self.weights['h' + str(i)]), self.biases['b' + str(i)])

			if cst.ACT_HIDDEN == 'SIGMOID':
				l = tf.nn.sigmoid(l)
			elif cst.ACT_HIDDEN == 'RELU':
				l = tf.nn.relu(l)
			elif cst.ACT_HIDDEN == 'TANH':
				l = tf.nn.tanh(l)
			elif cst.ACT_HIDDEN != 'NONE':
				raise ValueError("[ERROR]: The selected ACT_HIDDEN is not yet considered.")
			
			prev_layer = l

		output = tf.matmul(prev_layer, self.weights['out']) + self.biases['out']

		if cst.ACT_OUT == 'SIGMOID':
			output = tf.nn.sigmoid(output)
		elif cst.ACT_OUT == 'RELU':
			output = tf.nn.relu(output)
		elif cst.ACT_OUT == 'TANH':
			output = tf.nn.tanh(output)
		elif cst.ACT_OUT != 'NONE':
			raise ValueError("[ERROR]: The selected ACT_OUT is not yet considered.")

		"""
		layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
		layer_1 = tf.nn.sigmoid(layer_1)
		layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
		layer_2 = tf.nn.sigmoid(layer_2)
		output = tf.matmul(layer_2, weights['out']) + biases['out']
		output = tf.nn.sigmoid(output)
		"""

		return output

	#-------------------------------------------------------
	""" This function should return the problem domain. """
	@abstractmethod
	def _get_domain(self):
		pass

	#-------------------------------------------------------
	""" This function refers to the problem EDO. """
	@abstractmethod
	def _f(self, x, NN):
		pass

	#-------------------------------------------------------
	""" This function refers to the initial condition of the considered EDO. """
	@abstractmethod
	def _f0(self):
		pass

	#-------------------------------------------------------
	""" This function refers to the EDO's true solution (found analitically). """
	@abstractmethod
	def _true_solution(self, x):
		pass

	#-------------------------------------------------------
	""" This function refers to the universal approximator considered. """
	def _g(self, x):
		return   self._f(x,self._multilayer_perceptron(x))*x + self._f0() # self._multilayer_perceptron(x) + self._f0()

	#-------------------------------------------------------
	""" Custom loss function to approximate the derivatives. """
	def _custom_loss(self):
		summation = []
		for i, x in enumerate(self.x_vec):
			dNN = (self._g(x + self.inf_s) - self._g(x))/self.inf_s
			summation.append((dNN - self.fx_vec[i])**2)
		return tf.reduce_sum(tf.abs(summation))

	#-------------------------------------------------------
	def _train_step(self):
		with tf.GradientTape() as tape:
			loss = self._custom_loss()
		trainable_variables = list(self.weights.values()) + list(self.biases.values())
		gradients = tape.gradient(loss, trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, trainable_variables))

	#-------------------------------------------------------
	def _train_step_tf(self):
		with tf.GradientTape(persistent=True) as tape:
			summation = []
			for i, x in enumerate(self.x_vec):
				x = tf.constant([[[x]]], dtype='float32')
				tape.watch(x)
				dNN = tape.gradient(self._g(x), x)
				summation.append((dNN - self.fx_vec[i])**2)
			loss = tf.reduce_sum(tf.abs(summation))

		trainable_variables = list(self.weights.values()) + list(self.biases.values())
		gradients = tape.gradient(loss, trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, trainable_variables))

	#-------------------------------------------------------
	def train_model(self):
		for i in range(self.training_steps):
			if self.train_mode == 'BASE':
				self._train_step()
			elif self.train_mode == 'TF':
				self._train_step_tf()
			else:
				raise ValueError("[ERROR]: Other TRAIN_MODES are not yet considered.")

			if i % self.display_step == 0:
				print("Loss: %f " % (self._custom_loss()))

	#-------------------------------------------------------
	@abstractmethod
	def plot_results(self):
		pass
