import os, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

import numpy as np
import setup as stp
import constants as cst
import tensorflow as tf
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from tfdiffeq import odeint
from tfdiffeq.bfgs_optimizer import BFGSOptimizer

#************************************************************
#************************************************************
class UDE(ABC):
	def __init__(self):
		# Force float64
		tf.keras.backend.set_floatx('float64')

		# Set Seed
		self.seed = stp.SEED
		self._set_seed()

		# Set GPU/CPU
		self.gpu = stp.GPU
		self.device = self._set_device()

		# General Parameters
		self.initial_learning_rate = stp.INITIAL_LEARNING_RATE
		self.decay_steps = stp.DECAY_STEPS
		self.decay_rate = stp.DECAY_RATE
		self.epochs = stp.EPOCHS
		self.display_step = stp.DISPLAY_STEP

		# Network Parameters & Creation
		self.neurons_input = stp.NEURONS_INPUT
		self.neurons_output = stp.NEURONS_OUTPUT
		self.layers = stp.LAYERS
		self.bfgs_tol = stp.BFGS_TOL
		self.bfgs_max_iter = stp.BFGS_MAX_ITER
		self.act_hidden_name = stp.ACT_HIDDEN
		self.act_out_name = stp.ACT_OUT
		self.act_hidden = self._get_activation(self.act_hidden_name)
		self.act_out = self._get_activation(self.act_out_name)
		self.network = self._define_network()
		
		# Print Network Summary?
		self.network.summary()
		print()

		# Train & Plot Points
		self.train_points = stp.TRAIN_POINTS
		self.eval_points = stp.EVAL_POINTS

		# Select Active Models
		self._set_base_model()
		self._set_init_model()
		self._set_sindy_model()
		self.change_to_init_model()

		# Domains & Initial Condition
		self._domain = self._set_train_domain(self.train_points)
		self.eval_domain = self._set_domain(self.eval_points)
		self.initial_condition = self._set_initial_condition()

	#--------------------------------------------------------
	""" This function sets the device (GPU/CPU) that will be used. """
	def _set_device(self):
		if self.gpu is True:
			return 'gpu:0'
		else:
			return 'cpu:0'

	#--------------------------------------------------------
	""" This function sets a seed if the user defines it. """
	def _set_seed(self):
		if self.seed is not None:
			np.random.seed(self.seed)
			tf.random.set_seed(self.seed)

	#-------------------------------------------------------
	""" This function should return the base model domain. """
	@abstractmethod
	def _set_domain(self, points):
		pass

	#-------------------------------------------------------
	""" This function refers to the initial condition of the considered EDO. """
	@abstractmethod
	def _set_initial_condition(self):
		pass

	#-------------------------------------------------------
	""" This function should return the training domain. """
	@abstractmethod
	def _set_train_domain(self, points):
		pass

	#-------------------------------------------------------
	""" This function plots the results and training points. """
	@abstractmethod
	def plot_results(self):
		pass

	#--------------------------------------------------------
	""" This function returns the original ODE model. """
	@abstractmethod
	def _get_base_model(self):
		pass

	#--------------------------------------------------------
	""" This function returns the initial model to be trained. """
	@abstractmethod
	def _get_init_model(self):
		pass

	#--------------------------------------------------------
	""" This function returns the SINDy model to be trained. """
	@abstractmethod
	def _get_sindy_model(self):
		pass

	#--------------------------------------------------------
	""" This function creates a variable to store the base ODE model. """
	def _set_base_model(self):
		self.base_model = self._get_base_model()

	#--------------------------------------------------------
	""" This function creates a variable to store the initial model. """
	def _set_init_model(self):
		self.init_model = self._get_init_model()

	#--------------------------------------------------------
	""" This function creates a variable to store the SINDy model. """
	def _set_sindy_model(self):
		self.sindy_model = self._get_sindy_model()
	
	#--------------------------------------------------------
	""" This function sets the active model to be the initial model. """
	def change_to_init_model(self):
		self.active_model = self.init_model

	#--------------------------------------------------------
	""" This function sets the active model to be the SINDy model. """
	def change_to_sindy_model(self):
		self.active_model = self.sindy_model

	#-------------------------------------------------------
	""" Selects the activation functions to be used for the network. """
	def _get_activation(self, act_name):
		if act_name == cst.SWISH:
			return tf.nn.swish
		elif act_name == cst.RELU:
			return tf.nn.relu
		elif act_name == cst.SIGMOID:
			return tf.nn.sigmoid
		elif act_name == cst.TANH:
			return tf.nn.tanh
		elif act_name is None:
			return None
		else:
			raise ValueError("[ERROR]: The selected ACTIVATION_FUNCTION is not yet considered.")

	#--------------------------------------------------------
	""" This function defines the network that will be used in the active model. """
	def _define_network(self):
		network_layers = []

		for i in range(len(self.layers)):
			if i == 0:
				network_layers.append(tf.keras.layers.Dense(self.layers[i], activation=self.act_hidden, input_shape=(self.neurons_input,)))
			else:
				network_layers.append(tf.keras.layers.Dense(self.layers[i], activation=self.act_hidden))

		network_layers.append(tf.keras.layers.Dense(self.neurons_output, activation=self.act_out))
		return tf.keras.Sequential(network_layers)

		"""
		return tf.keras.Sequential([
			tf.keras.layers.Dense(32, activation=tf.nn.swish, input_shape=(1,)),
			tf.keras.layers.Dense(32, activation=tf.nn.swish),
			tf.keras.layers.Dense(32, activation=tf.nn.swish),
			tf.keras.layers.Dense(1, activation='relu')
		])
		"""

	#--------------------------------------------------------
	"""
	This function trains the model using Adam & BGFS.
	1. We pretrain the model for a few epochs to reduce the loss quickly;
	2. We retrain the model using BFGS to reduce the loss to a small tolerance (let's say 1E-06).
	"""
	def train_model(self):
		with tf.device(self.device):
			# Adam
			learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(self.initial_learning_rate, self.decay_steps, self.decay_rate, staircase=False)
			optimizer = tf.keras.optimizers.Adam(learning_rate)

			print('1. [Computing Train Solution & Generating Plot]')
			self.train_solution = odeint(self.base_model, self.initial_condition, self.train_domain)

			print('2. [Adam Optimizer]')
			for epoch in range(self.epochs):
				with tf.GradientTape() as tape:
					loss = self._loss_wrapper(self.active_model)
				grads = tape.gradient(loss, self.active_model.trainable_variables)
				optimizer.apply_gradients(zip(grads, self.active_model.trainable_variables))

				if epoch % self.display_step == 0:
					print(f'Epoch: {epoch:04d} - Loss: {loss.numpy().mean():.15f} - Learning Rate: {learning_rate(optimizer.iterations).numpy():.15f}')
			print('---')

		# BFGS Finetune
		print('3. [BFGS Finetune]')
		bfgs_optimizer = BFGSOptimizer(max_iterations=self.bfgs_max_iter, tolerance=self.bfgs_tol)
		self.active_model = bfgs_optimizer.minimize(self._loss_wrapper_bfgs, self.active_model)

		# Save Weights
		self.active_model.save_weights('./model_weights/ckpt', save_format='tf')
		
		# Extrapolation
		with tf.device(self.device):
			self.real_solution_complete_domain = odeint(self.base_model, self.initial_condition, self.eval_domain)
			self.trained_model_extrapolation = odeint(self.active_model, self.initial_condition, self.eval_domain)

	#--------------------------------------------------------
	""" This function refers to the loss that will be used by the Adam optimizer. """
	def _loss_wrapper(self, model):
		preds = odeint(model, self.initial_condition, self.train_domain)
		loss = tf.reduce_mean(tf.square(self.train_solution - preds), axis=1)
		return loss

	#--------------------------------------------------------
	""" This function refers to the loss that will be used by the BFGS optimizer. """
	def _loss_wrapper_bfgs(self, model):
		preds = odeint(model, self.initial_condition, self.train_domain, atol=self.bfgs_tol, rtol=self.bfgs_tol)
		loss = tf.reduce_mean(tf.square(self.train_solution - preds))
		return loss

	#--------------------------------------------------------
	""" This function prints all trainable variables of the active model. """
	def print_active_model(self):
		for i, param in enumerate(self.active_model.trainable_variables):
			param = tf.math.abs(param)
			print(f'Layer {i:02d} = ', param)
