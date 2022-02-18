import os, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

from ude import UDE
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#************************************************************
#************************************************************
class ToyModel(tf.keras.Model):
	def __init__(self, a, **kwargs):
		super().__init__(**kwargs)
		self.a = a

	#--------------------------------------------------------
	""" This function refers to the original ODE model. """
	@tf.function
	def call(self, t, u):
		dudt = self.a * u
		return dudt

#************************************************************
#************************************************************
class ToyModelTrainable(tf.keras.Model):
	def __init__(self, network, **kwargs):
		super().__init__(**kwargs)
		self.eqn = network

	#--------------------------------------------------------
	""" This function refers to the neural ODE component. """
	@tf.function
	def call(self, t, u):
		n_out = self.eqn(tf.reshape(u, [1, 1]))
		pred = n_out[0, 0]
		dudt = pred

		return dudt

#************************************************************
#************************************************************
class ToyModelSINDy(tf.keras.Model):
	def __init__(self, seed, **kwargs):
		super().__init__(**kwargs)
		# We can also initialize with the values found by SINDy for much faster convergence
		# self.parameters = tf.Variable([1.856], dtype=tf.float64)
		self.parameters = tf.Variable(tf.random.uniform(shape=[1], dtype=tf.float64, seed=seed))

	#--------------------------------------------------------
	""" This function refers to the SINDy results (after first problem run). """
	@tf.function
	def call(self, t, u):
		params = tf.math.abs(self.parameters)
		a = tf.unstack(params)

		dudt = a * u
		return dudt

#************************************************************
#************************************************************
class UDEToyModel(UDE):
	def __init__(self):
		super().__init__()
	
	#-------------------------------------------------------
	""" This function should return the base model domain. """
	def _set_domain(self, points):
		return tf.linspace(0.0, 1.0, points)

	#-------------------------------------------------------
	""" This function refers to the initial condition of the considered EDO. """
	def _set_initial_condition(self):
		return tf.convert_to_tensor([1], dtype=tf.float64)

	#-------------------------------------------------------
	""" This function returns the original ODE model. """
	def _get_base_model(self):
		return ToyModel(2.0)

	#-------------------------------------------------------
	""" This function returns the initial model to be trained. """
	def _get_init_model(self):
		return ToyModelTrainable(self.network)

	#-------------------------------------------------------
	""" This function returns the SINDy model to be trained. """
	def _get_sindy_model(self):
		return ToyModelSINDy(self.seed)

	#-------------------------------------------------------
	""" This function should return the training domain. """
	def _set_train_domain(self, points):
		return tf.linspace(0.0, 0.1, points)

	#-------------------------------------------------------
	""" This function plots the results and training points. """
	def plot_results(self):
		plt.figure(figsize=(9,7), dpi=300)
		plt.rcParams.update({'font.size': 15})

		plt.plot(self.eval_domain, self.trained_model_extrapolation, label='Neural Network Approximation', color='blue')
		plt.plot(self.eval_domain, self.real_solution_complete_domain, label='ODE Solution', color='red')
		plt.scatter(self.train_domain, self.train_solution, label="Train Points", color='red')
		plt.legend(loc=2)
		plt.xlabel('t')
		plt.ylabel('solution')
		plt.tight_layout()
		plt.savefig(f'./toy_seed={self.seed}_ilr={self.initial_learning_rate}_epochs={self.epochs}_layers={self.layers}_opt=adam+bfgs_hidden={self.act_hidden_name}_out={self.act_out_name}_train={self.train_points}p.png')
