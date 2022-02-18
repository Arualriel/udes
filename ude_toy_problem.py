from ude import UDE
import numpy as np
import matplotlib.pyplot as plt

#************************************************************
#************************************************************
#************************************************************
class UDEToyProblem(UDE):
	def __init__(self):
		super().__init__()

	#-------------------------------------------------------
	""" This function should return the problem domain. """
	def _get_domain(self, points):
		print()
		return np.linspace(0, 1, points) # Should we use points + 1 instead of points?

	#-------------------------------------------------------
	""" This function refers to the problem EDO. """
	def _f(self, x):
		return 2*x

	#-------------------------------------------------------
	""" This function refers to the initial condition of the considered EDO. """
	def _f0(self):
		return 1.0

	#-------------------------------------------------------
	""" This function refers to the EDO's true solution (found analitically). """
	def _true_solution(self, x):
		return x**2 + 1
 
	#-------------------------------------------------------
	def plot_results(self):
		plt.figure(figsize=(9,7), dpi=300)
		plt.rcParams.update({'font.size': 15})

		result = []
		x_vec_plot = self._get_domain(self.plot_points)
		for val in x_vec_plot:
			result.append(self._g(val).numpy()[0][0][0])

		S = self._true_solution(x_vec_plot)
		S_train = self._true_solution(self.x_vec)

		plt.plot(x_vec_plot, result, label="Neural Network Approximation", color='blue')
		plt.plot(x_vec_plot, S, label="ODE Solution", color='red')
		plt.scatter(self.x_vec, S_train, label="Train Points", color='red')
		plt.legend(loc=2)
		plt.xlabel('t')
		plt.ylabel('u(t)')
		plt.tight_layout()
		plt.savefig('./example1.png')
