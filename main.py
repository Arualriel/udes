import os, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

import numpy as np
from pysindy import SR3, SINDy
from pysindy.feature_library.polynomial_library import PolynomialLibrary
from tfdiffeq.sindy_utils import STRRidge
from ude_toy import UDEToyModel

# Toy Problem
ude_toy = UDEToyModel()
ude_toy.train_model()
ude_toy.print_active_model()
ude_toy.plot_results()

# Find Underlying Equations
real_results_np = ude_toy.real_solution_complete_domain.numpy()
model_preds_np = ude_toy.trained_model_extrapolation.numpy()
t_eval_np = ude_toy.eval_domain.numpy()

# Optimize Hyper Parameters & Polynomial Library
#sindy_optm = STRRidge(threshold=0.5, alpha=1.0)
sindy_optm = SR3(threshold=0.1, nu=1.0, max_iter=100)
sindy_library = PolynomialLibrary(degree=10, include_interaction=True, interaction_only=True)

# Fit SINDy Model
sindy_model = SINDy(optimizer=sindy_optm, feature_library=sindy_library, discrete_time=False)
sindy_model.fit(model_preds_np, t_eval_np)

# Print SINDy Solution
print('---')
print('4. [Computing SINDy Solution]')
print('SINDy Model: ', end='')
sindy_model.print()
sindy_model.coefficients()
sindy_model.equations()
print(f'SINDy Score: {sindy_model.score(model_preds_np, t_eval_np)}')
print('---')

# Change Trainable Model & Recalculate Solution
ude_toy.change_to_sindy_model()
ude_toy.train_model()
ude_toy.print_active_model()
ude_toy.plot_results()
