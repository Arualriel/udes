# Code Constants
import constants as cst

# General Parameters (Setup)
SEED = 7542            # Fixed Seed? None, 4, 8, 15, 16, 23, 42, (...) (FOR TOY: 7542, FOR LK: 42)
GPU = True             # Use GPU?
INITIAL_LEARNING_RATE = 1.0e-2
DECAY_STEPS = 100
DECAY_RATE = 0.5
EPOCHS = 100
DISPLAY_STEP = 10

# Network Parameters (Setup)
NEURONS_INPUT = 1      # Input Layer: #Neurons (FOR TOY: 1, FOR LK: 2)
NEURONS_OUTPUT = 1     # Output Layer: #Neurons (FOR TOY: 1, FOR LK: 2)
LAYERS = [32, 32, 32]  # Network Layers
BFGS_TOL = 1.0e-6      # BFGS Tolerance
BFGS_MAX_ITER = 50     # BFGS Max Iterations
ACT_HIDDEN = cst.SWISH # Activation: SWISH, RELU, SIGMOID, TANH, None
ACT_OUT = cst.RELU     # Activation: SWISH, RELU, SIGMOID, TANH, None

# Train & Evaluation Parameters (Setup)
TRAIN_POINTS = 30
EVAL_POINTS = 1000

# SEED Tips:
# According to Douglas Adams, the answer to the ultimate question of life, the universe and everything is 42.
