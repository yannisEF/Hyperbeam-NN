import argparse
from stable_baselines3 import SAC

from utils_sample import *

# Prepares for comparison a list of policies from the models entered as parameters
# python3 preparePolicies.py --inputNames "rl_model_7000_steps; rl_model_8000_steps" --outputName "pendulum_policies_7000_8000_1000"

def loadFromFile(filenames, folder):
	"""Returns a list of policies"""
	assert len(filenames) == 2
	return [SAC.load('{}/{}'.format(folder, f)) for f in filenames]

if __name__ == "__main__":
	print("Parsing arguments")
	parser = argparse.ArgumentParser()

	parser.add_argument('--inputFolder', default='Models', type=str) # Folder containing the input
	parser.add_argument('--inputNames', default="rl_model_7000_steps; rl_model_8000_steps", type=str) # Names of every model to load, separated by '; '

	args = parser.parse_args()
	
	print("Retrieving the models..")
	models = loadFromFile(filenames=args.inputNames.split('; '), folder=args.inputFolder)
	print("Processing the models' policies..")
	policies = [model.policy.parameters_to_vector() for model in models]

	nb_layers = 25
	nb_lines_per_layer = 50
	pixels_per_line = 50

	# We have a random vector corresponding to the learning between two policies
	u = policies[1] - policies[0]
	# We find the hyperplane orthogonal to that vector
	basis = get_hyperplane(u)

	# We set up a sampling rule for that hyperplane
	origin1 = np.zeros(len(u))
	combination1 = combination_random(len(basis), nb_lines_per_layer)
	S = sample_around(origin1, basis, combination1)

	# Now we sample the beam created by shifting our hyperplane by u
	beam = sample_beam(S, u, nb_layers)

	# For each of point, we trace a line to the center of the layer
	#   ... this is because the hyperplane is N-th dimensional
	#   ... and we have to use another technique later to visualize
	#   ... these N dimensions in 2D or 3D.
	get_landscape_beam(beam, pixels_per_line, save_path="Policies/test")