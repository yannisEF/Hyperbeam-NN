"""Saves a sampled beam layer by layer (storage space proportional to policy dimension).
Example : python3 model_prepare.py

FIXME: SB3's call to torch can sometimes throw an error due to multiprocessing, asking to use the 'spawn' method. However, doing so causes sub-processes to be killed (high memory use). For an unkown reason, this problem resolves eventually on its own...?"""

import json
import argparse
import torch.multiprocessing as mp

from stable_baselines3 import __dict__ as sb3_dict
from stable_baselines3.common.evaluation import evaluate_policy

from tqdm import tqdm

from utils_sample import *


def run_policy(policies, tuple_model, nb_eval):
	"""Evaluate a model with given policy."""

	model = tuple_model[0](**tuple_model[1])

	line = []
	for policy in policies:
		model.policy.load_from_vector(policy)

		score_mean, score_std = evaluate_policy(
			model, model.env,
			n_eval_episodes=nb_eval
		)

		line.append(score_mean)

	return line


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--parameters', default='parameters.json', type=str,
		help="Path to the json parameters file."
	)
	args = parser.parse_args()

	# Retrieving parameters
	parameters = json.load(open(args.parameters))
	parameters_training = parameters["training"]
	parameters_beam = parameters["beam"]

	algorithm = sb3_dict[parameters["algorithm"]]

	input_path = "Models/{}".format(parameters["name_model"])
	save_path = "Results/{}".format(parameters["name_result"])

	# Retrieving the policies
	print("Retrieving the policies..")
	policies = [
		model.policy.parameters_to_vector()
		for model in [
			algorithm.load(
				"{}/{}_{}_steps".format(input_path, parameters["name_model"], name)
			) for name in sorted(list(parameters_beam["vector"].values()))
		]
	]

	# Computing the beam
	print("Computing the beam..")
	# 	We find the vector corresponding to the learning between two policies
	u = policies[1] - policies[0]
	# 	We gather an orthonormal basis for the hyperplane orthogonal to that vector
	print("Finding hyperplane..")
	basis = get_hyperplane(u, verbose=True)

	# 	We set up a sampling rule for that hyperplane
	# TODO: Think about appropriate sampling rule
	print("Sampling original surface..")
	origin = policies[0]
	combination = combination_sphere(
		len(basis),
		np.linalg.norm(u),
		parameters_beam["nb_lines"]
	)
	S = center_around(origin, basis, combination)

	# 	Now we sample the beam created by shifting our hyperplane by u
	print("Shifting the surface to to create the beam's layers..")
	beam = sample_beam(S, u, parameters_beam["nb_layers"])

	# Sampling the beam's layers and evaluating them
	# import warnings	#	Dirty but nicer to remove SCOOP's warnings...
	# warnings.filterwarnings("ignore")

	#	Preparing the models for parallelization
	dict_model_params = {
			"policy":parameters["policy"],
			"env":parameters["environment"],
			"policy_kwargs":parameters_training["policy_kwargs"]
	}

	#	Evaluation and sampling loop
	indices = None
	list_results = []
	for layer in tqdm(beam, desc="Layer", position=0):

		# 	We get a landscape by sampling a line to the center of the layer for each point
		landscape = sample_landscape(layer, parameters_beam["nb_columns"] // 2)

		# 	Indices will remember a coherent order for each line across layers
		if indices is None:
			indices = utils.order_neighbours(landscape)[-1]

		# 	Evaluating each point
		landscape = iter(landscape)
		with mp.Pool(processes=mp.cpu_count()) as pool:
			list_results.append(pool.starmap(
				run_policy,
				[
					(
						iter(line),
						(algorithm, dict_model_params),
						parameters_beam["nb_episodes_per_eval"]
					) for line in landscape
				]
			))
	
	#	Now that all is done, we order the landscapes coherently
	print("Re-organizing lines")
	list_ordered_results = list(map(
		lambda x: utils.insert_with_indices(x, indices),
		list_results
	))
	
	# Saving results
	print("Saving results")
	with open("{}.pkl".format(save_path), 'wb') as handle:
		pickle.dump(list_ordered_results, handle)
