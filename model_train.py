"""Instructions to train a model using SB3.
Example : python3 model_train.py"""

import json
import argparse
import stable_baselines3

import gymnasium as gym

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

from utils import make_path


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

    save_path = "Models/{}".format(parameters["name_model"])
    make_path(save_path)

    # Initializing the model and its evaluation environment
    eval_env = gym.make(parameters["environment"])
    model = stable_baselines3.__dict__[parameters["algorithm"]](
        parameters["policy"],
        parameters["environment"],
        policy_kwargs=parameters_training["policy_kwargs"]
    )
    
    # Creating the Callbacks
    checkpoint_callback = CheckpointCallback(
        name_prefix=parameters["name_model"],
        save_freq=parameters_training["save_frequency"],
        save_path=save_path
    )

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=parameters_training["save_frequency"],
        best_model_save_path=save_path,
        log_path=save_path
    )

    list_callback = CallbackList([
        checkpoint_callback,
        eval_callback
    ])

    # Starting the learning process
    model.learn(parameters_training["nb_steps"], callback=list_callback)
