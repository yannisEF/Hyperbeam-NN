import argparse
import pickle
import scoop

import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

from utils import make_path


def run_policy(policies, nb_eval, nb=None):
    print("Starting", nb)
    policy_kwargs = dict(net_arch=dict(pi=[12, 12], qf=[40, 25]))
    model = SAC("MlpPolicy", "Pendulum-v1", policy_kwargs=policy_kwargs)

    line = []
    for i, policy in enumerate(policies):
        model.policy.load_from_vector(policy)
        score_mean, score_std = evaluate_policy(
            model, model.env,
            n_eval_episodes=nb_eval,
            warn=False
        )
        line.append(score_mean)
    
    print("Finishing", nb)
    return line

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--inputFolder', default='Policies/', type=str)
    parser.add_argument('--outputFolder', default='Results/', type=str)
    parser.add_argument('--name', default='test_1701445754', type=str)
    
    parser.add_argument('--nb_episodes', default=5, type=int)
    args = parser.parse_args()

    make_path(args.outputFolder + args.name)
    input_path = args.inputFolder + args.name + '/'
    output_path = args.outputFolder + args.name + '/'

    env = gym.make("Pendulum-v1")

    for n_file in range(25):
        
        input_name = input_path + str(n_file) + '.pkl'
        with open(input_name, 'rb') as handle:
            landscape = pickle.load(handle)

        results = list(scoop.futures.map(
            run_policy,
            landscape,
            [args.nb_episodes] * len(landscape),
            list(range(len(landscape)))
        ))

        output_name = output_path + str(n_file) + '.pkl'
        with open(output_name, 'wb') as handle:
            pickle.dump(results, handle)
