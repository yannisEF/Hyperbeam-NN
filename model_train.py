import gymnasium as gym
import argparse

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

# Saves a model's training process
# python3 trainModel.py --env "Pendulum-v1" --save_freq 500 --max_learn 10000

if __name__ == "__main__":
    print("Parsing arguments")
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--env', default='Pendulum-v1', type=str)
    parser.add_argument('--policy', default = 'MlpPolicy', type=str) # Policy of the model
    
    # Save parameters
    parser.add_argument('--save_path', default='Models', type=str) # path to save
    parser.add_argument('--name_prefix', default='rl_model', type=str) # prefix of saves' name
    parser.add_argument('--save_freq', default=1000, type=int) # frequency of the save
    parser.add_argument('--max_learn', default=20000, type=int) # Number of steps to learning process
    
    args = parser.parse_args()

    # Creating environment and initialising model and parameters
    print("Creating environment\n")
    eval_env = gym.make(args.env)

    policy_kwargs = dict(net_arch=dict(pi=[12, 12], qf=[40, 25]))
    model = SAC(args.policy, args.env, policy_kwargs=policy_kwargs)
    
    # Creating the Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path=args.save_path,
                                            name_prefix=args.name_prefix, verbose=2)
    eval_callback = EvalCallback(eval_env, eval_freq=args.save_freq, best_model_save_path=args.save_path)
    list_callback = CallbackList([checkpoint_callback, eval_callback])

    # Starting the learning process
    model.learn(args.max_learn, callback=list_callback)