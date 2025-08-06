from pathlib import Path
import sys
import argparse
import ray
from functools import partial

import numpy as np
import torch
import pickle
import shutil

from rl.algos.ppo import PPO
from rl.envs.wrappers import SymmetricEnv
from rl.utils.eval import EvaluateEnv

def import_env(env_name_str):
    if env_name_str=='jvrc_obstacle':
        from envs.jvrc.jvrc_obstacle import JvrcObstacleEnv as Env
    else:
        raise Exception("Check env name!")
    return Env

def run_experiment(args):
    # import the correct environment
    Env = import_env(args.env)

    # wrapper function for creating parallelized envs
    env_fn = partial(Env, path_to_yaml=args.yaml)
    _env = env_fn()
    if not args.no_mirror:
        try:
            print("Wrapping in SymmetricEnv.")
            env_fn = partial(SymmetricEnv, env_fn,
                             mirrored_obs=_env.robot.mirrored_obs,
                             mirrored_act=_env.robot.mirrored_acts,
                             clock_inds=_env.robot.clock_inds)
        except AttributeError as e:
            print("Warning! Cannot use SymmetricEnv.", e)

    # Set up Parallelism
    #os.environ['OMP_NUM_THREADS'] = '1'  # [TODO: Is this needed?]
    if not ray.is_initialized():
        ray.init(num_cpus=args.num_procs)

    # dump hyperparameters
    Path.mkdir(args.logdir, parents=True, exist_ok=True)
    pkl_path = Path(args.logdir, "experiment.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(args, f)

    # copy config file
    if args.yaml:
        config_out_path = Path(args.logdir, "config.yaml")
        shutil.copyfile(args.yaml, config_out_path)

    algo = PPO(env_fn, args)
    algo.train(env_fn, args.n_itr)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    if sys.argv[1] == 'train':
        sys.argv.remove(sys.argv[1])

        parser.add_argument("--env", required=True, type=str)
        parser.add_argument("--logdir", default=Path("/tmp/logs"), type=Path, help="Path to save weights and logs")
        parser.add_argument("--input-norm-steps", type=int, default=100000)
        parser.add_argument("--n-itr", type=int, default=20000, help="Number of iterations of the learning algorithm")
        parser.add_argument("--lr", type=float, default=4e-5, help="Adam learning rate") # Xie
        parser.add_argument("--eps", type=float, default=1e-5, help="Adam epsilon (for numerical stability)")
        parser.add_argument("--lam", type=float, default=0.95, help="Generalized advantage estimate discount")
        parser.add_argument("--gamma", type=float, default=0.99, help="MDP discount")
        parser.add_argument("--std-dev", type=float, default=0.223, help="Action noise for exploration")
        parser.add_argument("--learn-std", action="store_true", help="Exploration noise will be learned")
        parser.add_argument("--entropy-coeff", type=float, default=0.05, help="Coefficient for entropy regularization")
        parser.add_argument("--clip", type=float, default=0.2, help="Clipping parameter for PPO surrogate loss")
        parser.add_argument("--minibatch-size", type=int, default=1024, help="Batch size for PPO updates")
        parser.add_argument("--epochs", type=int, default=5, help="Number of optimization epochs per PPO update") #Xie
        parser.add_argument("--use-gae", type=bool, default=True,help="Whether or not to calculate returns using Generalized Advantage Estimation")
        parser.add_argument("--num-procs", type=int, default=16, help="Number of threads to train on")
        parser.add_argument("--batch-size", type=int, default=21504, help="Total batch size for one PPO update")
        parser.add_argument("--max-grad-norm", type=float, default=0.05, help="Value to clip gradients at")
        parser.add_argument("--max-traj-len", type=int, default=1300, help="Max episode horizon")
        parser.add_argument("--no-mirror", required=False, action="store_true", help="to use SymmetricEnv")
        parser.add_argument("--mirror-coeff", required=False, default=1.0, type=float, help="weight for mirror loss")
        parser.add_argument("--eval-freq", required=False, default=100, type=int, help="Frequency of performing evaluation")
        parser.add_argument("--continued", required=False, type=Path, help="path to pretrained weights")
        parser.add_argument("--recurrent", required=False, action="store_true", help="use LSTM instead of FF")
        parser.add_argument("--imitate", required=False, type=str, default=None, help="Policy to imitate")
        parser.add_argument("--imitate-coeff", required=False, type=float, default=0.3, help="Coefficient for imitation loss")
        parser.add_argument("--yaml", required=False, type=str, default=None, help="Path to config file passed to Env class")
        args = parser.parse_args()

        run_experiment(args)

    elif sys.argv[1] == 'eval':
        sys.argv.remove(sys.argv[1])

        parser.add_argument("--path", required=True, type=Path,
                            help="Path to a trained model directory OR a specific .pt file")
        parser.add_argument("--out-dir", required=False, type=Path, default=None,
                            help="Path to directory to save videos")
        parser.add_argument("--ep-len", required=False, type=int, default=80,
                            help="Episode length to play (in seconds)")
        args = parser.parse_args()

        path_to_actor = ""
        log_dir = None

        if args.path.is_file() and args.path.suffix == ".pt":
            path_to_actor = args.path
            log_dir = args.path.parent
            print(f"Loading specific model file: {path_to_actor}")
        elif args.path.is_dir():
            log_dir = args.path
            # Default to loading the "best" model if a directory is provided
            path_to_actor = Path(log_dir, "actor_best.pt")
            if not path_to_actor.exists():
                 # Fallback to the old default if best is not found
                path_to_actor_fallback = Path(log_dir, "actor.pt")
                if not path_to_actor_fallback.exists():
                    raise FileNotFoundError(f"Could not find 'actor_best.pt' or 'actor.pt' in directory: {log_dir}")
                path_to_actor = path_to_actor_fallback
            print(f"Loading model from directory: {path_to_actor}")
        else:
            raise Exception("Invalid path provided. Must be a directory or a .pt file: ", args.path)

        path_to_critic = Path(log_dir, "critic" + str(path_to_actor.name).split('actor')[1])
        path_to_pkl = Path(log_dir, "experiment.pkl")

        # --- Device Selection ---
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")

        # load experiment args
        run_args = pickle.load(open(path_to_pkl, "rb"))
        # load trained policy and move to device
        policy = torch.load(path_to_actor, map_location=device)
        critic = torch.load(path_to_critic, map_location=device)
        policy.eval()
        critic.eval()

        # import the correct environment
        Env = import_env(run_args.env)
        if "yaml" in run_args and run_args.yaml is not None:
            yaml_path = Path(run_args.yaml)
        else:
            yaml_path = None
        env = partial(Env, yaml_path)()

        # run
        e = EvaluateEnv(env, policy, args, device)
        e.run()
