# !/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
from mpi4py import MPI
import pdb
from gym_extensions.continuous import mujoco
import gym_miniworld

from baselines import logger
import sys

def train(env_id, num_timesteps, seed, num_options,app, saves ,wsaves, epoch,dc, render=False, caption='', deoc=False, tradeoff=0.1, term_mult=1.0, lr_mult=1.0, tdeoc=False):
    from baselines.Termination_DEOC import cnn_policy, pposgd_simple
    # U.make_session(num_cpu=1).__enter__()
    # set_global_seeds(seed)
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank() if seed is not None else None
    set_global_seeds(workerseed)
    env = gym.make(env_id)
    env.seed(workerseed)

    def policy_fn(name, ob_space, ac_space):
        return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=64, num_hid_layers=2, num_options=num_options, dc=dc)

    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), str(rank)))
    if num_options ==1:
        optimsize=64
    elif num_options >1 and num_options < 5:
        optimsize=32
    else:
        print("Only upto 3 options or primitive actions is currently supported.")
        sys.exit()

    # ATARI HYPERPARAMETERS
    # pposgd_simple.learn(env, policy_fn,
    #                     max_timesteps=num_timesteps*1.1,
    #                     timesteps_per_batch=256,
    #                     clip_param=0.2, entcoeff=0.001,
    #                     optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=optimsize,
    #                     gamma=0.99, lam=0.95, schedule='linear', num_options=num_options,
    #                     app=app, saves=saves, wsaves=wsaves, epoch=epoch, seed=seed,dc=dc, render=render, caption=caption,
    #                     deoc=deoc, tradeoff=tradeoff, term_mult=term_mult, lr_mult=lr_mult, tdeoc=tdeoc
    #                     )

    # MINIWORLD HYPERPARAMETERS
    pposgd_simple.learn(env, policy_fn,
                        max_timesteps=num_timesteps,
                        timesteps_per_batch=2048,
                        clip_param=0.2, entcoeff=0.01,
                        optim_epochs=4, optim_stepsize=3e-4, optim_batchsize=optimsize,
                        gamma=0.99, lam=0.95, schedule='linear', num_options=num_options,
                        app=app, saves=saves, wsaves=wsaves, epoch=epoch, seed=seed,dc=dc, render=render, caption=caption,
                        deoc=deoc, tradeoff=tradeoff, term_mult=term_mult, lr_mult=lr_mult, tdeoc=tdeoc
                        )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='MiniWorld-OneRoom-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=16)
    parser.add_argument('--opt', help='number of options', type=int, default=2)
    parser.add_argument('--app', help='Append to folder name', type=str, default='')
    parser.add_argument('--saves', dest='saves', action='store_true', default=False)
    parser.add_argument('--wsaves', dest='wsaves', action='store_true', default=False)
    parser.add_argument('--epoch', help='Epoch', type=int, default=-1)
    parser.add_argument('--dc', type=float, default=0.)
    parser.add_argument('--render', dest='render', action='store_true', default=False)
    parser.add_argument('--caption', help='Caption for run', default='')
    parser.add_argument('--deoc', help='Augment reward with diversity', action='store_true', default=False)
    parser.add_argument('--tradeoff', type=float, default=0.0)
    parser.add_argument('--term_mult', type=float, default=1.0)
    parser.add_argument('--lr_mult', type=float, default=1.0)
    parser.add_argument('--tdeoc', help='Use diversity in termination objective', action='store_true', default=False)



    args = parser.parse_args()

    if args.tdeoc and not args.deoc:
        print("Setting deoc arg to True...")
        args.deoc = True

    train(args.env, num_timesteps=2e6, seed=args.seed, num_options=args.opt, app=args.app, saves=args.saves,
          wsaves=args.wsaves, epoch=args.epoch,dc=args.dc,
          render=args.render, caption=args.caption, deoc=args.deoc, tradeoff=args.tradeoff, term_mult=args.term_mult, lr_mult=args.lr_mult, tdeoc=args.tdeoc)


if __name__ == '__main__':
    main()