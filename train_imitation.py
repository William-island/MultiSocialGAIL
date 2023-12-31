import os
import argparse
from datetime import datetime
import torch

from gail_airl_ppo.env import make_env
from gail_airl_ppo.buffer import *
from gail_airl_ppo.algo import ALGOS
from gail_airl_ppo.trainer import Trainer


def run(args):
    env = make_env(args,'Train')
    env_test = make_env(args,'Test')
    buffer_exp = SerializedBuffer_SA(
        path=args.buffer,
        device=torch.device(args.device if args.cuda else "cpu")
    )

    algo = ALGOS[args.algo](
        buffer_exp=buffer_exp,
        graph_feature_channels=args.graph_obs_past_len,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device(args.device if args.cuda else "cpu"),
        seed=args.seed,
        rollout_length=args.rollout_length
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, args.algo, f'seed{args.seed}-{time}')

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--buffer', type=str, default='./expert_traj/graph_demos.pth')  # small_tester.pth | graph_demos.pth 
    p.add_argument('--rollout_length', type=int, default=2048)  # 2048
    p.add_argument('--num_steps', type=int, default=int(4e4))   # 5.8e5 | 1.2e6
    p.add_argument('--eval_interval', type=int, default=500)       #  5000
    p.add_argument('--env_id', type=str, default="CrowdEnv")
    p.add_argument('--algo', type=str, default='gail')
    p.add_argument('--cuda', default=True, action='store_true')
    p.add_argument('--seed', type=int, default=0)

    # step_mode
    p.add_argument('--step_mode',default='multi')   # 'multi' for multi-agent | or 'replay'
    p.add_argument('--device', default='cuda:1')

    # Coustormed parser
    p.add_argument('--dataset_path',default="./gail_airl_ppo/crowd_env/datasets/GC_all.pkl")  # GC_all.pkl
    p.add_argument('--frame_interval',default=10)
    p.add_argument('--time_interval',default=0.4)
    p.add_argument('--regions',default=16)
    p.add_argument('--radius',default=6.0)
    p.add_argument('--map_size_bound',default=[-10,40,-20,50])         # [low_x, high_x, low_y, high_y] (int)
    p.add_argument('--with_last_speed',default=False)
    p.add_argument('--entity_type',default=False)
    p.add_argument('--observation_type',default='graph')               # 'radar' / 'graph'
    p.add_argument('--graph_obs_past_len',default=5)
    p.add_argument('--padd_to_number',default=70)                # 60      # the max number of people in radius to form a mini-batch
    p.add_argument('--graph_feature_len',default=5)
    p.add_argument('--output_len',default=2)
    # other hyperparamater
    # p.add_argument('--pretrained_model_path',default='./pretrained_models/GC_GNN.pt')
    
    args = p.parse_args()
    run(args)
