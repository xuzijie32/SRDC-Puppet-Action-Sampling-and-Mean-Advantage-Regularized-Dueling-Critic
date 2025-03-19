import numpy as np
import torch
import gymnasium as gym
import argparse
import os
import copy
import torch.nn as nn
import torch.nn.functional as F
from utils import ReplayBuffer,eval_policy
from DDPG_SRDC import DDPG_SRDC
from DDPG_original import DDPG_original
from DDPG_retuned import DDPG_retuned
from TD3_SRDC import TD3_SRDC
from TD3_original import TD3_original
from TD3_retuned import TD3_retuned
from SAC_SRDC import SAC_SRDC
from SAC_original import SAC_original
from SAC_retuned import SAC_retuned

# Implementation of Puppet-Action Sampling and Mean-Advantage Regularized Dueling Critic (SRDC)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--SRDC", default=True)  # Set "True" to implement SRDC
    parser.add_argument("--re_tuned", default=True)  # Whether re-tune the network architecture (if SRDC==False)
    parser.add_argument("--adaptive", default=True)  # Whether adaptive the loss coefficient in SRDC(if SRDC==True)
    parser.add_argument("--alpha_SRDC", default=0.75, type=float)  # Fixed value of the loss coefficient in SRDC(if SRDC==True and adaptive==False)
    parser.add_argument("--num_sample", default=5, type=int)  # Number of samples in SRDC(if SRDC==True)
    parser.add_argument("--policy", default="TD3")  # Policy name (DDPG, TD3 or SAC)
    parser.add_argument("--env", default="Ant-v4")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6+25e3, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise(for DDPG and TD3)
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update(for TD3)
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise(for TD3)
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates(for TD3)
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_known_args()[0]

    if args.SRDC:
        mechanism = "SRDC"
    elif args.re_tuned:
        mechanism = "retuned"
    else:
        mechanism = "original"
    file_name = f"{args.policy}-{mechanism}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}-{mechanism}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }
    if args.SRDC:
        kwargs["adaptive"] = args.adaptive
        kwargs["alpha_SRDC"] = args.alpha_SRDC
        kwargs["num_sample"] = args.num_sample

    # Initialize policy
    if args.policy == "DDPG":
        if args.SRDC:
            policy = DDPG_SRDC(**kwargs)
        elif args.re_tuned:
            policy = DDPG_retuned(**kwargs)
        else:
            policy = DDPG_original(**kwargs)
    elif args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        if args.SRDC:
            policy = TD3_SRDC(**kwargs)
        elif args.re_tuned:
            policy = TD3_retuned(**kwargs)
        else:
            policy = TD3_original(**kwargs)
    elif args.policy == "SAC":
        # Target entropy for SAC
        kwargs["target_entropy"] = -env.action_space.shape[0]
        if args.SRDC:
            policy = SAC_SRDC(**kwargs)
        elif args.re_tuned:
            policy = SAC_retuned(**kwargs)
        else:
            policy = SAC_original(**kwargs)


    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    state, done = env.reset(), False
    state = state[0]
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            if args.policy == "SAC":
                action = policy.select_action(np.array(state)).clip(-max_action, max_action)
            else:
                action = (
                        policy.select_action(np.array(state))
                        + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done1, done2, _ = env.step(action)
        done = done1 + done2
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} ")
            # Reset environment
            state, done = env.reset(), False
            state = state[0]
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/{file_name}", evaluations)

    policy.save(f"./models/{file_name}")

