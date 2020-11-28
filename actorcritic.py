import os
import pickle
import random
from itertools import count

import gym
import nle
import numpy as np
import torch
import torch.nn.functional as F
import torchviz
# NOTE: `pip install ansicolors`
from colors import *
from torch import optim

import wandb
from agents.ActorCriticAgent import ActorCriticAgent


def train():
    hyperparams = {
        "lr": 0.001,                   # the learning rate
        "seed": 432,                   # which seed to use
        "gamma": 0.99,                 # the discount factor
        "maxSteps": 1e7,               # Maximum steps before ending an episode
        "updateRate": 500,             # Environment steps between optimisation steps
        "filename": "v0latest.pt",     # Filename to save the model to
    }

    env = gym.make("NetHackScore-v0")
    # env = TensorWrapper(env)
    # env = StatsWrapper(env)

    if hyperparams["seed"] is not None:
        seed = hyperparams["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        env.seed(seed)

    wandb.init(config=hyperparams, tags=["Actor Critic"], )
    wandb.save(hyperparams["filename"])

    agent = ActorCriticAgent(env.observation_space,
                             env.action_space, training=True)
    # Low log frequency because updating weights is relatively rare
    wandb.watch(agent.model, log_freq=10)

    optimiser = optim.Adam(agent.model.parameters(),
                           lr=hyperparams["lr"])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=10)

    episodeRewards = []
    try:
        for episode in count():
            # Infinite episodes I guess
            print(green(underline(f"Episode {episode}:")))
            # Generate an episode
            state = env.reset()
            episodeReward = 0

            step = 0
            for step in range(int(hyperparams["maxSteps"])):
                action = agent.act(state)

                state, reward, done, info = env.step(action)

                agent.rewards.append(reward)
                episodeReward += reward

                print(blue("Step: ")+yellow(step), blue("\tReward: ") +
                      yellow(f"{episodeReward:.2f}"), end="\r")
                if done or (step+1) % hyperparams["updateRate"] == 0:
                    # Actual Learning Code
                    optimiser.zero_grad()
                    policyLoss = []  # Loss for the actor
                    criticLoss = []  # Loss for the critic
                    returns = []

                    R = 0
                    for r in agent.rewards[::-1]:
                        R = r + hyperparams["gamma"] * R
                        returns.insert(0, R)

                    returns = torch.tensor(returns)
                    # Normalise the returns
                    returns = (returns - returns.mean()) / \
                              (returns.std() + 1e-9)

                    for (log_prob, value), R in zip(agent.actions, returns):
                        advantage = R - value.item()
                        policyLoss.append(-log_prob * advantage)
                        criticLoss.append(F.mse_loss(
                            value, torch.tensor([R], device=agent.device)))

                    policyLoss = torch.stack(policyLoss).sum()
                    criticLoss = torch.stack(criticLoss).sum()

                    loss = policyLoss + criticLoss
                    loss.backward()
                    torchviz.make_dot(loss).save("/root/nethack/graph.gv")

                    optimiser.step()
                    agent.resetHistory()

                    # To track loss
                    wandb.log({
                        "Policy Loss": policyLoss,
                        "Critic Loss": criticLoss
                    })

                    # Save the weights whenever we update them
                    torch.save(agent.model.state_dict(), os.path.join(
                        "/root/nethack/models", hyperparams["filename"]))

                    if (done):
                        # TODO: Track ttyrec recording
                        # TODO: Track actual actions
                        # 'ttyrec', string
                        # To track episode stats
                        wandb.log({
                            "Episode": episode,
                            "Episode Duration": step,
                            "Total Reward": episodeReward,
                            # These are always 0 for some stupid reason
                            # "Max HP": last_state["blstats"][11],
                            # "Depth": last_state["blstats"][12],
                            # "Gold": last_state["blstats"][13],
                            # "Level": last_state["blstats"][18],

                            # This is always DEATH, because agent is bad.
                            # WandB cant show it nice anyway
                            # "End Status": info["end_status"].name
                        })
                        break
            print(
                f"Ended episode with total score of {cyan(f'{episodeReward:.2f}')}")
            episodeRewards.append(episodeReward)

    except KeyboardInterrupt:
        print(red("Stopping..."))

    wandb.finish()
    return agent


if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "run"
    wandb.login()
    train()