import os
import pickle
import random
from itertools import count

import gym
import nle
import numpy as np
import torch
import torch.nn.functional as F
# NOTE: `pip install ansicolors`
from colors import *
from torch import nn, optim

import wandb
from agents.ActorCriticAgent import ActorCriticAgent
from wrappers.ShapeReward import BotWrapper

action_names = {
    0: "More",
    1: "North",
    2: "East",
    3: "South",
    4: "West",
    5: "North-East",
    6: "Sout-East",
    7: "South-West",
    8: "North-West",
    9: "North Long",
    10: "East Long",
    11: "South Long",
    12: "West Long",
    13: "North-East Long",
    14: "Sout-East Long",
    15: "South-West Long",
    16: "North-West Long",
    17: "Up",
    18: "Down",
    19: "Wait",
    20: "Kick",
    21: "Eat",
    22: "Search"
}

def train():
    hyperparams = {
        "lr": 0.002,                    # the learning rate
        "seed": 432,                   # which seed to use
        "gamma": 0.9,                 # the discount factor
        "maxSteps": 1e7,               # Maximum steps before ending an episode
        # "updateRate": 5000,             # Environment steps between optimisation steps
        "filename": "v4latest.pt",     # Filename to save the model to
        "shapedRewards": False,         # Whether the reward is shaped
    }
    tags = ["Actor Critic"]
    notes = """Smooth l1 loss"""

    env = gym.make("NetHackScore-v0")
    if hyperparams["shapedRewards"]:
        env = BotWrapper(env)
        tags.append("Shaped Reward")
    else:
        tags.append("Standard Reward")

    if hyperparams["seed"] is not None:
        seed = hyperparams["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        env.seed(seed)

    wandb.init(config=hyperparams, tags=tags, notes=notes)
    wandb.save(hyperparams["filename"])

    agent = ActorCriticAgent(env.observation_space,
                             env.action_space, training=True)
    # Low log frequency because updating weights is relatively rare
    wandb.watch(agent.model, log_freq=10)

    optimiser = optim.RMSprop(agent.model.parameters(),
                              lr=hyperparams["lr"])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=10)

    bestReward = 0

    episodeRewards = []
    try:
        for episode in count():
            # Infinite episodes I guess
            print(green(underline(f"Episode {episode}:")))
            # Generate an episode
            state = env.reset()
            episodeReward = 0

            step = 0
            action_counts = [0] * env.action_space.n
            for step in range(int(hyperparams["maxSteps"])):
                action = agent.act(state)
                action_counts[action] += 1

                state, reward, done, info = env.step(action)

                agent.rewards.append(reward)
                episodeReward += reward

                if done:
                    actions = [[label, value]  for label, value in zip(action_names.keys(), action_counts)]
                    table = wandb.Table(data=actions, columns=["Action", "Frequency"])
                    # To track episode stats
                    wandb.log({
                        "Episode": episode,
                        "Episode Duration": step,
                        "Total Reward": episodeReward,
                        "Actions": wandb.plot.bar(table, "Action", "Frequency", title="Actions Taken")
                    })
                    break

                print(blue("Step: ")+yellow(step), blue("\tReward: ") +
                      yellow(f"{episodeReward:.2f}"), end="\r")
            print(
                f"Ended episode in {cyan(f'{step}')} steps with total score of {cyan(f'{episodeReward:.2f}')}")

            if (episodeReward > bestReward):
                bestReward = episodeReward
                print(yellow(blink2("New Best!")))
                torch.save(agent.model.state_dict(), os.path.join(
                    "/root/nethack/models", "best.pt"))

            episodeRewards.append(episodeReward)

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
                criticLoss.append(F.smooth_l1_loss(value,
                                             torch.tensor([R], device=agent.device)))

            policyLoss = torch.stack(policyLoss).sum()
            criticLoss = torch.stack(criticLoss).sum()

            loss = policyLoss + criticLoss
            loss.backward()

            nn.utils.clip_grad_norm_(agent.model.parameters(), 40.)

            optimiser.step()
            agent.reset()

            # To track loss
            wandb.log({
                "Policy Loss": policyLoss,
                "Critic Loss": criticLoss
            })

            # And also when we update
            torch.save(agent.model.state_dict(), os.path.join(
                "/root/nethack/models", hyperparams["filename"]))

    except KeyboardInterrupt:
        print(red("Stopping..."))

    wandb.finish()
    return agent


if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "run"
    wandb.login()
    train()
