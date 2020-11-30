from train import ACTION_NAMES
import os
import pickle

import gym
import nle
import wandb
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from agents.ActorCriticAgent import ActorCriticAgent


class ReplayDataset(Dataset):
    """A dataset that loads pickled replays from a folder."""
    def __init__(self, root="/root/nethack/pickles", gamma=0.9):
        super().__init__()
        self.root = root
        self.gamma = gamma
        self.filenames = os.listdir(self.root)

    def __getitem__(self, index):
        filename = os.path.join(self.root, self.filenames[index])
        game = pickle.load(open(filename, 'rb'))
        observations, actions = game.values()

        # Extract the states and rewards from the observations
        states, rewards, dones, infos = zip(*observations[1:])
        states = list(states)
        rewards = list(rewards)
        dones = list(dones)
        # Add the initial state
        states.insert(0, observations[0])

        # Calculate the returns
        returns = []
        G = 0
        for r in rewards:
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)

        # Normalise the returns
        returns = (returns - returns.mean()) / \
            (returns.std() + 1e-9)

        return {"States": states, "Actions": actions, "Returns": returns, "Dones": dones}

    def __len__(self):
        return len(self.filenames) 

env = gym.make("NetHackScore-v0") # Create an environment so we can copy the spaces
agent = ActorCriticAgent(env.observation_space, env.action_space, True)

wandb.init(project="nethack-pretraining")
optimiser = torch.optim.SGD(agent.model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=1)
for gameIdx, game in enumerate(ReplayDataset()):
    print(f"Game: {gameIdx}")
    loss = 0
    for step, (state,action,value,done) in enumerate(zip(*game.values())):
        # Compute the model's output
        probs, pred_value = agent.model(state)

        # Calculate loss
        policyLoss = F.nll_loss(probs, torch.tensor([action], device=agent.device))
        criticLoss = F.mse_loss(pred_value, value[None,None].to(agent.device))

        loss += policyLoss + criticLoss
        print(f"Step {step:3d}: {policyLoss:7.3f}, {criticLoss:7.3f} ({ACTION_NAMES[probs.argmax(dim=1).item()]:s}, {ACTION_NAMES[action]:s})                                                           ", end="\r")

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    agent.reset()

    # Test the agent
    print("\nPlaying episode")
    agent.training = False
    episodeReward = 0
    done = False
    state = env.reset()
    while not done:
        action = agent.act(state)

        state, reward, done, info = env.step(action)

        episodeReward += reward
        print(f"{episodeReward:9.3f}", end="\r")

    agent.training = True
    print(f"Episode Reward: {episodeReward}")

    scheduler.step()