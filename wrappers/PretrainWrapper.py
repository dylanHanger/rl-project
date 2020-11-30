import glob
from collections import namedtuple

import torch.nn as nn
import torch
import pickle
import gym
import nle
import numpy as np
import torch
from torch.distributions import Categorical

Action = namedtuple('Action', ['log_prob', 'value'])


class PretrainEnv(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.gameIndex = -1

        self.fileNames = glob.glob("./pickles/*")
        self.games = []
        for fileName in self.fileNames:
            pickleFile = open(fileName, 'rb')
            print(fileName)
            self.games.append(pickle.load(pickleFile))
        self.actionProbs = []
        self.trueActions = []
        self.predictedRewards = []
        self.predictedActions = []

    def reset(self):
        self.gameIndex += 1
        self.currentStep = 0
        if (self.gameIndex >= len(self.games)):
            return None
        self.currentGame = self.games[self.gameIndex]
        observations = self.currentGame['Observations'][0]
        observations['blstats'] += 1  # not great but still
        return observations

    def step(self, probs, value):

        self.trueActions.append(self.getAction())
        self.actionProbs.append(probs)
        self.currentStep += 1
        m = Categorical(probs)
        action = m.sample()
        self.predictedActions.append(
            Action(m.log_prob(action), value.squeeze(0)))

        observations = self.currentGame['Observations'][self.currentStep]
        new_state, reward, done, _ = observations
        if (self.currentStep+1 >= len(self.currentGame['Observations'])):
            done = True
            observations = (new_state, reward, done, _)
        #loss = nn.NLLLoss()
        #target = torch.tensor([self.currentGame['Actions'][self.currentStep-1]])

        return observations

    def getAction(self):
        return self.currentGame['Actions'][self.currentStep]

    def resetHistory(self):
        del self.trueActions[:]
        del self.actionProbs[:]
        del self.predictedRewards[:]
        del self.predictedActions[:]
