import copy
from collections import namedtuple

import nle.nethack as nh
import torch
from torch import nn
from torch.distributions import Categorical

from . import AbstractAgent

DEVICE = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")

Action = namedtuple('Action', ['log_prob', 'value'])


class ActorCriticModel(nn.Module):
    def __init__(self, num_actions, num_stats, world_shape):
        super().__init__()
        self.num_actions = num_actions
        self.num_stats = num_stats
        self.world_shape = world_shape

        ############################
        ##  Network Architecture  ##
        ############################
        # TODO: LSTM, Egocentric crop

        # 1: The map goes through a CNN
        self.world_features = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=128,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=10,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Flatten()
        )

        # 2: The stats go through a FCNN
        self.stats_features = nn.Sequential(
            nn.Linear(in_features=self.num_stats, out_features=256),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True),
        )

        # 3: The output of previous networks then go into another FCNN
        self.feature_size = self._getFeatureSize()
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 512),
        )

        # 4: This output then goes into the Actor and Critic heads
        # 4.1: Actor
        self.actor_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, self.num_actions),
            nn.Softmax(dim=1)
        )
        # 4.2: Critic
        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 1),
        )

    def _getFeatureSize(self):
        world = torch.zeros((1, 2, *self.world_shape)).float()
        stats = torch.zeros((1, self.num_stats)).float()

        wx = self.world_features(world)
        sx = self.stats_features(stats)

        x = torch.cat([sx, wx], dim=1)
        return x.view(-1).shape[0]

    def _getUsefulStats(self, observation):
        observation = copy.deepcopy(observation)
        blstats = observation["blstats"]
        x, y = blstats[0], blstats[1]
        hp, mHp = blstats[10], blstats[11]
        # Max energy can be 0, and I dont wanna deal with division by 0
        energy, mEnergy = blstats[14], blstats[15]
        hunger = blstats[21]

        observation["blstats"] = [x, y, float(
            hp)/float(mHp), energy, hunger]
        return observation

    def forward(self, state):
        ###########################
        ##  This should be done  ##
        ##      by wrappers      ##
        ###########################

        # 1: Remove some information from the observation
        state = self._getUsefulStats(state)

        # 2: Convert the observation into tensors that live on the correct device
        state = {k: torch.tensor(v).float().to(
            DEVICE).unsqueeze(0) for k, v in state.items()}

        ###########################
        ##  Normal Forward Pass  ##
        ###########################

        # 1: Separate the real info
        world = torch.stack([state["chars"], state["colors"]], dim=1)
        stats = state["blstats"]

        # 2: Compute the world and stats features
        wx = self.world_features(world)
        sx = self.stats_features(stats)

        # 3: Concatenate the features and pass through MLP
        x = torch.cat([sx, wx], dim=1)
        x = self.fc(x)

        # 4.1: Actor computes probabilities of each action
        probs = self.actor_head(x)

        # 4.2: Critic computes values of the current state
        value = self.value_head(x)

        return probs, value


class ActorCriticAgent(AbstractAgent):
    @property
    def device(self):
        return DEVICE

    def __init__(self, observation_space, action_space, training=False):
        self.observation_space = observation_space
        self.action_space = action_space
        self.training = training

        self.rewards = []
        self.actions = []

        # We only allow a subset of actions, so it isn't overwhelming to learn
        # TODO: Intercardinal directions? (NE/SW/etc)
        # TODO: Convert between reduced and actual action space
        self.reduced_action_space = [
            nh.CompassCardinalDirection.N,
            nh.CompassCardinalDirection.E,
            nh.CompassCardinalDirection.S,
            nh.CompassCardinalDirection.W,
            nh.MiscDirection.UP,
            nh.MiscDirection.DOWN,
            nh.MiscDirection.WAIT,
            nh.Command.KICK,
            nh.Command.EAT,
            nh.Command.SEARCH
        ]
        self.reduced_action_space = action_space

        # Initialise a model
        self.model = ActorCriticModel(self.reduced_action_space.n,
                                      5,
                                      observation_space["glyphs"].shape).to(DEVICE)
        self.model.train(training)
        if not training:
            # Try to load pretrained weights
            try:
                # NOTE: Change this when I change the model structure
                self.model.load_state_dict(torch.load(
                    "/root/nethack/models/v0latest.pt"))
                print("Weights successfully loaded.")
            except FileNotFoundError:
                print("Weights file not found.")

    def resetHistory(self):
        del self.rewards[:]
        del self.actions[:]

    def act(self, observation):
        with torch.set_grad_enabled(self.training):
            ############################
            ##  Get the actual move   ##
            ##        to make         ##
            ############################

            # 1: Consult our model
            probs, value = self.model(observation)

            # 2: Sample our action based on that
            m = Categorical(probs)
            action = m.sample()

            # 3: Save our action/prob pair for future loss calculations
            self.actions.append(Action(m.log_prob(action), value.squeeze(0)))

            # TODO: The action is in our reduced action space, get the 'true' action index
            return action.item()
