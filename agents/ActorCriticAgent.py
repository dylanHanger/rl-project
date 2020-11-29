import copy
from collections import namedtuple

import nle.nethack as nh
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from . import AbstractAgent

DEVICE = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")

Action = namedtuple('Action', ['log_prob', 'value'])

class Crop(nn.Module):
    """Class that crops input image"""
    def __init__(self, input_height, input_width, target_height, target_width):
        super().__init__()
        self.height = input_height
        self.width = input_width
        self.target_height = target_height
        self.target_width = target_width

        x = ((2/(self.width-1)) * torch.arange(-self.target_width//2,target_width//2))[None, :].expand(self.target_height, -1)
        y = ((2/(self.height-1)) * torch.arange(-self.target_height//2,target_height//2))[:, None].expand(-1,self.target_width)

        self.register_buffer("x_grid", x, persistent=False)
        self.register_buffer("y_grid", y, persistent=False)

    def forward(self, inputs, coords):
        """Calculates crop of inputs centered around x,y coords"""

        # Glyphs is HxW, so we add a "channel" layer
        inputs = inputs.unsqueeze(1).float()

        x = coords[:, 0]
        y = coords[:, 1]

        # Get the shift needed to center x,y
        x_shift = 2 / (self.width - 1) * (x.float() - self.width // 2)
        y_shift = 2 / (self.height - 1) * (y.float() - self.height // 2)

        grid = torch.stack(
            [
                self.x_grid[None, :, :] + x_shift[:, None, None],
                self.y_grid[None, :, :] + y_shift[:, None, None]
            ],
            dim=3
        )

        return (
            # FIXME: Pretty sure the rounding can be replaced with nearest neighbour interpolation, no?
            torch.round(F.grid_sample(inputs, grid, align_corners=True)) # We only want integers, so round.
            .squeeze(1) # Get rid of the 'channel' dimension
            .long() # cast to int
        )

# TODO: Crop layer
# TODO: RND
class ActorCriticModel(nn.Module):
    def __init__(self, num_actions, num_stats, world_shape):
        super().__init__()
        self.num_actions = num_actions
        self.num_stats = num_stats
        self.world_shape = world_shape

        ############################
        ##  Network Architecture  ##
        ############################

        # 1. Embeddings for the glyphs
        k = 32 # The embedding dimension
        self.embedding = nn.Embedding(nh.MAX_GLYPH, k)

        # 2: World encoding
        self.world_features = nn.Sequential(
            nn.Conv2d(k, 16, kernel_size=3, padding=1, stride=1),
            nn.ELU(),

            nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
            nn.ELU(),
            
            nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
            nn.ELU(),
            
            nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
            nn.ELU(),
            
            nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=1),
            nn.ELU(),

            nn.Flatten()
        )

        # 2: Ego-centric view. The features are identical to the world features
        self.crop = Crop(*world_shape, 9, 9)
        self.crop_features = nn.Sequential(
            nn.Conv2d(k, 16, kernel_size=3, padding=1, stride=1),
            nn.ELU(),

            nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
            nn.ELU(),
            
            nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
            nn.ELU(),
            
            nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
            nn.ELU(),
            
            nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=1),
            nn.ELU(),

            nn.Flatten()
        )

        # 3: Stats encoding
        self.stats_features = nn.Sequential(
            nn.Linear(self.num_stats-2, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # 4: Get latent representation
        self.feature_size = self._getFeatureSize()
        self.encoding = nn.Sequential(
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # 5: Recurrent policy
        self.lstm = nn.LSTM(128, 128)

        # TODO: More complex heads?
        # 6.1: Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, self.num_actions),
            nn.Softmax(dim=1)
        )

        # 6.2: Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            
            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 1),
        )

    def _getFeatureSize(self):
        world = torch.zeros((1, *self.world_shape)).long()
        stats = torch.zeros((1, self.num_stats)).float()

        crop = self.crop(world, stats[:, :2])
        crop = self.embedding(crop).transpose(1,3)
        world = self.embedding(world).transpose(1,3)

        wx = self.world_features(world)
        sx = self.stats_features(stats[:, 2:])
        cx = self.crop_features(crop)

        x = torch.cat([sx, wx, cx], dim=1)
        return x.view(-1).shape[0]

    def _getUsefulStats(self, observation):
        observation = copy.deepcopy(observation)
        blstats = observation["blstats"]
        x, y = blstats[0], blstats[1]
        hp, mHp = blstats[10], blstats[11]
        energy = blstats[14]
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
        state = {k: torch.tensor(v).to(DEVICE).unsqueeze(0)
                 for k, v in state.items()}

        ###########################
        ##  Normal Forward Pass  ##
        ###########################

        # 1: Separate the real info
        world = state["glyphs"].long()
        stats = state["blstats"].float()

        crop = self.crop(world, stats[:, :2])

        # 2: Compute embedding vectors
        crop = self.embedding(crop).transpose(1,3)
        world = self.embedding(world).transpose(1,3)

        # 3: Get features
        wx = self.world_features(world)
        sx = self.stats_features(stats[:, 2:])
        cx = self.crop_features(crop)


        # 4: Get low dimensional representation of the state
        x = torch.cat([sx, wx, cx], dim=1)
        x = self.encoding(x)

        # 5: Compute policy tings
        x, self.hidden = self.lstm(x.view(1,1,-1), self.hidden)
        x = x.flatten(0,1)

        # 6.1: Compute action probabilites
        probs = self.actor_head(x)

        # 6.2: Compute state value
        value = self.critic_head(x)

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

        # Initialise a model
        self.model = ActorCriticModel(self.action_space.n,
                                      5,
                                      observation_space["glyphs"].shape).to(DEVICE)
        self.reset()
        self.model.train(training)
        if not training:
            # Try to load pretrained weights
            try:
                # NOTE: Change this when I change the model structure
                self.model.load_state_dict(torch.load(
                    "/root/nethack/models/v4latest.pt"))
            except FileNotFoundError:
                print("Weights file not found.")

    def reset(self):
        del self.rewards[:]
        del self.actions[:]

        # initialize the hidden state.
        self.model.hidden = (torch.randn(1, 1, 128).to(DEVICE),
                       torch.randn(1, 1, 128).to(DEVICE))

    def act(self, observation):
        with torch.set_grad_enabled(self.training):
            ############################
            ##  Get the actual move   ##
            ##        to make         ##
            ############################

            # 1: Consult our model
            probs, value = self.model(observation)
            if self.training:
                # 2: Sample our action based on that
                m = Categorical(probs)
                action = m.sample()

                # 3: Save our action/prob pair for future loss calculations
                self.actions.append(Action(m.log_prob(action), value.squeeze(0)))
                return action.item()
            else:
                # Don't sample when training
                return probs.argmax()
