# Reinforcement Learning Project - The NetHack Learning Environment

This project is based on making a model capable of playing the [NLE provided by Facebook](https://github.com/facebookresearch/nle).

This repo specifically contains the code for our Actor-Critic implementation. If you're looking for our DQN implementation, you can find it [here](https://github.com/dkpoult/rl-project).

## Breakdown

The root contains the `train.py` and `evaluation.py` files, which are used to train a model, and test a model, respectively. 
The Actor-Critic algorithm can be found in `train.py`.
`pretrain.py` is used to attempt to train the agent to predict a human players moves.

The `agents` folder contains the primary file, `ActorCriticAgent.py`.
This file contains both the model definition as well as the actual agent itself.

The `pickles` folder contains pickled replays of games played by a human player

## Running the code

To train an Actor Critic agent yourself, just run the `train.py` script in the root directory.
To then evaluate the model (its no better than random), run `evaluation.py`.

### Note
The `MyAgent.py` required by the leaderboard is a copy of `ActorCriticAgent.py` with some small changes regarding file structure.