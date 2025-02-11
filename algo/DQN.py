import os.path
import time

import gym
import pandas as pd
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from utils import calculate_huber_loss, create_atari_environment, set_seed, format_timedelta
from GenerateDatasets.Data_Save_Load import EnvDataset


class Qnet(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Qnet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=obs_dim, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=action_dim)
        )
        for layer in self.model[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)
        torch.nn.init.uniform_(self.model[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.model[-1].bias, -3e-3, 3e-3)

    def forward(self, x):
        return self.model(x)


class DQN:
    def __init__(self, args, obs_dim, action_dim):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = args.device
        # Q net.
        self.q_net = Qnet(self.obs_dim, self.action_dim).to(self.device)

        # Target Q net.
        self.target_q_net = Qnet(self.obs_dim, self.action_dim).to(self.device)

        # Adam Optimizer.
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=args.learning_rate)

        # Discount.
        self.discount_rate = args.discount_rate

        # Target update frequency.
        self.target_update_fre = args.target_update_fre
        self.count = 0

    def take_action(self, state):
        """
        :param state:
        :return:
        """
        with torch.no_grad():
            self.q_net.eval()
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            return self.q_net(state).argmax().item()

    def update(self, transition_dict):
        self.q_net.train()

        states = transition_dict['state'].to(self.device)
        actions = transition_dict['action'].to(self.device).view(-1, 1)
        rewards = transition_dict['reward'].to(self.device).view(-1, 1)
        dones = transition_dict['done'].int().to(self.device).view(-1, 1)
        next_states = transition_dict['next_state'].to(self.device)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
            q_targets = rewards + self.discount_rate * next_q_values * (1 - dones)

        td_error = q_targets - q_values

        # Huber Loss.
        # dqn_loss = torch.mean(calculate_huber_loss(td_error, k=1.0))

        # SmoothL1 loss, if k = 1.0, huber loss = SmoothL1 loss
        loss_fn = torch.nn.SmoothL1Loss(reduce=True, size_average=True)
        dqn_loss = loss_fn(q_values, q_targets)

        # MSE loss using Pytorch Loss Function.
        # loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
        # dqn_loss = loss_fn(q_values, q_targets)

        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        # Hard update.
        if self.count % self.target_update_fre == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())
        self.count += 1

        return dqn_loss

    def save(self, file_path):
        """
        :param file_path:
        :return:
        """
        torch.save(self.q_net.state_dict(), file_path + f'/q_net')
        torch.save(self.target_q_net.state_dict(), file_path + f'/target_q_net')
        torch.save(self.optimizer.state_dict(), file_path + f'/optimizer')
        print("==============================================================")
        print("======================= Model Save.... =======================")
        print("==============================================================")

    def load(self, file_path):
        """
        :param file_path:
        :return:
        """
        self.q_net.load_state_dict(torch.load(file_path + f'/q_net'))
        self.target_q_net.load_state_dict(torch.load(file_path + f'/target_q_net'))
        self.optimizer.load_state_dict(torch.load(file_path + f'/optimizer'))
        print("==============================================================")
        print("======================= Model Load.... =======================")
        print("==============================================================")


def train_DQN_agent(args, seed):
    """
    :param seed:
    :param args:
    :return:
    """
    # =============================== Build Env. =============================== #
    env = gym.make(args.game_name)
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # =============================== Set Seed. =============================== #
    set_seed(seed)
    env.seed(seed)

    # ============================== Load Data. =================================== #
    training_data = EnvDataset(args.file_path_data)
    data_loader = DataLoader(dataset=training_data,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=8,
                             pin_memory=True,
                             drop_last=False)

    # ========================= Build DQN Network. ========================= #
    agent = DQN(args, observation_dim, action_dim)
    # if os.path.exists(args.file_path_model + f'/q_net'):
    #     agent.load(args.file_path_model)

    # =========================== Start Training. =========================== #
    start_time = time.time()
    return_list = []

    for ite in range(args.training_iteration):
        for i, sample_batch in enumerate(data_loader):  # 100000 / batch(100)
            loss = agent.update(sample_batch)

            # Evaluate
            if i % 1000 == 0:
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, info = env.step(action)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)

        elapsed_time = time.time() - start_time
        print('Iteration: {0:>4}; Cur_Return: {1:.3f}; Cur_loss: {2:.3f}; {3}h elapsed'.format(
            ite, episode_return, loss,format_timedelta(elapsed_time)))

    # =========================== Save Model. =========================== #
    return_pd = pd.concat([pd.read_csv(args.file_path_model + f'/Eval_Return_pd', index_col=0),
                           pd.DataFrame({'Episode': range(len(return_list)), 'Return': return_list})])
    return_pd.to_csv(args.file_path_model + f'/Eval_Return_pd')

    if np.mean(return_list) >= args.model_save_flag:
        agent.save(args.file_path_model)
        args.model_save_flag = np.mean(return_list)
    return return_pd


def test_DQN_agent(args, seed):
    """
    :param args:
    :param seed:
    :return:
    """
    # =============================== Build Env. =============================== #
    env = gym.make(args.game_name)
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # ========================= Set Seeds. ========================= #
    set_seed(seed)
    env.seed(seed)

    # ========================= Load Model. ========================= #
    agent = DQN(args, observation_dim, action_dim)
    agent.load(args.file_path_model)

    return_list = []
    # =========================== Start Testing. =========================== #
    start_time = time.time()
    for ite in range(args.test_iteration):
        episode_return = 0
        seed = seed + np.random.randint(0, 100)
        env.seed(seed)
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            if args.render:
                env.render()
            state = next_state
            episode_return += reward
        return_list.append(episode_return)

        elapsed_time = time.time() - start_time
        print('Iteration: {0:>4}; Episode_return: {1:.3f}; {2}h elapsed'.format(ite, episode_return,
                                                                                format_timedelta(elapsed_time)))

    return_pd = pd.concat([pd.read_csv(args.file_path_model + f'/Test_Return_pd', index_col=0),
                           pd.DataFrame({'Episode': range(len(return_list)), 'Return': return_list})])
    return_pd.to_csv(args.file_path_model + f'/Test_Return_pd')
    env.close()

    return return_pd
