import time
import os
import gym
import numpy
import pandas as pd
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


from torch.utils.data import DataLoader

from utils import init_weights, set_seed, format_timedelta
from GenerateDatasets.Data_Save_Load import EnvDataset


class Qnet_MultiNet(nn.Module):
    def __init__(self, obs_dim, action_dim, num_nets):
        super().__init__()

        self.models = nn.ModuleList([
            nn.Sequential(*[
                nn.Linear(in_features=obs_dim, out_features=256),
                # nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=128),
                # nn.BatchNorm1d(128),
                nn.ReLU(),
                # nn.Linear(in_features=128, out_features=128),
                # # nn.BatchNorm1d(128),
                # nn.ELU(),
                nn.Linear(in_features=128, out_features=64),
                # nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(in_features=64, out_features=action_dim)
            ]) for _ in range(num_nets)
        ])
        for model in self.models:
            # init as in the EDAC paper
            for layer in model[::2]:
                torch.nn.init.constant_(layer.bias, 0.1)
            torch.nn.init.uniform_(model[-1].weight, -3e-3, 3e-3)
            torch.nn.init.uniform_(model[-1].bias, -3e-3, 3e-3)

    def forward(self, state):
        return torch.stack([model(state) for model in self.models])


class Attention_net(nn.Module):
    def __init__(self, obs_dim, action_dim, num_nets):
        super(Attention_net, self).__init__()
        input_dim = obs_dim * 2 + 1 + num_nets * 2   # 1 represents the executing action.
        self.fc1 = nn.Linear(in_features=input_dim, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=num_nets)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)


class AEM:
    def __init__(self, args, obs_dim, action_dim):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = args.device
        # self.num_heads = args.num_heads
        self.num_nets = args.num_nets

        # Q net.
        self.q_net = Qnet_MultiNet(self.obs_dim, self.action_dim, self.num_nets).to(self.device)
        # Target Q net.
        self.target_q_net = Qnet_MultiNet(self.obs_dim, self.action_dim, self.num_nets).to(self.device)

        # Attention net
        self.attention_net = Attention_net(self.obs_dim, self.action_dim, self.num_nets).to(self.device)
        # self.attention_net.apply(init_weights)

        # Adam Optimizer.
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=args.learning_rate)
        self.attention_optimizer = torch.optim.Adam(self.attention_net.parameters(), lr=args.attention_learning_rate)

        # Discount.
        self.discount_rate = args.discount_rate

        # Target update frequency.
        self.target_update_fre = args.target_update_fre
        self.count = 0

    def take_first_action(self, state):
        """
        :param state:
        :return:
        """
        with torch.no_grad():
            self.q_net.eval()
            state = torch.tensor(state).float().unsqueeze(0).to(self.device)
            q_value = torch.mean(self.q_net(state).squeeze(), dim=0)  # [action_dim]
            action = torch.argmax(q_value)

            return action.cpu().detach().numpy()

    def take_action(self, state, next_state, action):
        """
        :param state:
        :return:
        """
        with torch.no_grad():
            self.q_net.eval()
            self.attention_net.eval()

            state = torch.tensor(state).float().unsqueeze(0).to(self.device)
            next_state = torch.tensor(next_state).float().unsqueeze(0).to(self.device)
            action = torch.tensor(action).float().unsqueeze(0).to(self.device)

            next_max_q_values, max_next_actions = torch.max(self.q_net(next_state), dim=2)  # [1, num_nets]

            attention_input = torch.concat((state, action.unsqueeze(-1), next_state, max_next_actions.T,
                                            next_max_q_values.T), dim=1).to(self.device)  #, next_max_q_values.T

            alpha = self.attention_net(attention_input).permute(1, 0)  # [num_nets, 1]
            att_q_value = torch.sum(alpha * self.q_net(state).squeeze(), dim=0)  # [action_dim]
            action = torch.argmax(att_q_value)
            return action.cpu().detach().numpy()

    def update(self, transition_dict):
        self.q_net.train()
        self.attention_net.train()

        states = transition_dict['state'].to(self.device)
        actions = transition_dict['action'].to(self.device)
        rewards = transition_dict['reward'].to(self.device)
        dones = transition_dict['done'].to(self.device)
        next_state = transition_dict['next_state'].to(self.device)

        # Get Q(s,a).
        q_values = self.q_net(states).gather(2, actions.unsqueeze(-1).expand(self.num_nets, -1, -1)).squeeze()
        # [num_nets, batch]

        # Get Q-target.
        with torch.no_grad():
            next_q_values = self.target_q_net(next_state)  # [num_nets, batch, action_dim]
            next_q_values, max_next_actions = torch.max(next_q_values, dim=2)  # [num_nets, batch]
            next_max_q_values_clone = next_q_values.clone().permute(1, 0)  # [batch, num_nets]
            dones = dones.expand(self.num_nets, -1)
            next_q_values[dones] = 0
        q_targets = (next_q_values * self.discount_rate) + rewards  # [num_nets, batch]

        # Get Attention.
        with torch.no_grad():
            q_targets_clone = q_targets.clone().permute(1, 0)  # [batch, num_nets]
            attention_input = torch.concat((states, actions.unsqueeze(-1), next_state, max_next_actions.T,
                                            next_max_q_values_clone), dim=1).to(self.device)  #  # next_max_q_values_clone # [batch, obs_dim*2+1+num_nets*2]
        alpha = self.attention_net(attention_input).permute(1, 0)  # [num_nets, batch]

        q_values = torch.sum(alpha * q_values, dim=0)  # [batch]
        # q_values = torch.mean(q_values, dim=0)
        q_targets = torch.sum(alpha * q_targets, dim=0)  # [batch]


        td_error = torch.abs(q_targets_clone - q_values.unsqueeze(-1).expand(-1, self.num_nets))
        td_error_softmax = F.softmax(td_error, dim=1)


        kl_fn = nn.KLDivLoss(reduction="batchmean")
        # kl = kl_fn((alpha.permute(1, 0) + 1e-6).log(), td_error_softmax)
        kl = kl_fn((td_error_softmax + 1e-6).log(), alpha.permute(1, 0))

        # TODO CrossEntropy FN.
        # ce_fn = nn.CrossEntropyLoss(reduction='mean')
        # ce = ce_fn(alpha.permute(1, 0), td_error_softmax)

        # SmoothL1 loss, if k = 1.0, huber loss = SmoothL1 loss
        loss_fn = torch.nn.SmoothL1Loss(reduce=True, size_average=True)
        loss = loss_fn(q_values, q_targets)

        # loss = loss - 1e-4 * kl

        # Back Propagation.
        self.q_optimizer.zero_grad()
        self.attention_optimizer.zero_grad()
        loss.backward()
        # for param in self.q_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.q_optimizer.step()
        # self.attention_optimizer.step()
        if self.count % (1) == 0:
            # self.attention_optimizer.zero_grad()
            self.attention_optimizer.step()

        # Hard update.
        if self.count % self.target_update_fre == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())
        self.count += 1

        return loss, kl.cpu().detach().numpy(), td_error_softmax.cpu().detach().numpy(), alpha.T.cpu().detach().numpy()

    def save(self, file_path):
        """
        :param file_path:
        :return:
        """
        torch.save(self.q_net.state_dict(), file_path + f'/q_net')
        torch.save(self.target_q_net.state_dict(), file_path + f'/target_q_net')
        torch.save(self.attention_net.state_dict(), file_path + f'/attention_net')
        torch.save(self.q_optimizer.state_dict(), file_path + f'/q_optimizer')
        torch.save(self.attention_optimizer.state_dict(), file_path + f'/attention_optimizer')
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
        self.attention_net.load_state_dict(torch.load(file_path + f'/attention_net'))
        self.q_optimizer.load_state_dict(torch.load(file_path + f'/q_optimizer'))
        self.attention_optimizer.load_state_dict(torch.load(file_path + f'/attention_optimizer'))
        print("==============================================================")
        print("======================= Model Load.... =======================")
        print("==============================================================")


def train_AEM_agent(args, seed):
    """
    :param seed:
    :param args:
    :return:
    """
    if not os.path.exists(args.file_path_model + f'/td_alpha'):
        os.makedirs(args.file_path_model + f'/td_alpha')

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
    agent = AEM(args, observation_dim, action_dim)
    # if os.path.exists(args.file_path_model + f'/q_net'):
    #     agent.load(args.file_path_model)

    # =========================== Start Training. =========================== #
    start_time = time.time()
    return_list = []
    kl_list = []
    success_rate = 0
    for ite in range(args.training_iteration):
        td_error_softmax_mean = np.zeros((args.batch_size, args.num_nets))
        alpha_mean = np.zeros((args.batch_size, args.num_nets))
        count = 0
        for i, sample_batch in enumerate(data_loader):
            loss, kl, td_error_softmax, alpha = agent.update(sample_batch)
            if sample_batch['state'].size()[0] == args.batch_size:
                td_error_softmax_mean += td_error_softmax
                alpha_mean += alpha
                count += 1

            # Evaluate
            if i % 1000 == 0:
                episode_return = 0
                state = env.reset()[:observation_dim]
                action = agent.take_first_action(state)
                done = False
                while not done:
                    next_state, reward, done, info = env.step(action)
                    action = agent.take_action(state, next_state[:observation_dim], action)
                    state = next_state[:observation_dim]
                    episode_return += reward
                return_list.append(episode_return)


            if ite == 0 and i == 0:
                np.save(args.file_path_model + f'/td_alpha' + f'/td_error_softmax_start', td_error_softmax)
                np.save(args.file_path_model + f'/td_alpha' + f'/alpha_start', alpha)

        td_error_softmax_mean /= count
        alpha_mean /= count

        np.save(args.file_path_model + f'/td_alpha' + f'/td_error_softmax_' + str(ite), td_error_softmax_mean)
        np.save(args.file_path_model + f'/td_alpha' + f'/alpha_' + str(ite), alpha_mean)

        kl_list.append(kl)
        elapsed_time = time.time() - start_time
        print('Iteration: {0:>4}; Cur_Return: {1:.3f}; Cur_loss: {2:.3f}; KL: {3:.3f} ,{4}h elapsed'.format(
            ite, episode_return, loss, kl, format_timedelta(elapsed_time)))

        # =========================== Save Model. =========================== #
    success_rate /= len(return_list)
    np.save(args.file_path_model + f'/Eval_success_rate_' + str(seed), success_rate)

    return_pd = pd.concat([pd.read_csv(args.file_path_model + f'/Eval_Return_pd', index_col=0),
                           pd.DataFrame({'Episode': range(len(return_list)), 'Return': return_list})])
    return_pd.to_csv(args.file_path_model + f'/Eval_Return_pd')

    kl_pd = pd.concat([pd.read_csv(args.file_path_model + f'/KL_pd', index_col=0),
                       pd.DataFrame({'Episode': range(len(kl_list)), 'KL': kl_list})])
    kl_pd.to_csv(args.file_path_model + f'/KL_pd')

    # =========================== Save Model. =========================== #
    if np.mean(return_list) >= args.model_save_flag:
        agent.save(args.file_path_model)
        args.model_save_flag = np.mean(return_list)
        print("Model_save_flag:{}".format(args.model_save_flag))

    env.close()
    return return_pd


def test_AEM_agent(args, seed):
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
    agent = AEM(args, observation_dim, action_dim)
    agent.load(args.file_path_model)

    return_list = []
    success_rate = 0

    # =========================== Start Testing. =========================== #
    start_time = time.time()
    for ite in range(args.test_iteration):
        episode_return = 0
        seed = seed + np.random.randint(0, 100)
        env.seed(seed)
        state = env.reset()[:observation_dim]
        action = agent.take_first_action(state)
        done = False
        while not done:
            next_state, reward, done, info = env.step(action)
            action = agent.take_action(state, next_state[:observation_dim], action)
            # env.render(mode="top_down")
            # if args.render:
            #     env.render()
            state = next_state[:observation_dim]
            episode_return += reward
        return_list.append(episode_return)

        elapsed_time = time.time() - start_time
        print('Iteration: {0:>4}; Episode_return: {1:.3f}; {2}h elapsed; seed: {3}'.format(ite, episode_return,
                                                                                format_timedelta(elapsed_time), config['start_seed']))
        env.close()

    success_rate /= len(return_list)
    np.save(args.file_path_model + f'/Test_success_rate_' + str(seed), success_rate)

    return_pd = pd.concat([pd.read_csv(args.file_path_model + f'/Test_Return_pd', index_col=0),
                           pd.DataFrame({'Episode': range(len(return_list)), 'Return': return_list})])
    return_pd.to_csv(args.file_path_model + f'/Test_Return_pd')


    return return_pd


