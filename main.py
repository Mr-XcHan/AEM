
import argparse
import torch
import os
import gym
import d4rl_atari
import numpy as np
import pandas as pd

from algo.DQN import train_DQN_agent, test_DQN_agent
from algo.QRDQN import train_QRDQN_agent, test_QRDQN_agent
from algo.Ensemble import train_Ensemble_agent, test_Ensemble_agent
from algo.REM import train_REM_agent, test_REM_agent
from algo.AEM import train_AEM_agent, test_AEM_agent
from GenerateDatasets.Generate_dataset import GenerateDataByDQN, GenerateDataByRandom

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
render = False


env_name = "MountainCar-v0"


if __name__ == '__main__':

    # ================================ Set Game Parameters and Mode. ================================== #
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_name", type=str)  # Set game name for load data.
    parser.add_argument("--game_seeds", default=[0, 1])  # 0, 1
    parser.add_argument("--file_path_data", type=str)
    parser.add_argument("--file_path_model", type=str)

    parser.add_argument("--generating_mode", type=bool, default=False)
    parser.add_argument("--training_mode", type=bool, default=True)
    parser.add_argument("--testing_mode", type=bool, default=False)

    parser.add_argument("--generating_algo", default='DQN')  # Random/DQN
    parser.add_argument("--training_algo", default='AEM')  # DQN/QRDQN/Ensemble/REM/AEM
    parser.add_argument("--testing_algo", default='AEM')  # DQN/QRDQN/Ensemble/REM/AEM

    # ================================ Set Datasets Parameters. ================================== #
    parser.add_argument("--dataset_size", type=int, default=int(2e5))

    # ================================ Set Common Parameters. ================================== #
    parser.add_argument("--render", default=render)  # render.
    parser.add_argument("--device", default=device)  # GPU or CPU.
    parser.add_argument("--model_save_flag", default=-5000)
    parser.add_argument("--batch_size", type=int, default=int(200))
    parser.add_argument("--target_update_fre", type=int, default=int(50))  # Hard update target network.
    parser.add_argument("--QRDQN_num_quantiles", type=int, default=int(32))
    parser.add_argument("--num_heads", type=int, default=int(32))
    parser.add_argument("--num_nets", type=int, default=int(32))
    parser.add_argument("--discount_rate", default=0.99)
    parser.add_argument("--learning_rate", default=1e-3)
    parser.add_argument("--attention_learning_rate", default=3e-3)
    parser.add_argument("--training_iteration", type=int, default=int(100))
    parser.add_argument("--test_iteration", type=int, default=int(100))

    args = parser.parse_args()
    args.game_name = env_name

    # =================================================================================================== #
    # ====================================== Generating Mode. =========================================== #
    # =================================================================================================== #
    if args.generating_mode:
        for game_seed in args.game_seeds:
            print("==========================================================================")
            print("============ Generating Mode; Seed:{}".format(game_seed) + "==============")
            print("==========================================================================")

            # ========================= Set File Path. ========================= #
            args.file_path_data = f"./Datasets" + f"/" + args.game_name + f"/" + args.generating_algo + \
                                  f"/seed_" + str(game_seed)
            if not os.path.exists(args.file_path_data):
                os.makedirs(args.file_path_data)

            # =========================== Algorithm. =========================== #
            if args.generating_algo == 'DQN':
                GenerateDataByDQN(args, game_seed)

            if args.generating_algo == 'Random':
                GenerateDataByRandom(args, game_seed)

    # =================================================================================================== #
    # ======================================= Training Mode. ============================================ #
    # =================================================================================================== #
    if args.training_mode:
        # ========================= Set File Path. ========================= #
        args.file_path_model = f"./SaveOfflineModel" + f"/" + args.game_name + f"/" + args.training_algo
        if not os.path.exists(args.file_path_model):
            os.makedirs(args.file_path_model)

        eval_return_pd = pd.DataFrame()
        eval_return_pd.to_csv(args.file_path_model + f'/Eval_Return_pd')

        kl_pd = pd.DataFrame()
        kl_pd.to_csv(args.file_path_model + f'/KL_pd')

        for game_seed in args.game_seeds:
            print("==========================================================================")
            print("============= Training Mode; Seed:{}".format(game_seed) + "===============")
            print("==========================================================================")

            # ========================= Set File Path. ========================= #
            args.file_path_data = f"./Datasets" + f"/" + args.game_name + f"/" + args.generating_algo + \
                                  f"/seed_" + str(game_seed)
            assert os.path.exists(args.file_path_data), 'The Dataset is not exist.'

            # =========================== Algorithm. =========================== #
            if args.training_algo == 'DQN':
                eval_return_pd = train_DQN_agent(args, game_seed)

            if args.training_algo == 'QRDQN':
                eval_return_pd = train_QRDQN_agent(args, game_seed)

            if args.training_algo == 'Ensemble':
                eval_return_pd = train_Ensemble_agent(args, game_seed)

            if args.training_algo == 'REM':
                eval_return_pd = train_REM_agent(args, game_seed)

            if args.training_algo == 'AEM':
                eval_return_pd = train_AEM_agent(args, game_seed)

        plot_eval(args, eval_return_pd)

    # =================================================================================================== #
    # ======================================== Testing Mode. ============================================ #
    # =================================================================================================== #
    if args.testing_mode:
        average_return_list = []
        expert_return_list = []

        args.file_path_model = f"./SaveOfflineModel" + f"/" + args.game_name + f"/" + args.testing_algo
        assert os.path.exists(args.file_path_model), 'The Model is not exist.'

        test_return_pd = pd.DataFrame()
        test_return_pd.to_csv(args.file_path_model + f'/Test_Return_pd')

        for game_seed in args.game_seeds:
            # Test the algorithms in new seeds.

            print("==========================================================================")
            print("============= Testing Mode; Seed:{}".format(game_seed) + "===============")
            print("==========================================================================")

            # ========================= Set File Path. ========================= #
            args.file_path_data = f"./Datasets" + f"/" + args.game_name + f"/" + args.generating_algo + \
                                  f"/seed_" + str(game_seed)
            assert os.path.exists(args.file_path_data), 'The Dataset is not exist.'

            # =========================== Algorithm. =========================== #
            if args.testing_algo == 'DQN':
                test_return_pd = test_DQN_agent(args, game_seed)

            if args.testing_algo == 'QRDQN':
                test_return_pd = test_QRDQN_agent(args, game_seed)

            if args.testing_algo == 'Ensemble':
                test_return_pd = test_Ensemble_agent(args, game_seed)

            if args.testing_algo == 'REM':
                test_return_pd = test_REM_agent(args, game_seed)

            if args.testing_algo == 'AEM':
                test_return_pd = test_AEM_agent(args, game_seed)

            # ================== Preparing data for plotting. ============================= #
            average_return_list.append(np.load(args.file_path_data + f'/return_per_seed.npy'))
            expert_return_list.append(np.load(args.file_path_data + f'/expert_return_per_seed.npy'))

        average_return = np.mean(average_return_list)
        expert_return = np.mean(expert_return_list)
