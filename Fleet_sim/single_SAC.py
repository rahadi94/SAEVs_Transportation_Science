from Deep_Reinforcement_Learning_Algorithms_with_PyTorch_master.utilities.data_structures.Config import Config
from rl_environment import ChargingHubInvestmentEnv

config = Config()
config.seed = 1
config.num_episodes_to_run = 100
config.file_to_save_data_results = 'result/result'
config.file_to_save_results_graph = 'result/plot'
config.visualise_individual_results = True
config.visualise_overall_agent_results = False
config.randomise_random_seed = False
config.runs_per_agent = 1
config.use_GPU = False
config.evaluation = False
config.learnt_network = False
config.average_score_required_to_win = -5000

config.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 0.005,
        "batch_size": 64,
        "buffer_size": 40000,
        "epsilon": 0.1,
        "epsilon_decay_rate_denominator": 200,
        "discount_rate": 0.99,
        "tau": 0.1,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.4,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 3,
        "linear_hidden_units": [256, 512, 256],
        "final_layer_activation": "None",
        "batch_norm": False,
        "gradient_clipping_norm": 5,
        "HER_sample_proportion": 0.8,
        "clip_rewards": False,
        "learning_iterations": 1
    },
    "Stochastic_Policy_Search_Agents": {
        "policy_network_type": "Linear",
        "noise_scale_start": 1e-2,
        "noise_scale_min": 1e-3,
        "noise_scale_max": 2.0,
        "noise_scale_growth_factor": 2.0,
        "stochastic_action_decision": False,
        "num_policies": 10,
        "episodes_per_policy": 1,
        "num_policies_to_keep": 5,
        "clip_rewards": False
    },
    "Policy_Gradient_Agents": {
        "learning_rate": 0.01,
        "linear_hidden_units": [20],
        "final_layer_activation": "SOFTMAX",
        "learning_iterations_per_round": 7,
        "discount_rate": 0.99,
        "batch_norm": False,
        "clip_epsilon": 0.1,
        "episodes_per_learning_round": 7,
        "normalise_rewards": False,
        "gradient_clipping_norm": 5,
        "mu": 0.0,  # only required for continuous action games
        "theta": 0.0,  # only required for continuous action games
        "sigma": 0.0,  # only required for continuous action games
        "epsilon_decay_rate_denominator": 1,
        "clip_rewards": False
    },

    "Actor_Critic_Agents": {

        "learning_rate": 0.0001,
        "linear_hidden_units": [64],
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": 25.0,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 10.0,
        "normalise_rewards": False,
        "automatically_tune_entropy_hyperparameter": True,
        "add_extra_noise": False,
        "min_steps_before_learning": 64,
        "do_evaluation_iterations": False,
        "clip_rewards": False,

        "Actor": {
            "learning_rate": 0.005,
            "linear_hidden_units": [256,256],
            "final_layer_activation": "Softmax",
            "batch_norm": False,
            "tau": 0.1,
            "gradient_clipping_norm": 5
        },

        "Critic": {
            "learning_rate": 0.005,
            "linear_hidden_units": [512,1024,512],
            "final_layer_activation": "None",
            "batch_norm": False,
            "buffer_size": 100000,
            "tau": 0.1,
            "gradient_clipping_norm": 5
        },

        "batch_size": 32,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.25,  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 100,
        "learning_updates_per_learning_session": 100,
        "HER_sample_proportion": 0.8,
        "exploration_worker_difference": 1.0
    },

    "SNN_HRL": {
        "SKILL_AGENT": {
            "num_skills": 20,
            "regularisation_weight": 1.5,
            "visitations_decay": 0.9999,
            "episodes_for_pretraining": 7,
            "batch_size": 256,
            "learning_rate": 0.001,
            "buffer_size": 40000,
            "linear_hidden_units": [20, 10],
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0, 1],
            "embedding_dimensions": [[300, 10], [20, 6]],
            "batch_norm": False,
            "gradient_clipping_norm": 2,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 500,
            "discount_rate": 0.999,
            "learning_iterations": 1,
            "tau": 0.01,
            "clip_rewards": False
        },

        "MANAGER": {
            "timesteps_before_changing_skill": 6,
            "linear_hidden_units": [10, 5],
            "learning_rate": 0.01,
            "buffer_size": 40000,
            "batch_size": 3,
            "final_layer_activation": "None",
            "columns_of_data_to_be_embedded": [0],
            "embedding_dimensions": [[300, 10]],
            "batch_norm": False,
            "gradient_clipping_norm": 5,
            "update_every_n_steps": 1,
            "epsilon_decay_rate_denominator": 50,
            "discount_rate": 0.99,
            "learning_iterations": 1,
            "tau": 0.01,
            "clip_rewards": False

        }

    }
}
config.hyperparameters = config.hyperparameters['Actor_Critic_Agents']