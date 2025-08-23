import torch

config = {
     "env_id": "HalfCheetah-v5",
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_train_steps": 1_000_000,
    "start_timesteps": 10_000,
    "batch_size": 256,
    "replay_buffer_size": 1_000_000,
    "trans_optimization_epochs": 2,
    "hidden_dim": 512,
    "skill_dim": 16,
    "discrete_skills": True,   # your code zero-centers one-hot
    "lr": 1e-4,
    "phi_lr": 3e-4,
    "tau": 0.005,
    "gamma": 0.99,
    "alpha": 0.2,  # auto-tuned
    "dual_reg": True,
    "dual_lam_init": 30.0,
    "dual_slack": 1e-3,
    "dual_lam_lr": 1e-4,
    "dual_lam_max": 50.0,
    "unit_length_skill": True,
    "skill_length": 25,
    "log_interval": 25_000,
    "eval_interval": 100_000,
    "num_eval_skills": 16,
    "enable_goal_reaching": True,
}
