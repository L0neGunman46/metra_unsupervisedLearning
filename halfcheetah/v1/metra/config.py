import torch

config = {
    "env_id": "HalfCheetah-v5",
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Training
    "num_train_steps": 1_000_000,
    "start_timesteps": 10_000,
    "batch_size": 256,
    "replay_buffer_size": 500_000,
    "trans_optimization_epochs": 32,

    # Network
    "hidden_dim": 512,
    "skill_dim": 16,                      # More capacity for complex env
    "discrete_skills": True,

    # SAC
    "lr": 1e-4,                          # Actor/Critic
    "phi_lr": 1e-3,                      # Faster φ(s) learning
    "tau": 0.005,
    "gamma": 0.99,
    "alpha": 0.2,

    # METRA-specific
    "dual_reg": True,
    "dual_lam_init": 1.0,                 # Very low initial λ
    "dual_slack": 1e-2,                   # Very loose constraint
    "dual_lam_lr": 5e-4,                  # Very slow λ growth
    "dual_lam_max": 300.0,                 # Low cap to prevent runaway
    "unit_length_skill": False,

    # HRL
    "skill_length": 25,

    # Logging/Eval
    "log_interval": 25_000,
    "eval_interval": 100_000,
    "num_eval_skills": 8,

    # Goal reaching
    "enable_goal_reaching": True,
}
