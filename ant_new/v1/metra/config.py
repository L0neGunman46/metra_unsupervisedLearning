import torch

# Tuned configuration for Ant-v5 in METRA
config = {
    "env_id": "Ant-v5",
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Training parameters
    "num_train_steps": 1_000_000,       # Full training run
    "start_timesteps": 10_000,          # Random actions before training
    "batch_size": 256,
    "replay_buffer_size": 500_000,
    "trans_optimization_epochs": 32,    # Updates per environment step

    # Network architecture
    "hidden_dim": 512,
    "skill_dim": 2,                      # More capacity for Ant
    "discrete_skills": False,

    # SAC hyperparameters
    "lr": 1e-4,                          # Actor/Critic learning rate
    "phi_lr": 3e-4,                      # Representation learning rate (faster)
    "tau": 0.005,
    "gamma": 0.99,
    "alpha": 0.2,

    # METRA-specific hyperparameters
    "dual_reg": True,
    "dual_lam_init": 1.0,               # Lower initial λ to avoid early collapse
    "dual_slack": 1e-2,                  # Looser constraint for high-dim env
    "dual_lam_lr": 1e-4,                 # Slow λ updates
    "dual_lam_max": 50.0,                # Cap λ to prevent runaway
    "unit_length_skill": True,

    # HRL parameters
    "skill_length": 25,

    # Logging and evaluation
    "log_interval": 25_000,
    "eval_interval": 100_000,
    "num_eval_skills": 8,

    # Goal reaching
    "enable_goal_reaching": True,
}