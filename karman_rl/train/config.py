CURRICULUM = {
    "stage1": {
        "total_timesteps" : 300_000,
        "wind_amp_range"  : (0.0,  0.0),
        "gamma0_range"    : (1.8,  2.0),
        "noise_scale"     : 0.0,
        "y_start_range"   : (0.0,  0.0),
        "description"     : "strong regular street, centered start",
    },
    "stage2": {
        "total_timesteps" : 400_000,
        "wind_amp_range"  : (0.0,  0.08),
        "gamma0_range"    : (1.4,  1.8),
        "noise_scale"     : 0.5,
        "y_start_range"   : (-0.8, 0.8),
        "description"     : "moderate noise, light wind",
    },
    "stage3": {
        "total_timesteps" : 1_000_000,
        "learning_rate"   : 1e-4,
        "wind_amp_range"  : (0.05, 0.15),
        "gamma0_range"    : (1.0,  1.5),
        "noise_scale"     : 1.0,
        "y_start_range"   : (-1.5, 1.5),
        "description"     : "full difficulty",
    },
}

PPO_CONFIG = {
    "learning_rate" : 3e-4,
    "n_steps"       : 4096,
    "batch_size"    : 128,
    "n_epochs"      : 10,
    "gamma"         : 0.99,
    "gae_lambda"    : 0.95,
    "clip_range"    : 0.2,
    "ent_coef"      : 0.001,
    "verbose"       : 1,
    "policy_kwargs" : dict(net_arch=[256, 256, 128]),
}

TRAIN_CONFIG = {
    "total_timesteps" : 400_000,
    "seed"            : 42,
    "log_interval"    : 1,
}