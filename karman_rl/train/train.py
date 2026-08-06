import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import functools
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from env.karman_env import KarmanEnv
from config import PPO_CONFIG, CURRICULUM

results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
models_dir  = os.path.join(results_dir, 'models')
logs_dir    = os.path.join(results_dir, 'logs')
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir,   exist_ok=True)


def make_env(curriculum):
    return functools.partial(KarmanEnv, curriculum=curriculum)


def train():
    model = None

    for stage_name, stage_cfg in CURRICULUM.items():
        print(f"\n{'='*50}")
        print(f"starting {stage_name}: {stage_cfg['description']}")
        print(f"timesteps: {stage_cfg['total_timesteps']}")
        print(f"{'='*50}")

        stage_models_dir = os.path.join(models_dir, stage_name)
        stage_logs_dir   = os.path.join(logs_dir,   stage_name)
        os.makedirs(stage_models_dir, exist_ok=True)
        os.makedirs(stage_logs_dir,   exist_ok=True)

        env      = make_vec_env(
            make_env(stage_cfg), n_envs=1, seed=42
        )
        eval_env = make_vec_env(
            make_env(stage_cfg), n_envs=1, seed=0
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path = stage_models_dir,
            log_path             = stage_logs_dir,
            eval_freq            = 10_000,
            n_eval_episodes      = 5,
            deterministic        = True,
            verbose              = 1
        )

        if model is None:
            # stage 1 — fresh model
            model = PPO(
                "MlpPolicy", env,
                **PPO_CONFIG,
                tensorboard_log = stage_logs_dir,
                seed            = 42
            )
        else:
            # subsequent stages — load best from previous stage
            prev_stage = list(CURRICULUM.keys())[
                list(CURRICULUM.keys()).index(stage_name) - 1
            ]
            prev_best = os.path.join(
                models_dir, prev_stage, 'best_model'
            )
            print(f"loading weights from {prev_best}")
            model.set_env(env)
            model = PPO.load(
                prev_best, env=env,
                tensorboard_log = stage_logs_dir,
            )
            if stage_name == 'stage3':
                model.learning_rate = 1e-4
                print("stage3: learning rate set to 1e-4")

        model.learn(
            total_timesteps = stage_cfg['total_timesteps'],
            callback        = eval_callback,
            progress_bar    = True,
            reset_num_timesteps = True
        )

        model.save(os.path.join(stage_models_dir, 'final_model'))
        print(f"{stage_name} complete.")

    print("\nall stages complete.")


if __name__ == "__main__":
    train()