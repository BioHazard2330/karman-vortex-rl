import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env.karman_env import KarmanEnv
from train.config import CURRICULUM

RESULTS_DIR   = os.path.join(os.path.dirname(__file__), '..', 'results')
N_SEEDS_PER_Y = 8
Y_STARTS      = np.linspace(-1.5, 1.5, 7)


def load_model(stage='stage3'):
    path  = os.path.join(RESULTS_DIR, 'models', stage, 'best_model')
    env   = KarmanEnv(curriculum=CURRICULUM[stage])
    model = PPO.load(path, env=env)
    return model


def run_fixed_start(model, y0, seed):
    cur = dict(CURRICULUM['stage3'])
    cur['y_start_range'] = (y0, y0)   
    env = KarmanEnv(curriculum=cur)
    obs, _ = env.reset(seed=seed)
    done, total_r, thrusts = False, 0.0, []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_r += r
        thrusts.append(float(action[0]))
    reached = env.uav.x >= env.x_goal - 0.5
    return total_r, np.mean(thrusts), reached


def run_sweep(model):
    results = {}
    for y0 in Y_STARTS:
        rewards, thrusts, flags = [], [], []
        for seed in range(N_SEEDS_PER_Y):
            r, t, reached = run_fixed_start(
                model, y0, seed=seed * 100 + int((y0 + 2) * 10)
            )

            rewards.append(r); thrusts.append(t); flags.append(float(reached))
        results[y0] = {
            'reward_mean': np.mean(rewards), 'reward_std': np.std(rewards),
            'completion_rate': np.mean(flags),
        }
        print(f"y0={y0:+.2f}  reward={results[y0]['reward_mean']:.2f}"
              f"±{results[y0]['reward_std']:.2f}  "
              f"completion={results[y0]['completion_rate']*100:.0f}%")
    return results


def plot_generalization(results, save_dir=None):
    plt.style.use('default')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor('white')

    y0s    = sorted(results.keys())
    r_mean = [results[y]['reward_mean'] for y in y0s]
    r_std  = [results[y]['reward_std']  for y in y0s]
    comp   = [results[y]['completion_rate'] * 100 for y in y0s]

    ax = axes[0]
    ax.errorbar(y0s, r_mean, yerr=r_std, marker='o', color='#006699',
                capsize=4, linewidth=1.8)
    ax.set_xlabel('starting lateral position y₀ (m)')
    ax.set_ylabel('episode reward')
    ax.set_title('reward vs. start position')
    ax.axvline(0, color='#999999', ls='--', lw=0.8)

    ax = axes[1]
    ax.bar([f'{y:+.1f}' for y in y0s], comp, color='#cc8800',
           edgecolor='#333333')
    ax.set_xlabel('starting lateral position y₀ (m)')
    ax.set_ylabel('goal completion rate (%)')
    ax.set_title('completion vs. start position')
    ax.set_ylim(0, 105)

    fig.suptitle(f'Generalization Across Start Positions '
                 f'({N_SEEDS_PER_Y} seeds per position)')
    plt.tight_layout()
    if save_dir:
        p = os.path.join(save_dir, 'fig7_generalization.png')
        plt.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"saved {p}")
    plt.show()


if __name__ == "__main__":
    model   = load_model('stage3')
    results = run_sweep(model)
    plot_generalization(results, save_dir=RESULTS_DIR)
