import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import functools

from env.karman_env import KarmanEnv
from train.config import PPO_CONFIG, CURRICULUM


RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
SENS_DIR    = os.path.join(RESULTS_DIR, 'sensitivity')
os.makedirs(SENS_DIR, exist_ok=True)


SWEEPS = {
    "gamma0": {
        "values"      : [0.8, 1.25, 1.6],
        "param_type"  : "env",
        "xlabel"      : "vortex strength γ₀",
        "description" : "effect of vortex strength on harvesting",
    },
    "ent_coef": {
        "values"      : [0.001, 0.005, 0.01],
        "param_type"  : "ppo",
        "xlabel"      : "entropy coefficient",
        "description" : "effect of exploration on convergence",
    },
    "nu": {
        "values"      : [0.00002, 0.00004, 0.00008],
        "param_type"  : "env",
        "xlabel"      : "viscous decay rate ν",
        "description" : "effect of vortex decay on harvesting",
    },
}

SENS_TIMESTEPS = 200_000
N_EVAL_EPS     = 10


def train_sensitivity_run(param_name, param_val, run_id):
    run_dir = os.path.join(SENS_DIR, param_name, f"val_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    cur     = dict(CURRICULUM['stage3'])
    ppo_cfg = dict(PPO_CONFIG)

    if param_name == 'gamma0':
        cur['gamma0_range'] = (param_val, param_val)
    elif param_name == 'nu':
        cur['nu'] = param_val
    elif param_name == 'ent_coef':
        ppo_cfg['ent_coef'] = param_val

    if param_name == 'nu':
        make_env_fn = functools.partial(
            KarmanEnv, curriculum=cur, nu_override=param_val
        )
    else:
        make_env_fn = functools.partial(KarmanEnv, curriculum=cur)

    env      = make_vec_env(make_env_fn, n_envs=1, seed=42)
    eval_env = make_vec_env(make_env_fn, n_envs=1, seed=0)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = run_dir,
        log_path             = run_dir,
        eval_freq            = 20_000,
        n_eval_episodes      = N_EVAL_EPS,
        deterministic        = True,
        verbose              = 0,
    )

    stage3_model = os.path.join(
        RESULTS_DIR, 'models', 'stage3', 'best_model'
    )
    model = PPO.load(
        stage3_model, env=env,
        tensorboard_log=run_dir,
    )
    # apply ppo overrides after loading
    if param_name == 'ent_coef':
        model.ent_coef = param_val

    model.learn(
        total_timesteps     = SENS_TIMESTEPS,
        callback            = eval_cb,
        progress_bar        = True,
        reset_num_timesteps = True,
    )

    # evaluate
    eval_single = make_env_fn()
    final_rewards, final_thrust, final_steps = [], [], []

    for ep in range(N_EVAL_EPS):
        obs, _ = eval_single.reset(seed=ep+100)
        done   = False
        ep_r   = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = eval_single.step(action)
            done  = terminated or truncated
            ep_r += r
        final_rewards.append(ep_r)
        met = eval_single.get_energy_metrics()
        final_thrust.append(
            met['total_thrust'] / max(eval_single.step_count, 1)
        )
        reached = eval_single.uav.x >= eval_single.x_goal - 0.5
        final_steps.append(
            eval_single.step_count if reached else eval_single.max_steps
        )

    return {
        'param_val'   : param_val,
        'mean_reward' : np.mean(final_rewards),
        'std_reward'  : np.std(final_rewards),
        'mean_thrust' : np.mean(final_thrust),
        'std_thrust'  : np.std(final_thrust),
        'mean_steps'  : np.mean(final_steps),
        'std_steps'   : np.std(final_steps),
    }


def run_all_sweeps():
    all_results = {}

    for param_name, sweep_cfg in SWEEPS.items():
        print(f"\n{'='*50}")
        print(f"sweeping: {param_name} — {sweep_cfg['description']}")
        print(f"values: {sweep_cfg['values']}")
        print(f"{'='*50}")

        param_results = []
        for i, val in enumerate(sweep_cfg['values']):
            print(f"\n  {param_name}={val} "
                  f"({i+1}/{len(sweep_cfg['values'])})")
            result = train_sensitivity_run(param_name, val, i)
            param_results.append(result)
            print(f"  reward={result['mean_reward']:.2f}"
                  f"±{result['std_reward']:.2f}  "
                  f"thrust={result['mean_thrust']:.4f}  "
                  f"steps={result['mean_steps']:.0f}")

        all_results[param_name] = param_results

    np.save(os.path.join(SENS_DIR, 'sensitivity_results.npy'),
            all_results, allow_pickle=True)
    print("\nall sweeps complete.")
    return all_results


def plot_sensitivity(all_results=None, save_dir=None):
    if all_results is None:
        all_results = np.load(
            os.path.join(SENS_DIR, 'sensitivity_results.npy'),
            allow_pickle=True
        ).item()

    plt.style.use('default')
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor('white')
    gs  = gridspec.GridSpec(len(SWEEPS), 3,
                             hspace=0.52, wspace=0.35)

    metrics = [
        ('mean_reward', 'std_reward', 'episode reward'),
        ('mean_thrust', 'std_thrust', 'mean thrust (N)'),
        ('mean_steps',  'std_steps',  'steps to reach goal'),
    ]

    row_colors = ['#006644', '#cc8800', '#cc0044']

    for row, (param_name, sweep_cfg) in enumerate(SWEEPS.items()):
        results = all_results[param_name]
        vals    = [r['param_val'] for r in results]
        color   = row_colors[row]

        for col, (mean_key, std_key, ylabel) in enumerate(metrics):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor('#f7f7f7')
            ax.tick_params(colors='#333333', labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor('#cccccc')
                sp.set_linewidth(0.8)

            means = [r[mean_key] for r in results]
            stds  = [r[std_key]  for r in results]

            ax.errorbar(vals, means, yerr=stds,
                        color=color,
                        marker='o', markersize=7,
                        linewidth=1.8, capsize=5,
                        capthick=1.5,
                        ecolor='#555555',
                        markerfacecolor=color,
                        markeredgecolor='white',
                        markeredgewidth=0.8)
            ax.fill_between(vals,
                            np.array(means) - np.array(stds),
                            np.array(means) + np.array(stds),
                            alpha=0.15, color=color)

            if row == 0:
                ax.set_title(ylabel, fontsize=10,
                             color='#222222', pad=6)
            if col == 0:
                ax.set_ylabel(sweep_cfg['xlabel'],
                              fontsize=8, color='#333333')
            ax.set_xlabel(sweep_cfg['xlabel'],
                          fontsize=7, color='#555555')
            ax.xaxis.set_tick_params(labelcolor='#333333')
            ax.yaxis.set_tick_params(labelcolor='#333333')

    fig.suptitle('Sensitivity Analysis',
                 fontsize=12, color='#222222')

    if save_dir:
        p = os.path.join(save_dir, 'fig6_sensitivity.png')
        plt.savefig(p, dpi=200, bbox_inches='tight',
                    facecolor='white')
        print(f"saved {p}")
    plt.show()


if __name__ == "__main__":
    results = run_all_sweeps()
    plot_sensitivity(results, save_dir=RESULTS_DIR)