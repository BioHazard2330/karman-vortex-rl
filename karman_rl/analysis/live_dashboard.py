import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection

from env.karman_env import KarmanEnv
from train.config import CURRICULUM

RESULTS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'results')
STAGES       = ['stage1', 'stage2', 'stage3']
REFRESH_SEC  = 20
RANDOM_THRUST_BASELINE = 0.5   # expected mean thrust for random policy


def load_eval_log(stage):
    path = os.path.join(RESULTS_DIR, 'logs', stage, 'evaluations.npz')
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path)
        return {
            'timesteps': data['timesteps'],
            'results'  : data['results'],
        }
    except Exception:
        return None


def get_active_stage():
    active = None
    for stage in STAGES:
        path = os.path.join(
            RESULTS_DIR, 'models', stage, 'best_model.zip'
        )
        if os.path.exists(path):
            active = stage
    return active


def get_model_mtime(stage):
    path = os.path.join(
        RESULTS_DIR, 'models', stage, 'best_model.zip'
    )
    try:
        return os.path.getmtime(path)
    except Exception:
        return None


def load_best_model(stage):
    try:
        from stable_baselines3 import PPO
        path = os.path.join(RESULTS_DIR, 'models', stage, 'best_model')
        env  = KarmanEnv(curriculum=CURRICULUM[stage])
        return PPO.load(path, env=env)
    except Exception:
        return None


def run_episode(model, stage, max_steps=600, seed = 0):
    env    = KarmanEnv(curriculum=CURRICULUM[stage])
    obs, _ = env.reset(seed=seed)

    traj = dict(x=[], y=[], thrust=[], heading=[],
                u_flow=[], v_flow=[], vrel=[])
    done  = False
    steps = 0

    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        u_f = float(obs[5])
        v_f = float(obs[6])
        vrel_x = env.uav.vx - u_f
        vrel_y = env.uav.vy - v_f

        traj['x'].append(env.uav.x)
        traj['y'].append(env.uav.y)
        traj['thrust'].append(float(action[0]))
        traj['heading'].append(env.uav.heading)
        traj['u_flow'].append(u_f)
        traj['v_flow'].append(v_f)
        traj['vrel'].append(np.sqrt(vrel_x**2 + vrel_y**2))
        steps += 1

    for k in traj:
        traj[k] = np.array(traj[k])

    metrics = env.get_energy_metrics()
    reached_goal = env.uav.x >= env.x_goal - 0.5

    return traj, metrics, env.street, reached_goal


def get_flow_field(street):
    x_range = np.linspace(0, 25, 40)
    y_range = np.linspace(-3.2, 3.2, 20)
    X, Y    = np.meshgrid(x_range, y_range)
    Uf      = np.zeros_like(X)
    Vf      = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Uf[i,j], Vf[i,j] = street.get_velocity(X[i,j], Y[i,j])
    return x_range, y_range, X, Y, Uf, Vf


def style_ax(ax):
    ax.set_facecolor('#111122')
    ax.tick_params(colors='white', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')


def draw_learning_curves(axes, colors):
    for i, stage in enumerate(STAGES):
        ax = axes[i]
        ax.cla()
        style_ax(ax)
        data = load_eval_log(stage)
        if data is None:
            ax.text(0.5, 0.5, f'{stage}\nwaiting...',
                    ha='center', va='center',
                    color='#555577', fontsize=9,
                    transform=ax.transAxes)
        else:
            ts    = data['timesteps']
            means = data['results'].mean(axis=1)
            stds  = data['results'].std(axis=1)
            ax.plot(ts, means, color=colors[i], lw=1.5)
            ax.fill_between(ts, means-stds, means+stds,
                            alpha=0.2, color=colors[i])
            ax.axhline(means[-1], color='white',
                       ls='--', lw=0.7, alpha=0.5)
            ax.text(0.97, 0.05,
                    f'{means[-1]:.1f}±{stds[-1]:.1f}',
                    ha='right', va='bottom',
                    color=colors[i], fontsize=8,
                    transform=ax.transAxes)
        ax.set_title(stage, color='white', fontsize=8)
        ax.set_xlabel('timesteps', color='white', fontsize=7)
        ax.set_ylabel('eval reward', color='white', fontsize=7)


def draw_panels(ax_flow, ax_thrust, ax_metrics, ax_info,
                traj, metrics, street, stage, reached_goal):

    # ── flow field ────────────────────────────────────────────────
    ax_flow.cla()
    style_ax(ax_flow)

    x_range, y_range, X, Y, Uf, Vf = get_flow_field(street)
    speed = np.sqrt(Uf**2 + Vf**2)
    ax_flow.contourf(X, Y, speed, levels=25,
                     cmap='magma', alpha=0.5, zorder=1)
    ax_flow.streamplot(x_range, y_range, Uf, Vf,
                       color='#aaaacc', linewidth=0.4,
                       density=0.5, arrowsize=0.5, zorder=2)

    # vortex cores with glow
    vorts = street.get_vortex_array()
    if len(vorts) > 0:
        for data_v, col in [(vorts[vorts[:,2]>0], '#ff4444'),
                            (vorts[vorts[:,2]<0], '#4488ff')]:
            if len(data_v) > 0:
                sizes = np.abs(data_v[:,2]) / street.gamma0 * 50
                for a, s in zip([0.05, 0.2, 0.8], [5.0, 3.0, 1.0]):
                    ax_flow.scatter(data_v[:,0], data_v[:,1],
                                    c=col, s=sizes*s,
                                    alpha=a, zorder=3, linewidths=0)

    # trajectory colored by relative airspeed
    if len(traj['x']) > 1:
        pts  = np.array([traj['x'], traj['y']]).T.reshape(-1,1,2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc   = LineCollection(segs, cmap='plasma',
                               linewidth=2.0, zorder=5)
        lc.set_array(traj['thrust'])
        lc.set_clim(0, 1.0)
        ax_flow.add_collection(lc)

        # start marker
        ax_flow.scatter([traj['x'][0]], [traj['y'][0]],
                        c='#ffffff', s=40, zorder=7,
                        linewidths=0, marker='o')

        # drone arrow at current position
        x_now = traj['x'][-1]
        y_now = traj['y'][-1]
        hdg   = traj['heading'][-1]
        arrow_col = '#00ff00' if reached_goal else '#ffff00'
        ax_flow.annotate('',
            xy=(x_now + 0.6*np.cos(hdg),
                y_now + 0.6*np.sin(hdg)),
            xytext=(x_now, y_now),
            arrowprops=dict(arrowstyle='->',
                            color=arrow_col, lw=2.0), zorder=8)
        ax_flow.scatter([x_now], [y_now], c=arrow_col,
                        s=60, zorder=9, linewidths=0)

    # goal line
    ax_flow.axvline(25.0, color='#00ff88', lw=1.0,
                    ls='--', alpha=0.5, zorder=4)

    ax_flow.set_xlim(0, 25)
    ax_flow.set_ylim(-3.5, 3.5)
    ax_flow.set_title(
        f'agent in flow — {stage}  '
        f'{"✓ GOAL REACHED" if reached_goal else ""}',
        color='#00ff88' if reached_goal else 'white', fontsize=8
    )
    ax_flow.set_xlabel('x (m)', color='white', fontsize=7)
    ax_flow.set_ylabel('y (m)', color='white', fontsize=7)

    # ── thrust panel ──────────────────────────────────────────────
    ax_thrust.cla()
    style_ax(ax_thrust)
    steps = np.arange(len(traj['thrust']))
    ax_thrust.plot(steps, traj['thrust'],
                   color='#ff9933', lw=0.8, alpha=0.9)
    ax_thrust.plot(steps, traj['vrel'],
                   color='#66ccff', lw=0.8, alpha=0.7,
                   label=f"vrel μ={traj['vrel'].mean():.3f}")
    ax_thrust.axhline(traj['thrust'].mean(), color='#ff9933',
                      ls='--', lw=1.0,
                      label=f"thrust μ={traj['thrust'].mean():.4f}")
    ax_thrust.set_ylim(0, max(1.05, traj['vrel'].max()*1.1))
    ax_thrust.set_title('thrust & relative airspeed',
                        color='white', fontsize=8)
    ax_thrust.set_xlabel('step', color='white', fontsize=7)
    ax_thrust.legend(fontsize=6, loc='upper right')

    # ── metrics panel ─────────────────────────────────────────────
    ax_metrics.cla()
    style_ax(ax_metrics)

    # show thrust reduction vs random as the key metric
    thrust_reduction = (RANDOM_THRUST_BASELINE -
                        traj['thrust'].mean()) / RANDOM_THRUST_BASELINE
    thrust_reduction = np.clip(thrust_reduction, 0, 1)

    mean_vrel  = traj['vrel'].mean()
    P_flow     = metrics['total_P_flow']
    P_input    = metrics['total_P_input']

    # bar chart: three meaningful metrics
    metric_vals   = [thrust_reduction,
                     min(1.0, P_flow / (P_input + 1e-3)),
                     min(1.0, 1.0 - mean_vrel)]
    metric_labels = ['thrust\nreduction', 'P_flow\nfraction', 'flow\nalignment']
    metric_colors = ['#00ffcc', '#ffcc00', '#ff6699']

    bars = ax_metrics.bar(metric_labels, metric_vals,
                          color=metric_colors,
                          edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, metric_vals):
        ax_metrics.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f'{val:.3f}',
            ha='center', va='bottom',
            color='white', fontsize=8
        )
    ax_metrics.set_ylim(0, 1.2)
    ax_metrics.set_title('performance metrics',
                         color='white', fontsize=8)
    ax_metrics.tick_params(axis='x', colors='white', labelsize=7)

    # ── info panel ────────────────────────────────────────────────
    ax_info.cla()
    ax_info.axis('off')
    goal_str = '✓ YES' if reached_goal else '✗ NO'
    info_text = (
        f"stage:         {stage}\n"
        f"steps:         {len(traj['x'])}\n"
        f"goal reached:  {goal_str}\n\n"
        f"mean thrust:   {traj['thrust'].mean():.4f} N\n"
        f"mean vrel:     {traj['vrel'].mean():.4f} m/s\n"
        f"thrust reduc:  {thrust_reduction*100:.1f}%\n\n"
        f"P_flow:        {P_flow:.3f}\n"
        f"P_input:       {P_input:.3f}\n"
    )
    ax_info.text(0.05, 0.95, info_text,
                 ha='left', va='top',
                 color='white', fontsize=8,
                 transform=ax_info.transAxes,
                 fontfamily='monospace')


def make_dashboard():
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#0d0d1a')

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.55, wspace=0.4,
                           height_ratios=[1, 1.4, 1])

    curve_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_status  = fig.add_subplot(gs[0, 3])
    ax_status.axis('off')

    ax_flow    = fig.add_subplot(gs[1, :])
    ax_thrust  = fig.add_subplot(gs[2, 0])
    ax_metrics = fig.add_subplot(gs[2, 1])
    ax_info    = fig.add_subplot(gs[2, 2])
    ax_info.axis('off')
    ax_log     = fig.add_subplot(gs[2, 3])
    ax_log.axis('off')

    colors = ['#00ffcc', '#ffcc00', '#ff6699']

    plt.ion()
    plt.show()

    cached_model = None
    cached_stage = None
    cached_mtime = None
    iteration    = 0
    log_lines    = []

    while True:
        iteration   += 1
        active_stage = get_active_stage()
        mtime        = get_model_mtime(active_stage) \
                       if active_stage else None

        # reload model if stage changed or best_model updated
        if active_stage and (active_stage != cached_stage or
                             mtime != cached_mtime):
            print(f"loading {active_stage} best_model...")
            cached_model = load_best_model(active_stage)
            cached_stage = active_stage
            cached_mtime = mtime
            log_lines.append(
                f"[{iteration:03d}] loaded {active_stage}"
            )
            log_lines = log_lines[-8:]   # keep last 8 lines

        # draw learning curves
        draw_learning_curves(curve_axes, colors)

        # status panel
        ax_status.cla()
        ax_status.axis('off')
        active_data  = load_eval_log(active_stage) \
                       if active_stage else None
        latest_reward = (active_data['results'].mean(axis=1)[-1]
                         if active_data is not None else 0)
        status_text = (
            f"active: {active_stage or 'waiting'}\n"
            f"latest eval: {latest_reward:.1f}\n"
            f"refresh: #{iteration}"
        )
        ax_status.text(0.1, 0.8, status_text,
                       ha='left', va='top',
                       color='white', fontsize=8,
                       transform=ax_status.transAxes,
                       fontfamily='monospace')

        if cached_model is not None:
            traj, metrics, street, reached = run_episode(
                cached_model, cached_stage , max_steps=600, seed = iteration % 10
            )
            draw_panels(
                ax_flow, ax_thrust, ax_metrics, ax_info,
                traj, metrics, street, cached_stage, reached
            )
            log_lines.append(
                f"[{iteration:03d}] "
                f"thrust={traj['thrust'].mean():.4f} "
                f"goal={'Y' if reached else 'N'}"
            )
            log_lines = log_lines[-8:]
        else:
            ax_flow.cla()
            style_ax(ax_flow)
            ax_flow.text(0.5, 0.5,
                         'waiting for first checkpoint...',
                         ha='center', va='center',
                         color='#555577', fontsize=11,
                         transform=ax_flow.transAxes)

        # log panel
        ax_log.cla()
        ax_log.axis('off')
        ax_log.text(0.05, 0.95, '\n'.join(log_lines),
                    ha='left', va='top',
                    color='#aaaaaa', fontsize=7,
                    transform=ax_log.transAxes,
                    fontfamily='monospace')

        fig.suptitle(
            f'Live Training Dashboard — Kármán Vortex RL  '
            f'|  refresh #{iteration}  |  '
            f'active: {active_stage or "waiting"}',
            color='white', fontsize=11
        )

        plt.pause(0.1)
        plt.draw()

        for remaining in range(REFRESH_SEC, 0, -1):
            fig.suptitle(
                f'Live Training Dashboard — Kármán Vortex RL  '
                f'|  refresh #{iteration}  |  '
                f'next in {remaining}s  |  '
                f'active: {active_stage or "waiting"}',
                color='white', fontsize=10
            )
            plt.pause(1.0)
            if not plt.fignum_exists(fig.number):
                print("dashboard closed.")
                return


if __name__ == "__main__":
    print(f"watching: {RESULTS_DIR}")
    print(f"refreshing every {REFRESH_SEC}s")
    print("close window to stop\n")
    make_dashboard()