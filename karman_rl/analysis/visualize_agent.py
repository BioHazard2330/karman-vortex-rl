import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from stable_baselines3 import PPO
from env.karman_env import KarmanEnv
from env.flow import VortexStreet
from train.config import CURRICULUM

BG_INNER = '#111122'
COL_GRID = '#1a1a33'
COL_W    = '#e8e8f0'
COL_DIM  = '#555577'


def style_white(ax):
    ax.set_facecolor('#f7f7f7')
    ax.tick_params(colors='#333333', labelsize=8)
    ax.xaxis.label.set_color('#333333')
    ax.yaxis.label.set_color('#333333')
    ax.title.set_color('#222222')
    for sp in ax.spines.values():
        sp.set_edgecolor('#cccccc')
        sp.set_linewidth(0.8)


def load_model(stage='stage3'):
    path  = os.path.join(os.path.dirname(__file__), '..', 'results',
                         'models', stage, 'best_model')
    env   = KarmanEnv(curriculum=CURRICULUM[stage])
    model = PPO.load(path, env=env)
    print(f"loaded {stage} best_model")
    return model, env


def run_episode(model, env, seed=None):
    obs, _  = env.reset(seed=seed)
    traj    = dict(x=[], y=[], vx=[], vy=[], heading=[],
                   thrust=[], u_flow=[], v_flow=[],
                   vrel=[], reward=[], dist_nearest=[])
    done    = False
    total_r = 0.0
    goal_step = None

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, _ = env.step(action)
        done    = terminated or truncated
        total_r += r

        uf     = float(obs[5])
        vf     = float(obs[6])
        vrel_x = env.uav.vx - uf
        vrel_y = env.uav.vy - vf
        vrel   = np.sqrt(vrel_x**2 + vrel_y**2)

        vorts = env.street.get_vortex_array()
        dist  = np.sqrt(((vorts[:,0]-env.uav.x)**2 +
                         (vorts[:,1]-env.uav.y)**2)).min() \
                if len(vorts) > 0 else 10.0

        if goal_step is None and env.uav.x >= env.x_goal - 0.5:
            goal_step = len(traj['x'])

        traj['x'].append(env.uav.x)
        traj['y'].append(env.uav.y)
        traj['vx'].append(env.uav.vx)
        traj['vy'].append(env.uav.vy)
        traj['heading'].append(env.uav.heading)
        traj['thrust'].append(float(action[0]))
        traj['u_flow'].append(uf)
        traj['v_flow'].append(vf)
        traj['vrel'].append(vrel)
        traj['reward'].append(r)
        traj['dist_nearest'].append(dist)

    for k in traj:
        traj[k] = np.array(traj[k])

    metrics   = env.get_energy_metrics()
    reached   = env.uav.x >= env.x_goal - 0.5

    print(f"  seed={seed}  steps={len(traj['x'])}  "
          f"reward={total_r:.2f}  "
          f"thrust={traj['thrust'].mean():.4f}  "
          f"vrel={traj['vrel'].mean():.4f}  "
          f"goal={'Y' if reached else 'N'}  "
          f"goal_step={goal_step}")

    return traj, metrics, reached, goal_step


def run_random_episode(env, seed=None):
    env_r  = KarmanEnv(curriculum=env.curriculum)
    obs, _ = env_r.reset(seed=seed)
    traj   = dict(x=[], y=[], thrust=[], heading=[], vrel=[])
    done   = False
    goal_step = None

    while not done:
        action = env_r.action_space.sample()
        obs, _, terminated, truncated, _ = env_r.step(action)
        done = terminated or truncated
        vrel_x = env_r.uav.vx - float(obs[5])
        vrel_y = env_r.uav.vy - float(obs[6])

        if goal_step is None and env_r.uav.x >= env_r.x_goal - 0.5:
            goal_step = len(traj['x'])

        traj['x'].append(env_r.uav.x)
        traj['y'].append(env_r.uav.y)
        traj['thrust'].append(float(action[0]))
        traj['heading'].append(env_r.uav.heading)
        traj['vrel'].append(np.sqrt(vrel_x**2 + vrel_y**2))

    for k in traj:
        traj[k] = np.array(traj[k])

    metrics   = env_r.get_energy_metrics()
    reached   = env_r.uav.x >= env_r.x_goal - 0.5
    return traj, metrics, reached, env_r, goal_step


def get_flow_snapshot(stage='stage3', n_steps=600):
    cur    = CURRICULUM[stage]
    street = VortexStreet(
        U=1.0, d=0.5, h=1.2,
        gamma0=float(np.mean(cur['gamma0_range'])),
        delta=0.08, dt=0.05, x_kill=30.0,
        nu=0.000004, noise_scale=0.0,
        rng=np.random.default_rng(42)
    )
    for _ in range(n_steps):
        street.step()
    return street


def compute_vorticity(street, x_lim=(0,25), y_lim=(-3.5,3.5),
                      nx=100, ny=50):
    xr   = np.linspace(*x_lim, nx)
    yr   = np.linspace(*y_lim, ny)
    X, Y = np.meshgrid(xr, yr)
    U    = np.zeros_like(X)
    V    = np.zeros_like(Y)
    for i in range(ny):
        for j in range(nx):
            U[i,j], V[i,j] = street.get_velocity(X[i,j], Y[i,j])
    dx    = xr[1] - xr[0]
    dy    = yr[1] - yr[0]
    omega = np.gradient(V, dx, axis=1) - np.gradient(U, dy, axis=0)
    return xr, yr, X, Y, U, V, omega


def draw_vorticity_background(ax, street, xr, yr, X, Y,
                               U, V, omega, white_bg=False,
                               stream_density=0.6):
    omega_lim = np.percentile(np.abs(omega), 95)
    ax.contourf(X, Y, omega, levels=80,
                cmap='RdBu_r', alpha=0.65,
                vmin=-omega_lim, vmax=omega_lim, zorder=1)
    stream_col = '#444444' if white_bg else '#9999bb'
    ax.streamplot(xr, yr, U, V,
                  color=stream_col, linewidth=0.3,
                  density=stream_density, arrowsize=0.4,
                  zorder=2)
    vorts = street.get_vortex_array()
    if len(vorts) > 0:
        for grp, col in [(vorts[vorts[:,2]>0], '#cc0000'),
                         (vorts[vorts[:,2]<0], '#0033cc')]:
            if len(grp) == 0:
                continue
            sz = np.abs(grp[:,2]) / street.gamma0 * 100
            for a, s in zip([0.04, 0.12, 0.35, 0.85],
                             [6.0,  3.5,  2.0,  1.0]):
                ax.scatter(grp[:,0], grp[:,1], c=col,
                           s=sz*s, alpha=a, zorder=3,
                           linewidths=0)


def colored_trajectory(ax, x, y, values, cmap='plasma',
                        vmin=0, vmax=1, lw=2.2, zorder=5):
    pts  = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, cmap=cmap, linewidth=lw,
                           zorder=zorder, capstyle='round')
    lc.set_array(values)
    lc.set_clim(vmin, vmax)
    ax.add_collection(lc)
    return lc


def fig1_hunt(model, env, street, seed=2, save_dir=None):
    traj, metrics, reached, goal_step = run_episode(
        model, env, seed=seed
    )

    xr, yr, X, Y, U, V, omega = compute_vorticity(street)

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    draw_vorticity_background(ax, street, xr, yr, X, Y,
                               U, V, omega, white_bg=True,
                               stream_density=0.55)

    lc = colored_trajectory(ax,
                             traj['x'], traj['y'],
                             traj['vrel'],
                             cmap='cool_r',
                             vmin=0, vmax=0.8, lw=2.5)

    ax.scatter(traj['x'][0], traj['y'][0],
               c='#222222', s=80, zorder=8,
               marker='o', linewidths=1.5,
               edgecolors='white')

    ax.axvline(25.0, color='#006633', lw=1.2,
               ls='--', alpha=0.7, zorder=4)

    n = len(traj['x'])
    for k in range(0, n-1, 55):
        hdg = traj['heading'][k]
        ax.annotate('',
            xy=(traj['x'][k] + 0.4*np.cos(hdg),
                traj['y'][k] + 0.4*np.sin(hdg)),
            xytext=(traj['x'][k], traj['y'][k]),
            arrowprops=dict(arrowstyle='->',
                            color='#222222', lw=0.8,
                            alpha=0.7), zorder=7)

    moments  = [int(n*0.06), int(n*0.38), int(n*0.82)]
    m_colors = ['#ff6600', '#006699', '#009933']
    for idx, col in zip(moments, m_colors):
        ax.scatter(traj['x'][idx], traj['y'][idx],
                   c=col, s=55, zorder=9,
                   linewidths=1.2, edgecolors='white')

    cb = plt.colorbar(lc, ax=ax, pad=0.01, shrink=0.8,
                      aspect=25)
    cb.set_label('relative airspeed (m/s)', fontsize=9,
                 color='#333333')
    cb.ax.tick_params(labelsize=8, colors='#333333')

    ax.set_xlim(0, 25)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel('x (m)', fontsize=10)
    ax.set_ylabel('y (m)', fontsize=10)
    ax.set_title('RL Agent — Kármán Vortex Street Energy Harvesting',
                 fontsize=11, pad=8)
    ax.tick_params(colors='#333333')
    for sp in ax.spines.values():
        sp.set_edgecolor('#cccccc')

    plt.tight_layout()
    if save_dir:
        p = os.path.join(save_dir, 'fig1_hunt.png')
        plt.savefig(p, dpi=200, bbox_inches='tight',
                    facecolor='white')
        print(f"saved {p}")
    plt.show()


def fig2_smoking_gun(model, env, seed=2, save_dir=None):
    traj, metrics, reached, _ = run_episode(
        model, env, seed=seed
    )
    steps = np.arange(len(traj['x']))

    plt.style.use('default')
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor('white')
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                             hspace=0.42, wspace=0.35)
    axes = [fig.add_subplot(gs[i//2, i%2]) for i in range(4)]
    for ax in axes:
        style_white(ax)

    # panel 1: thrust vs vortex proximity
    ax  = axes[0]
    ax2 = ax.twinx()
    ax.fill_between(steps, 0, traj['dist_nearest'],
                    alpha=0.12, color='#cc8800')
    ax.plot(steps, traj['dist_nearest'],
            color='#cc8800', lw=1.0, alpha=0.8,
            label='dist to nearest vortex')
    ax.set_ylabel('distance to vortex (m)', color='#cc8800',
                  fontsize=8)
    ax.tick_params(axis='y', colors='#cc8800')
    ax.set_ylim(0, 7)
    ax2.plot(steps, traj['thrust'],
             color='#cc2200', lw=1.5, alpha=0.9,
             label='thrust')
    ax2.set_ylabel('thrust (N)', color='#cc2200', fontsize=8)
    ax2.tick_params(axis='y', colors='#cc2200')
    ax2.set_ylim(0, 1.05)
    ax2.set_facecolor('#f7f7f7')
    ax.set_xlabel('step', fontsize=8)
    ax.set_title('thrust & vortex distance', fontsize=9)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labs1+labs2, fontsize=7,
              loc='upper right',
              facecolor='white', edgecolor='#cccccc')
    for sp in ax.spines.values():
        sp.set_edgecolor('#cccccc')
    for sp in ax2.spines.values():
        sp.set_edgecolor('#cccccc')

    # panel 2: lateral position colored by local flow speed
    ax = axes[1]
    u_speed = np.array(traj['u_flow'])
    sc = ax.scatter(steps, traj['y'],
                    c=u_speed, cmap='plasma',
                    s=4, zorder=3,
                    vmin=u_speed.min(),
                    vmax=u_speed.max())
    ax.axhline(0, color='#999999', ls='--', lw=0.6)
    ax.axhline(3.0,  color='#cc2200', ls=':', lw=0.8, alpha=0.5)
    ax.axhline(-3.0, color='#cc2200', ls=':', lw=0.8, alpha=0.5)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('u_flow (m/s)', fontsize=7, color='#333333')
    cb.ax.tick_params(labelsize=7, colors='#333333')
    ax.set_xlabel('step', fontsize=8)
    ax.set_ylabel('lateral position y (m)', fontsize=8)
    ax.set_title('lateral navigation', fontsize=9)
    ax.set_ylim(-3.8, 3.8)

    # panel 3: cumulative energy budget
    ax = axes[2]
    cum_thrust = np.cumsum(traj['thrust']) * 0.05
    cum_flow   = np.clip(
        np.cumsum(traj['u_flow'] - 1.0) * 0.05, 0, None
    )
    ax.plot(steps, cum_thrust, color='#cc2200', lw=1.5,
            label='thrust energy input')
    ax.fill_between(steps, 0, cum_thrust,
                    alpha=0.12, color='#cc2200')
    ax.plot(steps, cum_flow, color='#006699', lw=1.5,
            label='flow energy surplus')
    ax.fill_between(steps, 0, cum_flow,
                    alpha=0.12, color='#006699')
    ax.set_xlabel('step', fontsize=8)
    ax.set_ylabel('cumulative energy (N·s)', fontsize=8)
    ax.set_title('energy budget', fontsize=9)
    ax.legend(fontsize=7, facecolor='white', edgecolor='#cccccc')

    # panel 4: vrel distribution trained vs random
    ax = axes[3]
    traj_r, _, _, _, _ = run_random_episode(env, seed=seed)
    bins = np.linspace(0, 3.0, 45)
    ax.hist(traj['vrel'], bins=bins, density=True,
            color='#006699', alpha=0.75,
            label=f"trained  μ={traj['vrel'].mean():.3f}",
            edgecolor='none')
    ax.hist(traj_r['vrel'], bins=bins, density=True,
            color='#cc2200', alpha=0.55,
            label=f"random  μ={traj_r['vrel'].mean():.3f}",
            edgecolor='none')
    ax.axvline(traj['vrel'].mean(), color='#006699',
               ls='--', lw=1.5)
    ax.axvline(traj_r['vrel'].mean(), color='#cc2200',
               ls='--', lw=1.5)
    ax.set_xlabel('relative airspeed (m/s)', fontsize=8)
    ax.set_ylabel('probability density', fontsize=8)
    ax.set_title('relative airspeed distribution', fontsize=9)
    ax.legend(fontsize=7, facecolor='white', edgecolor='#cccccc')

    thrust_reduc = (0.5 - traj['thrust'].mean()) / 0.5 * 100
    fig.suptitle(
        f'Strategy Analysis  |  '
        f'mean thrust = {traj["thrust"].mean():.4f} N  |  '
        f'thrust reduction = {thrust_reduc:.1f}%  |  '
        f'mean vrel = {traj["vrel"].mean():.4f} m/s',
        fontsize=10, color='#222222'
    )

    plt.tight_layout()
    if save_dir:
        p = os.path.join(save_dir, 'fig2_smoking_gun.png')
        plt.savefig(p, dpi=200, bbox_inches='tight',
                    facecolor='white')
        print(f"saved {p}")
    plt.show()


def fig3_proof(model, env, n_eps=10, save_dir=None):
    plt.style.use('default')

    print(f"running trained agent ({n_eps} episodes)...")
    t_thrust, t_vrel, t_eta, t_reached, t_steps = \
        [], [], [], [], []
    for seed in range(n_eps):
        traj, met, reached, goal_step = run_episode(
            model, env, seed=seed
        )
        t_thrust.append(traj['thrust'].mean())
        t_vrel.append(traj['vrel'].mean())
        t_eta.append(met['harvest_efficiency'])
        t_reached.append(float(reached))
        t_steps.append(goal_step if goal_step is not None
                        else env.max_steps)

    print(f"running random policy ({n_eps} episodes)...")
    r_thrust, r_vrel, r_eta, r_reached, r_steps = \
        [], [], [], [], []
    for seed in range(n_eps):
        traj_r, met_r, reached_r, _, goal_step_r = \
            run_random_episode(env, seed=seed)
        r_thrust.append(traj_r['thrust'].mean())
        r_vrel.append(traj_r['vrel'].mean())
        r_eta.append(met_r['harvest_efficiency'])
        r_reached.append(float(reached_r))
        r_steps.append(goal_step_r if goal_step_r is not None
                        else env.max_steps)

    fig = plt.figure(figsize=(15, 5.5))
    fig.patch.set_facecolor('white')
    gs   = gridspec.GridSpec(1, 4, figure=fig, wspace=0.38)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    for ax in axes:
        style_white(ax)

    # replace eta with steps-to-goal
    metrics_cfg = [
        (t_thrust,  r_thrust,  'mean thrust (N)',
         'thrust usage'),
        (t_vrel,    r_vrel,    'mean relative airspeed (m/s)',
         'flow exploitation'),
        (t_steps,   r_steps,   'steps to reach goal',
         'navigation efficiency'),
        (t_reached, r_reached, 'goal completion rate',
         'goal completion'),
    ]

    labels = ['trained', 'random']
    colors = ['#006699', '#cc2200']

    for ax, (tv, rv, ylabel, title) in \
            zip(axes, metrics_cfg):
        t_mean, t_std = np.mean(tv), np.std(tv)
        r_mean, r_std = np.mean(rv), np.std(rv)

        bars = ax.bar(labels, [t_mean, r_mean],
                      yerr=[t_std, r_std],
                      color=colors, width=0.5,
                      edgecolor='#cccccc',
                      linewidth=0.8, capsize=5,
                      error_kw={'color':'#555555',
                                'elinewidth':1.2})

        for bar, mean, std in zip(bars,
                                   [t_mean, r_mean],
                                   [t_std,  r_std]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    mean + std + max(r_mean,t_mean)*0.04,
                    f'{mean:.1f}' if title == 'navigation efficiency'
                    else f'{mean:.4f}',
                    ha='center', va='bottom',
                    color='#222222', fontsize=8,
                    fontweight='bold')

        if r_mean > 0 and t_mean > 0:
            if title in ['thrust usage', 'flow exploitation',
                         'navigation efficiency']:
                mult = r_mean / t_mean
                txt  = f'{mult:.1f}× less'
            else:
                mult = t_mean / max(r_mean, 1e-6)
                txt  = f'{mult:.1f}× more'
            ax.text(0.5, 0.94, txt,
                    ha='center', va='top',
                    transform=ax.transAxes,
                    color='#555500', fontsize=8,
                    fontweight='bold')

        np.random.seed(0)
        jitter = np.random.uniform(-0.07, 0.07, n_eps)
        for j, v in enumerate(tv):
            ax.scatter(0+jitter[j], v, c='#003355',
                       s=18, zorder=5, alpha=0.7,
                       linewidths=0)
        for j, v in enumerate(rv):
            ax.scatter(1+jitter[j], v, c='#550011',
                       s=18, zorder=5, alpha=0.7,
                       linewidths=0)

        if title == 'goal completion':
            ax.set_ylim(0, 1.25)

        ax.set_title(title, fontsize=9, pad=6)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticklabels(labels, color='#333333')

    fig.suptitle(
        f'Trained Agent vs Random Policy  —  '
        f'{n_eps} episodes each  |  '
        f'stage 3: full noise + crosswind + random starts',
        fontsize=10, color='#222222'
    )

    plt.tight_layout()
    if save_dir:
        p = os.path.join(save_dir, 'fig3_proof.png')
        plt.savefig(p, dpi=200, bbox_inches='tight',
                    facecolor='white')
        print(f"saved {p}")
    plt.show()


def fig4_curriculum(save_dir=None):
    results_dir = os.path.join(os.path.dirname(__file__),
                               '..', 'results')
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor('white')

    stage_cfg = {
        'stage1': ('Stage 1 — Regular Street',   '#006644'),
        'stage2': ('Stage 2 — Moderate Difficulty', '#cc8800'),
        'stage3': ('Stage 3 — Full Difficulty',   '#cc0044'),
    }

    for ax, (stage, (title, color)) in \
            zip(axes, stage_cfg.items()):
        style_white(ax)
        path = os.path.join(results_dir, 'logs',
                            stage, 'evaluations.npz')
        if not os.path.exists(path):
            ax.text(0.5, 0.5, 'no data',
                    ha='center', va='center',
                    color='#999999',
                    transform=ax.transAxes)
            continue

        data  = np.load(path)
        ts    = data['timesteps']
        means = data['results'].mean(axis=1)
        stds  = data['results'].std(axis=1)

        ax.plot(ts, means, color=color, lw=2.0, zorder=3)
        ax.fill_between(ts, means-stds, means+stds,
                        alpha=0.18, color=color, zorder=2)
        ax.axhline(0, color='#aaaaaa', ls='--',
                   lw=0.7, alpha=0.6)

        # convergence marker — skip for stage3
        if stage != 'stage3':
            threshold = means[-1] * 0.95
            conv_idxs = np.where(means >= threshold)[0]
            if len(conv_idxs) > 0:
                ci = conv_idxs[0]
                ax.axvline(ts[ci], color=color,
                           ls=':', lw=1.0, alpha=0.6)

        ax.text(0.97, 0.04,
                f'final: {means[-1]:.1f}±{stds[-1]:.1f}',
                ha='right', va='bottom',
                transform=ax.transAxes,
                color=color, fontsize=8,
                fontweight='bold')

        ax.set_title(title, fontsize=9, pad=6)
        ax.set_xlabel('timesteps', fontsize=8)
        ax.set_ylabel('eval reward', fontsize=8)

    fig.suptitle('Curriculum Learning Progression',
                 fontsize=11, color='#222222')
    plt.tight_layout()

    if save_dir:
        p = os.path.join(save_dir, 'fig4_curriculum.png')
        plt.savefig(p, dpi=200, bbox_inches='tight',
                    facecolor='white')
        print(f"saved {p}")
    plt.show()


def fig5_storyboard(model, env, street, seed=2,
                    save_dir=None):
    traj_t, _, reached_t, _ = run_episode(
        model, env, seed=seed
    )
    traj_r, _, reached_r, env_r, _ = run_random_episode(
        env, seed=seed
    )

    xr, yr, X, Y, U, V, omega = compute_vorticity(street)
    omega_lim = np.percentile(np.abs(omega), 95)

    def moments(traj):
        n = len(traj['x'])
        return [int(n*0.05), int(n*0.45), n-1]

    mt = moments(traj_t)
    mr = moments(traj_r)

    plt.style.use('default')
    fig = plt.figure(figsize=(16, 7.5))
    fig.patch.set_facecolor('white')
    gs  = gridspec.GridSpec(
        2, 4, figure=fig,
        width_ratios=[1,1,1,0.06],
        hspace=0.32, wspace=0.18
    )

    # third column label differs by row
    col_labels_trained = ['departure', 'vortex lock-on', 'arrival']
    col_labels_random  = ['departure', 'vortex lock-on', 'outcome']

    row_info = [
        (traj_t, mt, 'trained agent', '#004488',
         reached_t, col_labels_trained),
        (traj_r, mr, 'random policy', '#aa1100',
         reached_r, col_labels_random),
    ]

    for row, (traj, midxs, rlabel, rcol,
              reached, clabels) in enumerate(row_info):
        for col, (idx, clabel) in \
                enumerate(zip(midxs, clabels)):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor('#f5f5f5')

            ax.contourf(X, Y, omega, levels=40,
                        cmap='RdBu_r', alpha=0.55,
                        vmin=-omega_lim, vmax=omega_lim,
                        zorder=1)
            ax.streamplot(xr, yr, U, V,
                          color='#555555', linewidth=0.28,
                          density=0.45, arrowsize=0.35,
                          zorder=2)

            vorts = street.get_vortex_array()
            if len(vorts) > 0:
                for grp, gc in [
                        (vorts[vorts[:,2]>0], '#cc0000'),
                        (vorts[vorts[:,2]<0], '#0033cc')]:
                    if len(grp) == 0:
                        continue
                    sz = np.abs(grp[:,2])/street.gamma0*70
                    ax.scatter(grp[:,0], grp[:,1], c=gc,
                               s=sz, alpha=0.75, zorder=3,
                               linewidths=0)

            if idx > 1:
                colored_trajectory(
                    ax,
                    traj['x'][:idx+1],
                    traj['y'][:idx+1],
                    traj['thrust'][:idx+1],
                    cmap='YlOrRd',
                    vmin=0, vmax=1.0,
                    lw=2.0, zorder=4
                )

            ax.scatter(traj['x'][idx], traj['y'][idx],
                       c=rcol, s=100, zorder=6,
                       marker='D', linewidths=1.2,
                       edgecolors='white')

            hdg = traj['heading'][idx]
            ax.annotate('',
                xy=(traj['x'][idx]+0.45*np.cos(hdg),
                    traj['y'][idx]+0.45*np.sin(hdg)),
                xytext=(traj['x'][idx], traj['y'][idx]),
                arrowprops=dict(arrowstyle='->',
                                color=rcol, lw=1.4),
                zorder=7)

            # thrust gauge
            gx, gy, gw = 0.4, -3.1, 4.0
            tv = traj['thrust'][idx]
            ax.barh(gy, gw, height=0.28,
                    color='#dddddd', zorder=5, left=gx)
            ax.barh(gy, gw*tv, height=0.28, zorder=6,
                    color='#dd3300' if tv > 0.15
                          else '#008833',
                    left=gx)
            ax.text(gx+gw/2, gy+0.35,
                    f'T = {tv:.4f} N',
                    ha='center', va='center',
                    fontsize=6, fontweight='bold',
                    color='white', zorder=8)

            ax.axvline(25.0, color='#006633',
                       lw=0.9, ls='--', alpha=0.6, zorder=4)

            ax.set_xlim(0, 25)
            ax.set_ylim(-3.8, 3.8)

            if row == 0:
                ax.set_title(clabel, fontsize=9,
                             fontweight='bold',
                             color='#222222', pad=5)

            if col == 0:
                ax.set_ylabel(rlabel, fontsize=9,
                              fontweight='bold',
                              color=rcol, labelpad=6)
            else:
                ax.set_yticklabels([])

            if row == 1:
                ax.set_xlabel('x (m)', fontsize=8,
                              color='#444444')
            else:
                ax.set_xticklabels([])

            ax.tick_params(colors='#666666', labelsize=6.5)
            for sp in ax.spines.values():
                sp.set_edgecolor('#cccccc')
                sp.set_linewidth(0.7)

            ax.text(0.97, 0.96, f'step {idx}',
                    ha='right', va='top',
                    transform=ax.transAxes,
                    fontsize=6.5, color='#666666')

    ax_cb = fig.add_subplot(gs[:, 3])
    sm    = plt.cm.ScalarMappable(
        cmap='RdBu_r',
        norm=mcolors.Normalize(-omega_lim, omega_lim)
    )
    sm.set_array([])
    cb = plt.colorbar(sm, cax=ax_cb)
    cb.set_label('vorticity ω (s⁻¹)', fontsize=8,
                 color='#444444')
    cb.ax.tick_params(labelsize=7, colors='#444444')

    fig.suptitle(
        'Episode Storyboard — Trained Agent vs Random Policy',
        fontsize=11, color='#222222', y=1.02
    )

    plt.tight_layout()
    if save_dir:
        p = os.path.join(save_dir, 'fig5_storyboard.png')
        plt.savefig(p, dpi=200, bbox_inches='tight',
                    facecolor='white')
        print(f"saved {p}")
    plt.show()


if __name__ == "__main__":
    save_dir = os.path.join(os.path.dirname(__file__),
                            '..', 'results')
    os.makedirs(save_dir, exist_ok=True)

    model, env = load_model(stage='stage3')
    street     = get_flow_snapshot(stage='stage3', n_steps=600)

    fig1_hunt(model, env, street, seed=2, save_dir=save_dir)
    fig2_smoking_gun(model, env, seed=2, save_dir=save_dir)
    fig3_proof(model, env, n_eps=20, save_dir=save_dir)
    fig4_curriculum(save_dir=save_dir)
    fig5_storyboard(model, env, street, seed=2, save_dir=save_dir)