import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from env.flow import VortexStreet


def plot_vortex_street(n_steps=400):
    street = VortexStreet(U=1.0, d=0.5, h=1.2, gamma0=1.5,delta=0.08, dt=0.05,x_kill = 30.0,nu = 0.000004,rng=np.random.default_rng(42))

    # evolve the street
    for _ in range(800):
        street.step()

    vorts = street.get_vortex_array()

    # velocity field on a grid
    x_range = np.linspace(0, 28, 120)
    y_range = np.linspace(-3, 3, 40)
    X, Y    = np.meshgrid(x_range, y_range)
    U_field = np.zeros_like(X)
    V_field = np.zeros_like(Y)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            U_field[i,j], V_field[i,j] = street.get_velocity(X[i,j], Y[i,j])

    fig, ax = plt.subplots(figsize=(12, 5))

    # velocity magnitude as background
    speed = np.sqrt(U_field**2 + V_field**2)
    cf = ax.contourf(X, Y, speed, levels=30, cmap='RdBu_r', alpha=0.6)
    plt.colorbar(cf, ax=ax, label='velocity magnitude (m/s)')

    # streamlines
    ax.streamplot(x_range, y_range, U_field, V_field,
                  color='k', linewidth=0.5, density=1.2, arrowsize=0.8)

    # vortex cores — color by sign of circulation
    if len(vorts) > 0:
        pos = vorts[vorts[:, 2] > 0]
        neg = vorts[vorts[:, 2] < 0]
        # replace the scatter lines with:
        if len(pos) > 0:
            sizes = np.abs(pos[:, 2]) / street.gamma0 * 80
            ax.scatter(pos[:, 0], pos[:, 1], c='red',
                       s=sizes, alpha=0.8, zorder=5, label='+Γ')
        if len(neg) > 0:
            sizes = np.abs(neg[:, 2]) / street.gamma0 * 80
            ax.scatter(neg[:, 0], neg[:, 1], c='blue',
                       s=sizes, alpha=0.8, zorder=5, label='-Γ')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Kármán Vortex Street — regularized vortex blob model')
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    vorts = street.get_vortex_array()
    print(f"total vortices: {len(vorts)}")
    if len(vorts) > 0:
        print(f"x range: {vorts[:, 0].min():.1f} to {vorts[:, 0].max():.1f}")
        print(f"gamma range: {vorts[:, 2].min():.3f} to {vorts[:, 2].max():.3f}")
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'vortex_street.png'), dpi=150)
    plt.show()


if __name__ == "__main__":
    plot_vortex_street()