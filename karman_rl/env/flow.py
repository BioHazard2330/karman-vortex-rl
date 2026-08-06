import numpy as np


def induced_velocity(x, y, vortices, delta=0.1):
    # vortices: Nx4 array — [xv, yv, gamma, age]
    # age column ignored here, only used for decay
    xv  = vortices[:, 0]
    yv  = vortices[:, 1]
    gam = vortices[:, 2]

    dx = x - xv
    dy = y - yv
    r2 = dx**2 + dy**2 + delta**2

    u = np.sum(-gam * dy / (2 * np.pi * r2))

    v = np.sum( gam * dx / (2 * np.pi * r2))

    return u, v


class VortexStreet:
    def __init__(self, U=1.0, d=0.5, h=1.2, gamma0=1.5,noise_scale=1.0,
                 delta=0.08, dt=0.05, x_kill=30.0,
                 nu=0.00004, rng=None):
        """
        U      : freestream velocity (m/s)
        d      : cylinder diameter (m) — controls shedding frequency
        h      : street half-width (m)
        gamma0 : base vortex circulation strength
        delta  : regularization core radius
        dt     : timestep (s)
        x_kill : downstream cutoff (m)
        nu     : kinematic viscosity for vortex decay
        rng    : numpy random Generator for reproducibility
        """
        self.U      = U
        self.d      = d
        self.h      = h
        self.gamma0 = gamma0
        self.delta  = delta
        self.dt     = dt
        self.x_kill = x_kill
        self.nu     = nu
        self.rng    = rng if rng is not None else np.random.default_rng()
        self.noise_scale=noise_scale

        # randomize shedding parameters each episode
        f_base       = 0.2 * U / d   # Strouhal St=0.2
        f_rand       = f_base * self.rng.uniform(0.85, 1.15)
        self.T_shed  = 1.0 / f_rand

        # vortices stored as Nx4: [x, y, gamma, age]
        self.vortices    = np.empty((0, 4))
        self.t           = 0.0
        self.t_last_shed = -self.T_shed  # shed immediately at t=0

    def _shed_pair(self):
        dy_noise = self.rng.uniform(-0.15, 0.15) * self.noise_scale
        dg_noise = self.rng.uniform(-0.20, 0.20) * self.noise_scale
        g = self.gamma0 * (1.0 + dg_noise)

        # small x-stagger — half street width
        # creates visual alternation without binding
        x_stag = self.h * 0.3

        top = [0.0, self.h / 2 + dy_noise, g, 0.0]
        bot = [x_stag, -self.h / 2 + dy_noise, -g, 0.0]

        new_pair = np.array([top, bot])
        if len(self.vortices) == 0:
            self.vortices = new_pair
        else:
            self.vortices = np.vstack([self.vortices, new_pair])

    def _advect_and_decay(self):
        n = len(self.vortices)
        if n == 0:
            return

        xv = self.vortices[:, 0]
        yv = self.vortices[:, 1]
        gam = self.vortices[:, 2]
        age = self.vortices[:, 3]

        # freestream advection only
        xv_new = xv + self.U * self.dt
        yv_new = yv.copy()

        # incremental decay only — not cumulative
        age_new = age + self.dt
        decay = np.exp(-4 * np.pi * self.nu * self.dt / self.delta ** 2)
        gam_new = gam * decay

        alive = (xv_new < self.x_kill) & (np.abs(gam_new) > 0.001)
        self.vortices = np.column_stack([
            xv_new[alive], yv_new[alive],
            gam_new[alive], age_new[alive]
        ])

    def step(self):
        if self.t - self.t_last_shed >= self.T_shed:
            self._shed_pair()
            self.t_last_shed = self.t

        self._advect_and_decay()
        self.t += self.dt

    def get_velocity(self, x, y):
        if len(self.vortices) == 0:
            return self.U, 0.0
        u_ind, v_ind = induced_velocity(x, y, self.vortices, self.delta)
        return self.U + u_ind, v_ind

    def get_vortex_array(self):
        # returns Nx3 [x, y, gamma] for compatibility with env/viz
        if len(self.vortices) == 0:
            return np.empty((0, 3))
        return self.vortices[:, :3]


if __name__ == "__main__":
    # Biot-Savart check
    v_test = np.array([[0.0, 0.0, 1.0, 0.0]])
    u, v   = induced_velocity(1.0, 0.0, v_test, delta=0.01)
    print(f"u={u:.4f}, v={v:.4f}  (expected v~{1/(2*np.pi):.4f})")

    # decay test
    street = VortexStreet(nu=0.00004)
    for _ in range(50):
        street.step()
    vorts = street.get_vortex_array()
    if len(vorts) > 0:
        print(f"\nafter 50 steps:")
        print(f"  n_vortices={len(vorts)}")
        print(f"  max |gamma|={np.abs(vorts[:,2]).max():.4f}")
        print(f"  min |gamma|={np.abs(vorts[:,2]).min():.4f}")
        print(f"  (gamma0={street.gamma0:.2f}, "
              f"so oldest vortex retains "
              f"{np.abs(vorts[:,2]).min()/street.gamma0*100:.1f}%)")

    # randomization test
    s1 = VortexStreet(rng=np.random.default_rng(1))
    s2 = VortexStreet(rng=np.random.default_rng(2))
    for _ in range(100):
        s1.step(); s2.step()
    v1 = s1.get_vortex_array()
    v2 = s2.get_vortex_array()
    print(f"\ns1 vortex count: {len(v1)}, s2 vortex count: {len(v2)}")
    if len(v1) > 0 and len(v2) > 0:
        print(f"s1 gamma range: {v1[:,2].min():.4f} to {v1[:,2].max():.4f}")
        print(f"s2 gamma range: {v2[:,2].min():.4f} to {v2[:,2].max():.4f}")
        print(f"  (ranges should differ due to randomization)")
    street = VortexStreet(nu=0.00004, noise_scale=0.0)
    for i in range(400):
        street.step()
        if i % 50 == 0:
            v = street.get_vortex_array()
            print(f"step {i}: n={len(v)}, "
                  f"x_range=[{v[:, 0].min():.1f},{v[:, 0].max():.1f}] "
                  f"if len(v)>0 else 'empty'")


