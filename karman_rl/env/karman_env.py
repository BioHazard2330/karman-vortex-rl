import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env.flow import VortexStreet
from env.dynamics import UAV


class KarmanEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None, curriculum=None,
                 nu_override=None):
        super().__init__()

        self.U         = 1.0
        self.x_goal    = 25.0
        self.y_bounds  = 3.0
        self.dt        = 0.05
        self.max_steps = 600
        self.nu_override = nu_override

        self.curriculum = curriculum if curriculum is not None else {
            "wind_amp_range" : (0.05, 0.15),
            "gamma0_range"   : (1.0,  1.5),
            "noise_scale"    : 1.0,
            "y_start_range"  : (-1.5, 1.5),
        }

        # placeholders
        self.wind_amp   = 0.0
        self.wind_freq  = 0.0
        self.wind_phase = 0.0
        self.gamma0     = 1.5

        # action: [thrust, heading_rate]
        # heading_rate in rad/s, clipped by UAV hdg_rate_max
        self.action_space = spaces.Box(
            low  = np.array([0.0, -np.pi/4], dtype=np.float32),
            high = np.array([1.0,  np.pi/4], dtype=np.float32)
        )

        obs_high = np.array([
            30., 4., 5., 5., np.pi,  # x, y, vx, vy, heading
            4.,  4., 2.,             # u_flow, v_flow, dvdy
            10., 4., 1.,             # vortex 1: dx, dy, sign
            10., 4., 1.,             # vortex 2
            10., 4., 1.,             # vortex 3
            0.2                      # v_wind
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low  = -obs_high,
            high =  obs_high,
            dtype = np.float32
        )

        self.render_mode   = render_mode
        self.street        = None
        self.uav           = None
        self.step_count    = 0
        self.total_thrust  = 0.0
        self.total_P_flow  = 0.0
        self.total_P_input = 0.0

    def _get_obs(self):
        u_flow, v_flow = self.street.get_velocity(self.uav.x, self.uav.y)

        _, v_above = self.street.get_velocity(self.uav.x, self.uav.y + 0.1)
        _, v_below = self.street.get_velocity(self.uav.x, self.uav.y - 0.1)
        dvdy = (v_above - v_below) / 0.2

        vorts   = self.street.get_vortex_array()
        vx_near = np.zeros(6)
        gm_near = np.zeros(3)

        if len(vorts) > 0:
            ahead = vorts[:, 0] > self.uav.x
            if ahead.any():
                vorts_ahead = vorts[ahead]
                dx   = vorts_ahead[:, 0] - self.uav.x
                dy   = vorts_ahead[:, 1] - self.uav.y
                dist = np.sqrt(dx**2 + dy**2)
                idx  = np.argsort(dist)[:3]
                for i, j in enumerate(idx):
                    vx_near[2*i]   = dx[j]
                    vx_near[2*i+1] = dy[j]
                    gm_near[i]     = np.sign(vorts_ahead[j, 2])

        v_wind = self.wind_amp * np.sin(
            2 * np.pi * self.wind_freq * self.step_count * self.dt
            + self.wind_phase
        )

        return np.array([
            self.uav.x,       self.uav.y,
            self.uav.vx,      self.uav.vy,
            self.uav.heading,
            u_flow,           v_flow,      dvdy,
            vx_near[0], vx_near[1], gm_near[0],
            vx_near[2], vx_near[3], gm_near[1],
            vx_near[4], vx_near[5], gm_near[2],
            v_wind
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        cur      = self.curriculum
        rng      = self.np_random

        self.gamma0  = float(rng.uniform(*cur["gamma0_range"]))
        wind_amp     = float(rng.uniform(*cur["wind_amp_range"]))
        y_start      = float(rng.uniform(*cur["y_start_range"]))
        noise        = cur["noise_scale"]
        nu           = self.nu_override if self.nu_override else 0.00004

        self.street = VortexStreet(
            U=self.U, d=0.5, h=1.2,
            gamma0=self.gamma0,
            delta=0.08, dt=self.dt,
            x_kill=30.0,
            nu=nu,
            noise_scale=noise,
            rng=np.random.default_rng(int(rng.integers(0, 2**31)))
        )
        for _ in range(400):
            self.street.step()

        self.uav           = UAV(x0=0.5, y0=y_start, vx0=0.7, vy0=0.0)
        self.step_count    = 0
        self.total_thrust  = 0.0
        self.total_P_flow  = 0.0
        self.total_P_input = 0.0

        self.wind_amp   = wind_amp
        self.wind_freq  = float(rng.uniform(0.05, 0.15))
        self.wind_phase = float(rng.uniform(0, 2*np.pi))

        return self._get_obs(), {}

    def _wrap_street(self):
        self.uav.x = 0.5
        nu = self.nu_override if self.nu_override else 0.00004
        self.street = VortexStreet(
            U=self.U, d=0.5, h=1.2,
            gamma0=self.gamma0,
            delta=0.08, dt=self.dt,
            x_kill=30.0,
            nu=nu,
            noise_scale=self.curriculum["noise_scale"],
            rng=np.random.default_rng()
        )
        for _ in range(200):
            self.street.step()

    def step(self, action):
        thrust = float(action[0])
        hdg_rate = float(action[1])

        # crosswind
        v_wind = self.wind_amp * np.sin(
            2 * np.pi * self.wind_freq * self.step_count * self.dt
            + self.wind_phase
        )

        # local flow velocity
        u_flow, v_flow = self.street.get_velocity(self.uav.x, self.uav.y)
        v_flow_total = v_flow + v_wind

        # step UAV — heading rate control
        self.uav.step(thrust, hdg_rate, u_flow, v_flow_total, self.dt)
        self.street.step()
        self.step_count += 1

        # energy metrics
        P_thrust = thrust * np.sqrt(self.uav.vx ** 2 + self.uav.vy ** 2)

        vrel_x = self.uav.vx - u_flow
        vrel_y = self.uav.vy - v_flow_total
        vrel = np.sqrt(vrel_x ** 2 + vrel_y ** 2) + 1e-6
        V_agent = np.sqrt(self.uav.vx ** 2 + self.uav.vy ** 2) + 1e-6

        alpha = np.arctan2(-vrel_y, vrel_x) - self.uav.heading
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        CL = np.clip(2 * np.pi * alpha, -1.2, 1.2)
        CD = 0.02 + 0.05 * CL ** 2
        q = 0.5 * 1.225 * vrel ** 2
        D = q * 0.1 * CD

        P_flow = D * (u_flow * vrel_x + v_flow_total * vrel_y) / \
                 (vrel * V_agent + 1e-6)

        self.total_thrust += thrust
        self.total_P_flow += max(0.0, P_flow)
        self.total_P_input += P_thrust

        # reward components
        harvest = 0.3 * max(0.0, P_flow)
        progress = min(max(0.0, self.uav.vx - self.U) * self.dt, 0.02)
        thrust_pen = 0.15 * (thrust / self.uav.T_max) ** 2
        lat_pen = 0.04 * self.uav.y ** 2
        boundary_pen = 0.1 * max(0.0, abs(self.uav.y) - 2.0)
        survival = 0.005 if self.uav.vx > self.U * 0.8 else -0.05
        # penalize falling behind freestream
        slow_pen = 0.05 * max(0, self.U * 0.8 - self.uav.vx)

        reward = (progress + harvest + survival
                  - thrust_pen - lat_pen - boundary_pen - slow_pen)

        # termination
        terminated = bool(abs(self.uav.y) > self.y_bounds)
        truncated = bool(self.step_count >= self.max_steps)

        # goal reached — efficiency bonus replaces step reward
        if self.uav.x >= self.x_goal:
            mean_thrust = self.total_thrust / max(self.step_count, 1)
            efficiency_bonus = 20.0 * np.exp(-6.0 * mean_thrust)
            reward = efficiency_bonus
            terminated = True
            return self._get_obs(), reward, terminated, truncated, {}

        return self._get_obs(), reward, terminated, truncated, {}

    def get_energy_metrics(self):
        eps = 1e-6
        eta = self.total_P_flow / \
              (self.total_P_input + self.total_P_flow + eps)
        return {
            'total_thrust'       : self.total_thrust,
            'total_P_flow'       : self.total_P_flow,
            'total_P_input'      : self.total_P_input,
            'harvest_efficiency' : eta,
        }


if __name__ == "__main__":
    env   = KarmanEnv()
    obs, _ = env.reset()
    print(f"obs shape: {obs.shape}")
    print(f"initial obs: {obs}")

    total_reward = 0
    steps        = 0
    for _ in range(800):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps        += 1
        if terminated or truncated:
            break

    print(f"total reward:      {total_reward:.3f}")
    print(f"steps survived:    {steps}")
    print(f"final x:           {obs[0]:.3f}")
    print(f"final y:           {obs[1]:.3f}")
    print(f"reward per step:   {total_reward/steps:.4f}")
    metrics = env.get_energy_metrics()
    print(f"harvest efficiency:{metrics['harvest_efficiency']:.4f}")
    print(f"total P_flow:      {metrics['total_P_flow']:.4f}")
    print(f"total P_input:     {metrics['total_P_input']:.4f}")
