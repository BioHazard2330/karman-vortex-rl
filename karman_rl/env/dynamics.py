import numpy as np


class UAV:
    def __init__(self, x0=0.0, y0=0.0, vx0=0.7, vy0=0.0):
        self.x       = x0
        self.y       = y0
        self.vx      = vx0
        self.vy      = vy0
        self.heading = 0.0   # radians, 0 = pointing downstream

        # physical parameters
        self.mass    = 1.0      # kg
        self.rho     = 1.225    # kg/m^3
        self.S       = 0.1      # wing reference area m^2
        self.T_max   = 1.0      # max thrust N

        # drag polar: CD = CD0 + k*CL^2
        self.CD0     = 0.02
        self.k       = 0.05
        self.CL_max  = 1.2

        # heading rate limits — max turn rate rad/s
        self.hdg_rate_max = np.pi / 4   # 45 deg/s

        # heading limits — keep UAV roughly pointing downstream
        self.hdg_max = np.pi / 1.5       # 90 degrees max bank

    def _aero_forces(self, vrel_x, vrel_y):
        V     = np.sqrt(vrel_x**2 + vrel_y**2) + 1e-6
        alpha = np.arctan2(-vrel_y, vrel_x) - self.heading
        alpha = (alpha + np.pi) % (2*np.pi) - np.pi

        CL    = np.clip(2*np.pi*alpha, -self.CL_max, self.CL_max)
        CD    = self.CD0 + self.k*CL**2
        q     = 0.5 * self.rho * V**2

        D     = q * self.S * CD
        L     = q * self.S * CL

        # drag opposes relative velocity
        Fd_x  = -D * vrel_x / V
        Fd_y  = -D * vrel_y / V

        # lift perpendicular to relative velocity
        Fl_x  =  L * vrel_y / V
        Fl_y  = -L * vrel_x / V

        return Fd_x + Fl_x, Fd_y + Fl_y

    def step(self, thrust, hdg_rate, u_flow, v_flow, dt):
        # integrate heading — rate command with hard limits
        self.heading += hdg_rate * dt
        self.heading  = np.clip(self.heading,
                                -self.hdg_max, self.hdg_max)

        # relative velocity w.r.t local flow
        vrel_x = self.vx - u_flow
        vrel_y = self.vy - v_flow

        # aerodynamic forces
        Fa_x, Fa_y = self._aero_forces(vrel_x, vrel_y)

        # thrust along current heading
        Ft_x   = thrust * np.cos(self.heading)
        Ft_y   = thrust * np.sin(self.heading)

        # equations of motion — Euler integration
        ax     = (Ft_x + Fa_x) / self.mass
        ay     = (Ft_y + Fa_y) / self.mass

        self.vx += ax * dt
        self.vy += ay * dt
        self.x  += self.vx * dt
        self.y  += self.vy * dt

    @property
    def state(self):
        return np.array([self.x, self.y,
                         self.vx, self.vy, self.heading])


if __name__ == "__main__":
    uav = UAV()

    # case 1: still air, fly straight with thrust
    for _ in range(20):
        uav.step(thrust=0.5, hdg_rate=0.0,
                 u_flow=0.0, v_flow=0.0, dt=0.05)
    print(f"still air: x={uav.x:.3f}, vx={uav.vx:.3f}, "
          f"heading={np.degrees(uav.heading):.1f}deg")

    # case 2: flow matching UAV — zero drag
    uav2 = UAV(vx0=1.0)
    for _ in range(20):
        uav2.step(thrust=0.5, hdg_rate=0.0,
                  u_flow=uav2.vx, v_flow=0.0, dt=0.05)
    print(f"zero drag: x={uav2.x:.3f}, vx={uav2.vx:.3f}")

    # case 3: turning — apply heading rate
    uav3 = UAV(vx0=1.0)
    for _ in range(20):
        uav3.step(thrust=0.5, hdg_rate=np.pi/8,
                  u_flow=1.0, v_flow=0.0, dt=0.05)
    print(f"turning: heading={np.degrees(uav3.heading):.1f}deg, "
          f"y={uav3.y:.3f}")