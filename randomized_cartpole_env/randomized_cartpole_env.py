import math
import random
import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding


class RandomizedCartpoleEnv(gym.Env):
    """
    # Modification
    # Add the friction in pole and cart
    # Change the starting pole position
    # Continuous action: -2.0(left) ~ 0(stop) ~ 2.0(right)
    # Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        # environment
        self.seed()
        self.gravity = 9.8
        self.statesize = 4
        self.actionsize = 1
        # cart
        self.masscart = 7.0
        self.friction_cart = 1.0  # default
        # pole
        self.masspole = 1.5
        self.length = 0.3
        self.polemass_length = self.masspole * self.length
        self.friction_pole = 0.995  # smaller value is more stable pole behavior
        # cartpole
        self.total_mass = self.masspole + self.masscart
        # control
        self.force_mag = 60.0  # cart movement gain
        self.tau = 0.010  # seconds between state updates, bigger:faster default: 0.02
        self.state = None
        self.initial_state = (0, 0, 0.5, 0)  # pole downward, sin(47.2/2) = -1.0)
        # viewer
        self.viewer = None
        self.pole_color = 0.0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        x, x_dot, theta, theta_dot = self.state

        # continuous action
        force = action * self.force_mag

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass

        # d omega
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )

        # d velocity
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # friction
        x_dot = x_dot * self.friction_cart
        theta_dot = theta_dot * self.friction_pole

        # calculate state
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        return np.array(self.state), {}, {}, {}

    def reset(self):
        self.randomize_domain()
        self.state = self.initial_state
        return np.array(self.state)

    def randomize_domain(self):
        self.gravity = 9.8 + random.uniform(-1.0, 1.0)
        self.masscart = 7.0 + random.uniform(1.0, 2.0)
        self.masspole = 1.5 + random.uniform(1.0, 2.0)
        self.force_mag = 50.0 + random.uniform(-5.0, 5.0)
        self.length = random.uniform(0.0, 1.0)
        self.polemass_length = self.masspole * self.length
        self.total_mass = self.masspole + self.masscart
        self.viewer = None

    def render(self, mode="human"):
        # screen setting
        screen_width = 3840
        screen_height = 1000
        x_threshold = 2.4  # task space in x
        world_width = x_threshold * 2
        scale = screen_width / world_width

        # object setting
        cartwidth = 100.0
        cartheight = 50.0
        polewidth = 20.0
        carty = 600  # center line height
        polelen = scale * self.length

        # render setting (only one time)
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)

            # viewer cart
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

            # viewer pole
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            self.pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            axleoffset = cartheight / 4.0
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            self.pole.add_attr(self.poletrans)
            self.pole.add_attr(self.carttrans)
            self.viewer.add_geom(self.pole)
            self._pole_geom = self.pole

            # viewer free joint
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)

            # viewer center line
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        ### change viewer
        x = self.state

        # move cart
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)

        # rotate pole
        self.poletrans.set_rotation(-x[2])

        # change pole color
        self.pole.set_color(self.pole_color, 0.0, 0.0)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
