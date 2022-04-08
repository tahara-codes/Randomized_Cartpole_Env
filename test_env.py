import time
import random
import numpy as np

import gym
import randomized_cartpole_env


class TestEnv:
    def __init__(self):
        self.envname = "randomized_cartpole_env-v0"
        self.env = gym.make(self.envname)

        self.env.reset()

    def main(self):
        while True:
            self.env.reset()

            for i in range(100):
                action = 1
                obs, _, _, _ = self.env.step(action)
                self.env.render("human")

            self.env.close()


if __name__ == "__main__":
    test_env = TestEnv()
    test_env.main()
