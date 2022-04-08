from gym.envs.registration import register

register(
    id="randomized_cartpole_env-v0",
    entry_point="randomized_cartpole_env.randomized_cartpole_env:RandomizedCartpoleEnv",
)
