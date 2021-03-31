from gym.envs.registration import register

register(
    id='RecSim-v0', entry_point='recsim_env.recsim_gym:RecSimGymEnv'
)
