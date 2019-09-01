from gym.envs.registration import register

register(
    id='hand_cube-v0',
    entry_point='gym_test.envs:HandGymEnv',
    
)
