import gym
import gym_test


def random_agent(episodes=100):
	env = gym.make("hand_cube-v0")
	env.reset()
	env.render()

if __name__ == "__main__":
    random_agent()
