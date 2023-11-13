import gym
from gail_airl_ppo.crowd_env import CrowdEnv

gym.logger.set_level(40)


def make_env(args,TT_type):
    env = CrowdEnv(args, TT_type)
    return NormalizedEnv(env)
    # return NormalizedEnv(gym.make(env_id))


class NormalizedEnv(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps

        self.scale = env.action_space.high

    def step(self, actions):
        # scale the actions
        scaled_actions = {}
        for id in actions.keys():
            scaled_actions[id] = actions[id] * self.scale
        return self.env.step(scaled_actions)
