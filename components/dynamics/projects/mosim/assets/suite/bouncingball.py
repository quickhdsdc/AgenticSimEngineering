from pathlib import Path
import collections
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np


_DEFAULT_TIME_LIMIT = 10
SUITE = containers.TaggedTasks()


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model(str(Path(__file__).resolve().parents[2] / 'assets/suite/bouncingball.xml')), common.ASSETS


@SUITE.add('benchmarking')
def bounce(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Bouncing Ball task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = BouncingBallTask(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Bouncing Ball domain."""

  

class BouncingBallTask(base.Task):
    """A task where a ball bounces on the floor."""

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)
         # 设置球的初始速度
        physics.named.data.qvel[:3] = [1, 2, 0]  # x, y, z 的线速度
        physics.named.data.qvel[3:] = [0., 0., 0.]  # x, y, z 的角速度

        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of the ball's position and velocity."""
        obs = collections.OrderedDict()
        obs['position'] = physics.named.data.qvel[:3]
        obs['velocity'] = physics.named.data.qvel[:3]
        return obs

    def get_reward(self, physics):
        """Returns a reward for the task based on the ball's height."""
        # Example reward: higher rewards for higher bounces
        # ball_height = physics.named.data.xipos['ball'][2]
        reward = 0
        return reward
