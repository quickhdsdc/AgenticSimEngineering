"""Panda Robot Domain."""
from pathlib import Path

import collections
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards

# Define maximum time limit
_DEFAULT_TIME_LIMIT = 2

# Define task container
SUITE = containers.TaggedTasks()

def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    base_path = str(Path(__file__).resolve().parents[2] / 'assets/suite/assets_panda') + '/'
    
    assets = {
        'link4_2.obj': common.read_model(base_path + 'link4_2.obj'),
        'link6_13.obj': common.read_model(base_path + 'link6_13.obj'),
        'hand.stl': common.read_model(base_path + 'hand.stl'),
        'link7_1.obj': common.read_model(base_path + 'link7_1.obj'),
        'link7_2.obj': common.read_model(base_path + 'link7_2.obj'),
        'link6.stl': common.read_model(base_path + 'link6.stl'),
        'link6_3.obj': common.read_model(base_path + 'link6_3.obj'),
        'link6_12.obj': common.read_model(base_path + 'link6_12.obj'),
        'link0_1.obj': common.read_model(base_path + 'link0_1.obj'),
        'link1.stl': common.read_model(base_path + 'link1.stl'),
        'link5_2.obj': common.read_model(base_path + 'link5_2.obj'),
        'link3_1.obj': common.read_model(base_path + 'link3_1.obj'),
        'link7_0.obj': common.read_model(base_path + 'link7_0.obj'),
        'hand_0.obj': common.read_model(base_path + 'hand_0.obj'),
        'link5_collision_0.obj': common.read_model(base_path + 'link5_collision_0.obj'),
        'link2.obj': common.read_model(base_path + 'link2.obj'),
        'link6_14.obj': common.read_model(base_path + 'link6_14.obj'),
        'link6_16.obj': common.read_model(base_path + 'link6_16.obj'),
        'hand_2.obj': common.read_model(base_path + 'hand_2.obj'),
        'link0.stl': common.read_model(base_path + 'link0.stl'),
        'link4_3.obj': common.read_model(base_path + 'link4_3.obj'),
        'hand_3.obj': common.read_model(base_path + 'hand_3.obj'),
        'hand_1.obj': common.read_model(base_path + 'hand_1.obj'),
        'link6_5.obj': common.read_model(base_path + 'link6_5.obj'),
        'link7_4.obj': common.read_model(base_path + 'link7_4.obj'),
        'link7_3.obj': common.read_model(base_path + 'link7_3.obj'),
        'link3_0.obj': common.read_model(base_path + 'link3_0.obj'),
        'link7_6.obj': common.read_model(base_path + 'link7_6.obj'),
        'link5_collision_1.obj': common.read_model(base_path + 'link5_collision_1.obj'),
        'finger_0.obj': common.read_model(base_path + 'finger_0.obj'),
        'link7.stl': common.read_model(base_path + 'link7.stl'),
        'link3.stl': common.read_model(base_path + 'link3.stl'),
        'hand_4.obj': common.read_model(base_path + 'hand_4.obj'),
        'link0_3.obj': common.read_model(base_path + 'link0_3.obj'),
        'link6_0.obj': common.read_model(base_path + 'link6_0.obj'),
        'link6_6.obj': common.read_model(base_path + 'link6_6.obj'),
        'link4_1.obj': common.read_model(base_path + 'link4_1.obj'),
        'link6_1.obj': common.read_model(base_path + 'link6_1.obj'),
        'link0_10.obj': common.read_model(base_path + 'link0_10.obj'),
        'link3_3.obj': common.read_model(base_path + 'link3_3.obj'),
        'link0_7.obj': common.read_model(base_path + 'link0_7.obj'),
        'link3_2.obj': common.read_model(base_path + 'link3_2.obj'),
        'link6_10.obj': common.read_model(base_path + 'link6_10.obj'),
        'link7_5.obj': common.read_model(base_path + 'link7_5.obj'),
        'link2.stl': common.read_model(base_path + 'link2.stl'),
        'link7_7.obj': common.read_model(base_path + 'link7_7.obj'),
        'link0_8.obj': common.read_model(base_path + 'link0_8.obj'),
        'link6_2.obj': common.read_model(base_path + 'link6_2.obj'),
        'link6_15.obj': common.read_model(base_path + 'link6_15.obj'),
        'link6_8.obj': common.read_model(base_path + 'link6_8.obj'),
        'finger_1.obj': common.read_model(base_path + 'finger_1.obj'),
        'link5_0.obj': common.read_model(base_path + 'link5_0.obj'),
        'link5_collision_2.obj': common.read_model(base_path + 'link5_collision_2.obj'),
        'link0_4.obj': common.read_model(base_path + 'link0_4.obj'),
        'link6_7.obj': common.read_model(base_path + 'link6_7.obj'),
        'link5_1.obj': common.read_model(base_path + 'link5_1.obj'),
        'link0_5.obj': common.read_model(base_path + 'link0_5.obj'),
        'link6_4.obj': common.read_model(base_path + 'link6_4.obj'),
        'link1.obj': common.read_model(base_path + 'link1.obj'),
        'link0_11.obj': common.read_model(base_path + 'link0_11.obj'),
        'link4_0.obj': common.read_model(base_path + 'link4_0.obj'),
        'link0_9.obj': common.read_model(base_path + 'link0_9.obj'),
        'link6_11.obj': common.read_model(base_path + 'link6_11.obj'),
        'link4.stl': common.read_model(base_path + 'link4.stl'),
        'link0_0.obj': common.read_model(base_path + 'link0_0.obj'),
        'link0_2.obj': common.read_model(base_path + 'link0_2.obj'),
        'link6_9.obj': common.read_model(base_path + 'link6_9.obj'),
    }
    
    return common.read_model('scene.xml'), assets

@SUITE.add('benchmarking')
def grasp(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the panda robot grasping task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = PandaTask(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics simulator for the Panda robot with extra features."""

    def get_joint_positions(self):
        """Returns joint positions."""
        joints = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        positions = {joint: self.named.data.qpos[joint].copy() for joint in joints}
        return positions

    def get_joint_velocities(self):
        """Returns joint velocities."""
        joints = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        velocities = {joint: self.named.data.qvel[joint].copy() for joint in joints}
        return velocities

    def get_finger_positions(self):
        """Returns finger positions."""
        fingers = ['finger_joint1', 'finger_joint2']
        positions = {finger: self.named.data.qpos[finger].copy() for finger in fingers}
        return positions


class PandaTask(base.Task):
    """A task class for the Panda robot."""

    def __init__(self, random=None):
        """Initializes the Panda task."""
        self._random = random or np.random.RandomState()

    def initialize_episode(self, physics):
        """Initializes the environment at the beginning of each episode."""
        random_qpos = physics.data.qpos[:]
        random_qvel = np.random.randn(physics.model.nv)

        physics.data.qpos[:] = random_qpos
        physics.data.qvel[:] = random_qvel

        for _ in range(50):
            physics.step()

        physics.data.time = 0
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns observations for the agent."""
        obs = collections.OrderedDict()
        obs['position'] = physics.data.qpos[:].copy()
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        """Returns the reward for the agent."""
        target_qpos = np.zeros(physics.model.nq)
        qpos = physics.data.qpos
        balance_reward = rewards.tolerance(
            np.abs(qpos - target_qpos).sum(),
            bounds=(0, 0.5),
            margin=0.5,
            value_at_margin=0,
            sigmoid='linear'
        )
        return balance_reward
