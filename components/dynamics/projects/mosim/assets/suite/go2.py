# Copyright 2024 Your Name.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Go2 Domain."""

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np

_DEFAULT_TIME_LIMIT = 2
_STAND_HEIGHT_MIN = 0.15
_STAND_HEIGHT_MAX = 0.3
SUITE = containers.TaggedTasks()


def get_model_and_assets():
    assets = {
        'base_0.obj': common.read_model('assets/base_0.obj'),
        'base_1.obj': common.read_model('assets/base_1.obj'),
        'base_2.obj': common.read_model('assets/base_2.obj'),
        'base_3.obj': common.read_model('assets/base_3.obj'),
        'base_4.obj': common.read_model('assets/base_4.obj'),
        'hip_0.obj': common.read_model('assets/hip_0.obj'),
        'hip_1.obj': common.read_model('assets/hip_1.obj'),
        'thigh_0.obj': common.read_model('assets/thigh_0.obj'),
        'thigh_1.obj': common.read_model('assets/thigh_1.obj'),
        'thigh_mirror_0.obj': common.read_model('assets/thigh_mirror_0.obj'),
        'thigh_mirror_1.obj': common.read_model('assets/thigh_mirror_1.obj'),
        'calf_0.obj': common.read_model('assets/calf_0.obj'),
        'calf_1.obj': common.read_model('assets/calf_1.obj'),
        'calf_mirror_0.obj': common.read_model('assets/calf_mirror_0.obj'),
        'calf_mirror_1.obj': common.read_model('assets/calf_mirror_1.obj'),
        'foot.obj': common.read_model('assets/foot.obj'),
    }
    return common.read_model('smooth_test.xml'), assets


@SUITE.add('benchmarking')
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Go2Task(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


@SUITE.add('benchmarking')
def balance(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Go2Task(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


@SUITE.add('benchmarking')
def hop(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Go2Task(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


class Physics(mujoco.Physics):
    def get_joint_positions(self):
        joints = ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
                  'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
                  'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
                  'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint']
        positions = {joint: self.named.data.qpos[joint].copy() for joint in joints}
        return positions

    def get_joint_velocities(self):
        joints = ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
                  'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
                  'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
                  'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint']
        velocities = {joint: self.named.data.qvel[joint].copy() for joint in joints}
        return velocities

    def get_imu_data(self):
        gyro = self.named.data.sensordata['gyro'].copy()
        accelerometer = self.named.data.sensordata['accelerometer'].copy()
        orientation = self.named.data.sensordata['orientation'].copy()
        global_position = self.named.data.sensordata['global_position'].copy()
        global_linvel = self.named.data.sensordata['global_linvel'].copy()
        global_angvel = self.named.data.sensordata['global_angvel'].copy()
        return {
            'gyro': gyro,
            'accelerometer': accelerometer,
            'orientation': orientation,
            'global_position': global_position,
            'global_linvel': global_linvel,
            'global_angvel': global_angvel
        }


def randomize_limited_and_rotational_joints(physics, random=None):
    random = random or np.random
    qpos = physics.data.qpos
    for joint_id in range(physics.model.njnt):
        joint_type = physics.model.jnt_type[joint_id]
        is_limited = physics.model.jnt_limited[joint_id]
        range_min, range_max = physics.model.jnt_range[joint_id]

        if is_limited:
            if joint_type in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
                qpos[joint_id] = random.uniform(range_min, range_max)
        else:
            if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                qpos[joint_id] = random.uniform(-np.pi, np.pi)
            elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                quat = random.randn(4)
                quat /= np.linalg.norm(quat)
                qpos[joint_id:joint_id + 4] = quat
            elif joint_type == mujoco.mjtJoint.mjJNT_FREE:
                quat = random.randn(4)
                quat /= np.linalg.norm(quat)
                qpos[joint_id+3:joint_id+7] = quat


class Go2Task(base.Task):
    def __init__(self, task=1, random=None):
        self._random = random or np.random.RandomState()
        self.task_type = task
        super().__init__(random=random)

    def initialize_episode(self, physics):
        initial_qpos = np.array([
            0, 0, 0.27 + np.random.uniform(0, 0.02),
            1, 0, 0, 0,
            0, 0.9, -1.8,
            0, 0.9, -1.8,
            0, 0.9, -1.8,
            0, 0.9, -1.8
        ])
        initial_qvel = np.zeros(physics.model.nv)
        physics.data.qpos[:] = initial_qpos
        physics.data.qvel[:] = initial_qvel
        physics.data.time = 0
        self._timeout_progress = 0
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.data.qpos[:].copy()
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        height = physics.named.data.qpos[2]
        standing = rewards.tolerance(height, (_STAND_HEIGHT_MIN, 0.3))
        if self.task_type == 3:
            vertical_speed = physics.named.data.qvel[2]
            jump_reward = rewards.tolerance(
                vertical_speed,
                bounds=(1.0, float('inf')),
                margin=0.5,
                value_at_margin=0,
                sigmoid='linear'
            )
            forward_speed = physics.named.data.qvel[2][0]
            forward_reward = rewards.tolerance(
                forward_speed,
                bounds=(1.0, float('inf')),
                margin=0.5,
                value_at_margin=0,
                sigmoid='linear'
            )
            total_hop_reward = standing * (jump_reward + forward_reward)
            return total_hop_reward
        elif self.task_type == 1:
            forward_speed = physics.named.data.qvel[0]
            run_reward = rewards.tolerance(
                forward_speed,
                bounds=(1.0, float('inf')),
                margin=0.5,
                value_at_margin=0,
                sigmoid='linear'
            )
            return standing * run_reward
