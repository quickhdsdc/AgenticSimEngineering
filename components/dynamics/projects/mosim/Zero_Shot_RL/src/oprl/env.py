from abc import ABC, abstractmethod
from collections import OrderedDict
from platform import node
import sys
from typing import Any
import numpy as np
import numpy.typing as npt
from dm_control import suite
from MoSim.src.tools import cal_prob, load_ode_from_ckpt
import torch
from oprl.trainers.buffers.episodic_buffer import EpisodicReplayBuffer
class BaseEnv(ABC):
    @abstractmethod
    def reset(self) -> tuple[npt.ArrayLike, dict[str, Any]]:
        pass

    @abstractmethod
    def step(
        self, action: npt.ArrayLike
    ) -> tuple[npt.ArrayLike, npt.ArrayLike, bool, bool, dict[str, Any]]:
        pass

    @abstractmethod
    def sample_action(self) -> npt.ArrayLike:
        pass

    @property
    def env_family(self) -> str:
        return ""


class DummyEnv(BaseEnv):
    def reset(self) -> tuple[npt.ArrayLike, dict[str, Any]]:
        return np.array([]), {}

    def step(
        self, action: npt.ArrayLike
    ) -> tuple[npt.ArrayLike, npt.ArrayLike, bool, bool, dict[str, Any]]:
        return np.array([]), np.array([]), False, False, {}

    def sample_action(self) -> npt.ArrayLike:
        return np.array([])

    @property
    def env_family(self) -> str:
        return ""


class SafetyGym(BaseEnv):
    def __init__(self, env_name: str, seed: int):
        import safety_gymnasium as gym

        self._env = gym.make(env_name)
        self._seed = seed

    def step(
        self, action: npt.ArrayLike
    ) -> tuple[npt.ArrayLike, npt.ArrayLike, bool, bool, dict[str, Any]]:
        obs, reward, cost, terminated, truncated, info = self._env.step(action)
        info["cost"] = cost
        return obs, reward, terminated, truncated, info

    def reset(self) -> tuple[npt.ArrayLike, dict[str, Any]]:
        obs, info = self._env.reset(seed=self._seed)
        self._env.step(self._env.action_space.sample())
        return obs, info

    def sample_action(self):
        return self._env.action_space.sample()

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def env_family(self) -> str:
        return "safety_gymnasium"


class DMControlEnv(BaseEnv):
    def __init__(self, env: str, seed: int ,if_eval:bool, buffer = None):
        domain, task = env.split("-")
        self.random_state = np.random.RandomState(seed)
        self.env = suite.load(domain, task, task_kwargs={"random": self.random_state})

        self._render_width = 200
        self._render_height = 200
        self._camera_id = 0
        self.if_eval = if_eval

    def reset(self, *args, **kwargs) -> tuple[npt.ArrayLike, dict[str, Any]]:
        obs = self._flat_obs(self.env.reset().observation)
        return obs, {}

    def step(
        self, action: npt.ArrayLike
    ) -> tuple[npt.ArrayLike, npt.ArrayLike, bool, bool, dict[str, Any]]:
        time_step = self.env.step(action)
        obs = self._flat_obs(time_step.observation)

        terminated = False
        truncated = self.env._step_count >= self.env._step_limit

        return obs, time_step.reward, terminated, truncated, {}

    def sample_action(self) -> npt.ArrayLike:
        spec = self.env.action_spec()
        action = self.random_state.uniform(spec.minimum, spec.maximum, spec.shape)
        return action

    @property
    def observation_space(self) -> npt.ArrayLike:
        return np.zeros(
            sum(int(np.prod(v.shape)) for v in self.env.observation_spec().values())
        )

    @property
    def action_space(self) -> npt.ArrayLike:
        return np.zeros(self.env.action_spec().shape[0])

    def render(self) -> npt.ArrayLike:
        """
        returned shape: [1, W, H, C]
        """
        img = self.env.physics.render(
            camera_id=self._camera_id,
            height=self._render_width,
            width=self._render_width,
        )
        img = img.astype(np.uint8)
        return np.expand_dims(img, 0)
    
    def set_state_and_get_obs(self, qpos, qvel):
        batch_size = qpos.shape[0]
        rewards = []
        obs_ = []

        for i in range(batch_size):
            self.env.physics.data.qpos[:] = qpos[i].cpu().numpy()
            self.env.physics.data.qvel[:] = qvel[i].cpu().numpy()
            self.env.physics.forward()  

            reward = self.env.task.get_reward(self.env.physics)
            obs = self.env.task.get_observation(self.env.physics)
            obs=self._flat_obs( obs, 'position')
            rewards.append(reward)
            obs_.append(obs)

        rewards = torch.tensor(rewards, dtype=torch.float32).reshape(batch_size, 1).to(qpos.device)
        obs_ = torch.tensor(obs_, dtype=torch.float32).reshape(batch_size, -1).to(qpos.device)

        
        return rewards,obs_



    def _flat_obs(self, obs: OrderedDict) -> npt.ArrayLike:
        obs_flatten = []
        for key, o in obs.items():
            if len(o.shape) == 0:
                obs_flatten.append(np.array([o]))
            elif len(o.shape) == 2 and o.shape[1] > 1:
                obs_flatten.append(o.flatten())
            else:
                obs_flatten.append(o)
        return np.concatenate(obs_flatten, dtype="float32")
    @property
    def env_family(self) -> str:
        return "dm_control"



class MOSimEnv(DMControlEnv):
    def __init__(self, env: str, seed: int, ckpt_path: str, DEVICE, dt, step_limit,nfm,buffer = None):
        domain, task, _ = env.split("-")
        self.random_state = np.random.RandomState(seed)
        self.env = suite.load(domain, task, task_kwargs={"random": self.random_state})
        self.device = DEVICE
        self.MoSim = load_ode_from_ckpt(ckpt_path).to(self.device)
        if nfm != None:
            self.nfm = nfm
        else:
            self.nfm = nfm
        self.MoSim.manually_int_q = True
        self.state = None
        self.dt = dt
        self._render_width = 200
        self._render_height = 200
        self._camera_id = 0
        self._step_limit = step_limit
        self._step_count = 0
        self.buffer = buffer 
    
    def reset(self, *args, **kwargs) -> tuple[npt.ArrayLike, dict[str, Any]]:
        obs = self._flat_obs(self.env.reset().observation)
        self.state = torch.tensor(np.concatenate((self.env.physics.position(),self.env.physics.velocity()),axis=0)).to(self.device).unsqueeze(0)
        self._step_count = 0
        return obs, {}

    def step(
        self, action: npt.ArrayLike
    ) -> tuple[npt.ArrayLike, npt.ArrayLike, bool, bool, dict[str, Any]]:
        with torch.no_grad():
            state = self.MoSim(self.state.to(self.device).float(), torch.tensor(action, dtype=torch.float32).to(self.device).unsqueeze(0), 0, self.dt)
            input_data = torch.cat((self.state.to(self.device).float(),torch.tensor(action, dtype=torch.float32).to(self.device).unsqueeze(0)),dim=-1)
            regul_term = cal_prob(self.nfm,input_data,lower=-50,upper=-10).cpu().numpy()
        self.state= state.cpu().detach().numpy() 
        qpos = self.state[:,:9].squeeze(0)
        qvel = self.state[:,9:].squeeze(0)
        reward,obs = self.set_state_and_get_obs(qpos, qvel)
        reward = np.clip(reward + regul_term, a_min=0, a_max=None)
        self.state = torch.tensor(self.state).to(self.device)
        obs = self._flat_obs(obs)
        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self._step_limit
        return obs, reward, terminated, truncated, regul_term

    def sample_action(self) -> npt.ArrayLike:
        spec = self.env.action_spec()
        action = self.random_state.uniform(spec.minimum, spec.maximum, spec.shape)
        return action

    @property
    def observation_space(self) -> npt.ArrayLike:
        return np.zeros(
            sum(int(np.prod(v.shape)) for v in self.env.observation_spec().values())
        )

    @property
    def action_space(self) -> npt.ArrayLike:
        return np.zeros(self.env.action_spec().shape[0])
    
    def set_state_and_get_obs(self,qpos, qvel):
        """
        Manually sets the qpos and qvel, and returns the reward for the current step.

        Args:
            env: A dm_control environment object.
            qpos: A numpy array representing joint positions [shoulder_angle, wrist_angle].
            qvel: A numpy array representing joint velocities [shoulder_vel, wrist_vel].

        Returns:
            reward: float, the reward value for the current step.
        """

        self.env.physics.data.qpos[:] = qpos
        self.env.physics.data.qvel[:] = qvel
        self.env.physics.forward()  
        obs = self.env.task.get_observation(self.env.physics)
        reward = self.env.task.get_reward(self.env.physics)
        return reward,obs

    def _flat_obs(self, obs: OrderedDict) -> npt.ArrayLike:
        obs_flatten = []
        for _, o in obs.items():
            if len(o.shape) == 0:
                obs_flatten.append(np.array([o]))
            elif len(o.shape) == 2 and o.shape[1] > 1:
                obs_flatten.append(o.flatten())
            else:
                obs_flatten.append(o)
        return np.concatenate(obs_flatten, dtype="float32")


    @property
    def env_family(self) -> str:
        return "MoSim"






ENV_MAPPER = {
    "dm_control": set(
        [
            "acrobot-swingup",
            "ball_in_cup-catch",
            "cartpole-balance",
            "cartpole-swingup",
            "cheetah-run",
            "finger-spin",
            "finger-turn_easy",
            "finger-turn_hard",
            "fish-upright",
            "fish-swim",
            "hopper-stand",
            "hopper-hop",
            "humanoid-stand",
            "humanoid-walk",
            "humanoid-run",
            "pendulum-swingup",
            "point_mass-easy",
            "reacher-easy",
            "reacher-hard",
            "swimmer-swimmer6",
            "swimmer-swimmer15",
            "walker-stand",
            "walker-walk",
            "walker-run",
        ]
    ),
    "safety_gymnasium": set(
        [
            "SafetyPointGoal1-v0",
            "SafetyPointGoal2-v0",
            "SafetyPointButton1-v0",
            "SafetyPointButton2-v0",
            "SafetyPointPush1-v0",
            "SafetyPointPush2-v0",
            "SafetyPointCircle1-v0",
            "SafetyPointCircle2-v0",
            "SafetyCarGoal1-v0",
            "SafetyCarGoal2-v0",
            "SafetyCarButton1-v0",
            "SafetyCarButton2-v0",
            "SafetyCarPush1-v0",
            "SafetyCarPush2-v0",
            "SafetyCarCircle1-v0",
            "SafetyCarCircle2-v0",
            "SafetyAntGoal1-v0",
            "SafetyAntGoal2-v0",
            "SafetyAntButton1-v0",
            "SafetyAntButton2-v0",
            "SafetyAntPush1-v0",
            "SafetyAntPush2-v0",
            "SafetyAntCircle1-v0",
            "SafetyAntCircle2-v0",
            "SafetyDoggoGoal1-v0",
            "SafetyDoggoGoal2-v0",
            "SafetyDoggoButton1-v0",
            "SafetyDoggoButton2-v0",
            "SafetyDoggoPush1-v0",
            "SafetyDoggoPush2-v0",
            "SafetyDoggoCircle1-v0",
            "SafetyDoggoCircle2-v0",
        ]
    ),
    "MoSim": set(
        [
            "reacher-easy-MoSim",
            "reacher-hard-MoSim",
            "cheetah-run-MoSim",
            "hopper-stand-MoSim",
            "acrobot-swingup-MoSim",
            "hopper-hop-MoSim",
        ]
    ),
}


def make_env(name: str, seed: int, ckpt_path, DEVICE, dt, step_limit, nfm = None, if_eval=False):
    """
    Args:
        name: Environment name.
    """
    for env_type, env_set in ENV_MAPPER.items():
        if name in env_set and if_eval == False:
            if env_type == "dm_control":
                return DMControlEnv(name, seed=seed, if_eval=if_eval)
            elif env_type == "safety_gymnasium":
                return SafetyGym(name, seed=seed)
            elif env_type == "MoSim":
                return MOSimEnv(name, seed, ckpt_path, DEVICE, dt, step_limit,nfm)
        elif name in env_set and if_eval == True:
            name = name[:-6]
            if env_type == "dm_control" or env_type == "MoSim":
                return DMControlEnv(name, seed=seed, if_eval=if_eval)
            elif env_type == "safety_gymnasium":
                return SafetyGym(name, seed=seed)
    else:
        raise ValueError(f"Unsupported environment: {name}")
