from pathlib import Path
import numpy as np
import torch
import sys
sys.path.append(str(Path(__file__).resolve().parents[3] / 'Zero_Shot_RL/src'))
sys.path.append(str(Path(__file__).resolve().parents[3] / '.'))
from env import DMControlEnv, MOSimEnv
# 初始化 DMControlEnv 和 MOSimEnv
dm_env = DMControlEnv(env="reacher-easy", seed=43, if_eval=False)
mosim_env = MOSimEnv(
    env="reacher-easy-MoSim", 
    seed=43, 
    ckpt_path=str(Path(__file__).resolve().parents[3] / 'ckpt/reacher_1crts_int_10010019/ckpt_4710M.pth'), 
    DEVICE="cuda:7", 
    dt=0.02, 
    step_limit=200
)

# 1. 获取 DMControlEnv 的初始观测值
obs_dm, _ = dm_env.reset()
obs_dm, _ = mosim_env.reset()
# 2. 将 DMControlEnv 的初始状态设置到 MOSimEnv
qpos = obs_dm[:2]  # 假设前两维是 qpos
qvel = obs_dm[4:]  # 剩余的是 qvel
# mosim_env.set_state_and_get_obs(qpos, qvel)
# state = np.concatenate((qpos, qvel), axis=0)

# # 转换为 Tensor 并移动到 CUDA 设备
# mosim_env.state = torch.tensor(state, dtype=torch.float32).to("cuda:7").unsqueeze(0)
# 3. 执行相同的动作 5 步，并比较 obs 和 reward
for step in range(200):
    # 从 DMControlEnv 采样相同的动作
    action = dm_env.sample_action()
    # 在两个环境中执行相同的动作
    obs_dm, reward_dm, terminated_dm, truncated_dm, _ = dm_env.step(action)
    obs_mosim, reward_mosim, terminated_mosim, truncated_mosim, _ = mosim_env.step(action)

    # 比较 obs 和 reward 是否基本相同
    obs_diff = np.allclose(obs_dm, obs_mosim, atol=1e-2)
    reward_diff = np.isclose(reward_dm, reward_mosim, atol=1e-2)

    print(f"Step {step + 1}:")
    print(f"  Observation difference: {obs_diff}")
    print(f"  Reward difference: {reward_diff}")
    print(f"  DMControlEnv Obs: {obs_dm}, Reward: {reward_dm}")
    print(f"  MOSimEnv Obs: {obs_mosim}, Reward: {reward_mosim}")

    if terminated_dm or truncated_dm or terminated_mosim or truncated_mosim:
        print("One of the environments terminated or truncated.")
        break
