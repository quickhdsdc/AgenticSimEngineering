import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

sys.path.append('projects/Neural-Simulator/Zero_Shot_RL/src')
from MoSim.src.tools import initialize_residual_flow, load_ode_from_ckpt
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from oprl.algos.tqc import TQC
from oprl.configs.utils import create_logdir, parse_args
from oprl.utils.utils import set_logging

set_logging(logging.INFO)
from oprl.env import make_env as _make_env
from oprl.utils.logger import FileLogger, Logger
from oprl.utils.run_training import run_training

args = parse_args()


def make_env(seed: int, ckpt_path, DEVICE, dt, step_limit, nfm= None):
    return _make_env(args.env, seed=seed, ckpt_path=ckpt_path, DEVICE=DEVICE, dt=dt, step_limit=step_limit, nfm=nfm)



env = make_env(seed=0, ckpt_path='/projects/Neural-Simulator/ckpt/vsdreamer/cheetah/base/ckpt_450M.pth', DEVICE=args.device, dt=0.01, step_limit=100)
STATE_DIM: int = env.observation_space.shape[0]
ACTION_DIM: int = env.action_space.shape[0]
transition_model=None
# --------  Config params -----------

config = {
    "horizen": 1,
    "state_dim": STATE_DIM,
    "action_dim": ACTION_DIM,
    "num_steps": int(1_000_000000),
    "eval_every": 2500,
    "device": args.device,
    "save_buffer": False,
    "visualise_every": 0,
    "estimate_q_every": 0,  # TODO: Here is the unsupported logic
    "log_every": 2500,
    "ckpt_path": "/projects/Neural-Simulator/ckpt/vsdreamer/cheetah/base/ckpt_450M.pth",
    "dt": 0.01,
    "step_limit": 100,
    'task_name': 'cheetah-run-MoSim'
}

# -----------------------------------
def make_env_eval(seed: int, ckpt_path=config['ckpt_path'], DEVICE=config['device'], dt=config['dt'], step_limit=config['step_limit']):
    return _make_env(args.env, seed=seed, if_eval=True, ckpt_path=ckpt_path, DEVICE=DEVICE, dt=dt, step_limit=step_limit)


ckpt_path = '/projects/Neural-Simulator/data/archive/cheetah_old/cheetah_random_test/'
K = 16
latent_size = 24
hidden_units = 128
hidden_layers = 3
state_dim = 24
batch_size = 50000
if_ckpt = '/projects/Neural-Simulator/ckpt/flow/cheetah/cheetah_02111639/ckpt_14046M.pth'
nfm = initialize_residual_flow(K,latent_size,hidden_units,hidden_layers,state_dim,args.device, ckpt_path, batch_size, if_ckpt)





def make_algo(logger: Logger):
    return TQC(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device=args.device,
        logger=logger,
    )


def make_logger(seed: int) -> Logger:
    global config
    log_dir = create_logdir(logdir="logs", algo="TQC", env=args.env, seed=seed)
    return FileLogger(log_dir, config)


if __name__ == "__main__":
    args = parse_args()
    run_training(make_algo, make_env, make_env_eval, make_logger, nfm, transition_model, config, args.seeds, args.start_seed)
# python src/oprl/configs/tqc.py --env cheetah-run-MoSim --device cuda:2


# 4-51-acrobot
# 3-52-cheetah
