import torch
import numpy as np
from MoSim.src.tools import train_ode, load_data, load_ode_from_ckpt, rebalance_data
from MoSim.src.models import initialize_model
from MoSim.src.models.modules import Relative_MSELoss
import sys, argparse, yaml, os
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def parse_args():
    parser = argparse.ArgumentParser(description="training parameters")

    # 添加参数
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Configure file path."
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        required=True,
        help='Training device,such as "cpu" or "cuda"',
    )

    # 解析参数
    args = parser.parse_args()

    return args


args = parse_args()

DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")

abs_path = os.path.abspath(args.config)
if not os.path.exists(abs_path):
    print("Path doesn't exist.")
    sys.exit(1)

print(f"Load config in {abs_path}")

with open(abs_path, "r") as f:
    config = yaml.safe_load(f)

task_name = os.path.splitext(os.path.basename(abs_path))[0]

model_param = config["model"]

training_param = config["training"]
from_ckpt = training_param.get("from_ckpt")
continue_training = training_param.get("continue_training", True)

train_data_path = config["data"]["train_data_path"]
test_data_path = config["data"]["test_data_path"]
dr_config = config["data"].get("data_rebalance")

training_param.setdefault("task_name", task_name)

v_dim = model_param["v_dim"]
action_dim = model_param["action_dim"]

q_dim = model_param.get("q_dim", v_dim)
model_param["q_dim"] = q_dim

fullsize_dim = q_dim + v_dim + action_dim

torch.manual_seed(training_param["seed"])
print("device = ", DEVICE)

if from_ckpt == None:
    print("Initializing model...")
    # Initialize model:
    ode = initialize_model(**model_param).to(DEVICE)
    print("Model initialized.")
else:
    print(f"Loading model from checkpoint:{from_ckpt}...")
    if continue_training:
        ode = load_ode_from_ckpt(from_ckpt).to(DEVICE)
    else:
        override_corrector_configs = {
            "crt_num": model_param.get("crt_num", 0),
            "crt_hidden_block_num": model_param.get("crt_hidden_block_num", 0),
            "crt_network_width": model_param.get("crt_network_width", 0),
        }
        ode = load_ode_from_ckpt(
            from_ckpt, override_corrector_configs=override_corrector_configs
        )
    print("Model loaded.")

# criterion = Relative_MSELoss(if_norm=True).to(DEVICE)
criterion = torch.nn.MSELoss(reduction="none")
# load data:
print("Start loading data...")

if dr_config is not None:
    rebalanced_data_save_folder = "./data/rebalanced_data"
    os.makedirs(rebalanced_data_save_folder, exist_ok=True)

    sd_name = os.path.splitext(os.path.basename(os.path.normpath(train_data_path)))[0]
    cm_name = os.path.splitext(os.path.basename(dr_config["criterion_model_ckpt"]))[0]
    rebalanced_data_name = f"sd={sd_name}_cm={cm_name}_alpha={str(dr_config['alpha']).replace('.','d')}_ar={dr_config['allocation_rate'][0]}_{dr_config['allocation_rate'][1]}.npz"
    rebalanced_data_save_path = os.path.join(
        rebalanced_data_save_folder, rebalanced_data_name
    )

    if os.path.isfile(rebalanced_data_save_path):
        print(f"Found reblanced data in {rebalanced_data_save_path}.")
        data = np.load(rebalanced_data_save_path)
        state_train = torch.from_numpy(data["state_train"])
        action_train = torch.from_numpy(data["action_train"])
        target_train = torch.from_numpy(data["target_train"])
        print("Rebalanced data loaded.")

    else:
        state_train, action_train, target_train, _ = load_data(
            train_data_path, flatten=True
        )
        print("Rebalancing data...")
        criterion_model = load_ode_from_ckpt(dr_config["criterion_model_ckpt"])
        state_train, action_train, target_train = rebalance_data(
            criterion_model,
            state_train,
            action_train,
            target_train,
            training_param["t"],
            criterion,
            DEVICE,
            alpha=dr_config["alpha"],
            allocation_rate=dr_config["allocation_rate"],
            batch_size=10000000,
        )
        print("Data reblanced.")
        np.savez(
            rebalanced_data_save_path,
            state_train=state_train.numpy(),
            action_train=action_train.numpy(),
            target_train=target_train.numpy(),
        )
        print(f"Rebalanced data saved in {rebalanced_data_save_path}.")
else:
    state_train, action_train, target_train, _ = load_data(train_data_path, flatten=True)

state_test, action_test, target_test, _ = load_data(test_data_path)


print("Data loaded.")

optimizer = torch.optim.Adam(ode.parameters(), lr=training_param["lr"])
T_0 = 25
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0)
print(f"lr = {training_param['lr']}")
print("Start training...")
train_ode(
    model=ode,
    criterion=criterion,
    state_train=state_train,
    action_train=action_train,
    target_train=target_train,
    state_test=state_test,
    action_test=action_test,
    target_test=target_test,
    q_dim=q_dim,
    v_dim=v_dim,
    Visual_data=None,
    Visual_data_test=None,
    device=DEVICE,
    optimizer=optimizer,
    scheduler=scheduler,
    model_param=model_param,
    dr_config=dr_config,
    **training_param,
)
