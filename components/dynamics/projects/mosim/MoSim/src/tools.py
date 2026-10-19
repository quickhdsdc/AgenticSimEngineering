from pathlib import Path
import os
import sys
import shutil
import tempfile
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from matplotlib import pyplot as plt
import yaml
import pytz
import wandb

from .models import initialize_model
import normflows as nf


def shuffle_data(data):
    """
    Shuffles the input data and returns the shuffled result.

    Parameters:
    - data: Input tensor of shape (num_samples, feature_dim).

    Returns:
    - shuffled_data: Shuffled tensor with the same shape as the input.
    """
    # Generate random indices for shuffling the data
    shuffle_indices = torch.randperm(data.size(0))

    # Shuffle the data according to the generated random indices
    shuffled_data = data[shuffle_indices]

    return shuffled_data


def estimate_sample_size(tensor_dataset, epsilon=0.01, initial_samples=1000):
    mse_list = []
    batch_size = tensor_dataset.size(0)
    cirtertion = torch.nn.MSELoss()

    for _ in range(initial_samples):
        idx1, idx2 = torch.randint(0, batch_size, (2,))
        while idx1 == idx2:
            idx2 = torch.randint(0, batch_size, (1,)).item()

        state1, state2 = tensor_dataset[idx1], tensor_dataset[idx2]
        mse = cirtertion(state1, state2).item()
        mse_list.append(mse)

    sigma = np.std(mse_list)

    n = int((sigma / epsilon) ** 2)
    print(f"Required number of samples:{n}")
    return n


def calculate_confidence_interval(tensor_dataset, sample_num, sem):
    """
    Calculates the 95% confidence interval based on batch sampling and standard error.

    Parameters:
    - tensor_dataset: A tensor of shape (num_samples, feature_dim) representing the dataset.
    - sample_num: The number of samples to draw.
    - sem: The standard error of the mean (SEM) calculated from the samples.

    Returns:
    - ci_lower: Lower bound of the 95% confidence interval.
    - ci_upper: Upper bound of the 95% confidence interval.
    """

    total_size = tensor_dataset.size(0)

    if total_size < 2 * sample_num:
        raise ValueError(
            "The dataset is too small to generate enough sample pairs. Please reduce sample_num or increase the dataset size."
        )

    indices = torch.randint(0, total_size, (2 * sample_num,))
    idx1, idx2 = indices[:sample_num], indices[sample_num:]

    state1, state2 = tensor_dataset[idx1], tensor_dataset[idx2]

    loss_fn = torch.nn.MSELoss(reduction="none")
    mse = loss_fn(state1, state2).mean(dim=1)

    mse_mean = mse.mean().item()
    mse_std = mse.std().item()

    ci_lower = mse_mean - 1.96 * sem
    ci_upper = mse_mean + 1.96 * sem

    print(f"95% Confidence Interval:[{ci_lower}, {ci_upper}]")
    return ci_lower, ci_upper


def load_data(path, device="cpu", flatten=False):
    with np.load(path, allow_pickle=True) as f:
        data = f["data"]
        metadata = f["metadata"].item()

    tensor_data = torch.tensor(data, dtype=torch.float32, device=device)

    if flatten:
        tensor_data = tensor_data.flatten(0, 1)
    nq, nv = metadata["nq"], metadata["nv"]
    state = tensor_data[..., : nq + nv]
    action = tensor_data[..., nq + nv : -nq - nv]
    target = tensor_data[..., -nq - nv :]
    return state, action, target, metadata


def generate_multistep_test_set(
    test_data_path, save_path, horizon, evaluate_num_per_seq, seed=10792221
):
    rng = np.random.RandomState(seed)

    state, action, target, metadata = load_data(test_data_path, device="cpu")
    data = torch.cat((state, action, target), dim=-1)
    seq_num = data.shape[0]
    seq_length = data.shape[1]

    begin = rng.randint(0, seq_length - horizon, size=(seq_num, evaluate_num_per_seq))
    begin = torch.from_numpy(begin)

    index = begin.unsqueeze(-1) + torch.arange(horizon).unsqueeze(0).unsqueeze(0)

    expanded_data = data.unsqueeze(1).expand(-1, evaluate_num_per_seq, -1, -1)
    expanded_index = index.unsqueeze(-1).expand(-1, -1, -1, data.shape[-1])

    ms_test_set = torch.gather(expanded_data, 2, expanded_index)
    ms_test_set = ms_test_set.reshape(
        seq_num * evaluate_num_per_seq, horizon, data.shape[-1]
    )
    ms_test_set = ms_test_set[torch.from_numpy(rng.permutation(ms_test_set.shape[0]))]

    ms_test_data = np.array(ms_test_set, dtype=np.float32)
    metadata["horizon"] = horizon
    metadata["seed"] = seed
    np.savez_compressed(save_path, data=ms_test_data, metadata=metadata)


def load_ode_from_ckpt(ckpt_path, override_corrector_configs=None, load_parameter=True):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if override_corrector_configs:
        if (
            checkpoint["model_param"]["crt_num"]
            == override_corrector_configs["crt_num"]
            and checkpoint["model_param"]["crt_hidden_block_num"]
            == override_corrector_configs["crt_hidden_block_num"]
            and checkpoint["model_param"]["crt_network_width"]
            == override_corrector_configs["crt_network_width"]
        ):
            pass
        else:
            checkpoint["model_param"]["crt_num"] = override_corrector_configs["crt_num"]
            checkpoint["model_param"]["crt_hidden_block_num"] = (
                override_corrector_configs["crt_hidden_block_num"]
            )
            checkpoint["model_param"]["crt_network_width"] = override_corrector_configs[
                "crt_network_width"
            ]
            print(f"Overrided corrector configs to {override_corrector_configs}.")
    ode = initialize_model(**checkpoint["model_param"])
    if load_parameter:
        ode.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        return ode
    return ode


def load_ode_from_ckpt_with_param(ckpt_path, load_parameter=True):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ode = initialize_model(**checkpoint["model_param"])
    if load_parameter == True:
        ode.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        return ode, checkpoint["model_param"]
    return ode, checkpoint["model_param"]


def initialize_ode_from_config(config_path):
    abs_path = os.path.abspath(config_path)
    if not os.path.exists(abs_path):
        print("Path doesn't exist.")
        sys.exit(1)
    with open(abs_path, "r") as f:
        config = yaml.safe_load(f)
    model_param = config["model"]
    ode = initialize_model(**model_param)
    return ode


def rebalance_data(
    criterion_model,
    state,
    action,
    target,
    t,
    criterion,
    device,
    alpha=0.2,
    allocation_rate=[1, 1],
    batch_size=300000,
):
    """
    Rebalances the dataset by splitting it into easy and hard samples
    based on prediction loss, and applying different resampling rates.

    Requirements:
    - `action` and `target` must be flattened and on CPU.

    Parameters:
    - criterion_model: The model used to compute loss.
    - state, action, target: Input tensors.
    - t: Time step or context (passed to model).
    - criterion: Loss function.
    - device: CUDA or CPU device.
    - alpha: Float in [0, 1], determines the proportion of easiest data.
    - allocation_rate: List of 2 ints ≥ 0, [easy_data_rate, hard_data_rate].
    - batch_size: Batch size for evaluation.

    Returns:
    - Rebalanced state, action, target tensors.
    """
    criterion_model = criterion_model.to(device)
    criterion_model.eval()
    all_losses = []

    dataset = torch.utils.data.TensorDataset(state, action, target)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    # Temporary file for storing intermediate batch losses
    temp_file = tempfile.NamedTemporaryFile(delete=False)

    with torch.no_grad():
        for i, (x, a, tgt) in enumerate(dataloader):
            x, a, tgt = x.to(device), a.to(device), tgt.to(device)

            output = criterion_model(x, a, 0.0, t)
            losses = torch.mean(criterion(output, tgt), dim=1).cpu()

            # Save batch losses to temporary disk files
            torch.save(losses, temp_file.name + f"_part_{i}.pt")

            # Free up GPU memory
            del x, a, tgt, output, losses
            torch.cuda.empty_cache()

    # Load all batch losses from temporary files
    all_losses = torch.cat(
        [torch.load(temp_file.name + f"_part_{i}.pt") for i in range(len(dataloader))],
        dim=0,
    )
    print("All losses shape:", all_losses.shape)

    final_loss_file = "final_all_losses.pt"
    torch.save(all_losses, final_loss_file)
    print(f"All losses have been saved to: {final_loss_file}")

    # Remove temporary files
    for i in range(len(dataloader)):
        file_path = temp_file.name + f"_part_{i}.pt"
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted temporary file: {file_path}")

    print("Temporary files removed successfully.")

    sorted_indices = torch.argsort(all_losses)
    cut_idx = int(alpha * state.shape[0])

    def repeat_or_sample(tensor, rate):
        """
        Resample the tensor based on the given rate:
        - If rate >= 1: duplicate the data.
        - If rate < 1: randomly sample a proportion of the data.
        """
        if rate >= 1:
            return tensor.repeat(int(rate), 1)
        else:
            num_samples = int(len(tensor) * rate)
            indices = torch.randperm(len(tensor))[:num_samples]
            return tensor[indices]

    def rebalance(tensor):
        tensor = tensor[sorted_indices]
        head = tensor[:cut_idx]
        tail = tensor[cut_idx:]
        return torch.cat(
            (
                repeat_or_sample(head, allocation_rate[0]),
                repeat_or_sample(tail, allocation_rate[1]),
            ),
            dim=0,
        )

    return rebalance(state).cpu(), rebalance(action).cpu(), rebalance(target).cpu()


def train_ode(
    model,
    criterion,
    state_train,
    action_train,
    target_train,
    state_test,
    action_test,
    target_test,
    q_dim,
    v_dim,
    t,
    training_stage,
    num_epochs,
    optimizer,
    scheduler=None,
    batch_size=10000,
    task_name="",
    save_path="",
    from_ckpt=None,
    continue_training=True,
    dr_config=None,
    model_param=None,
    keep_top=5,
    multistep_test=False,
    device=torch.device("cpu"),
    is_ode=True,
    **kwargs,
):

    epoch = 0
    step = 0
    time_shift = 0
    top_ckpt_list = []
    checkpoint_dir = ""
    train_time = datetime.now(pytz.timezone("US/Pacific")).strftime("%m%d%H%M")
    start_time = datetime.now()
    loss = torch.tensor(0.0)
    elapsed_time = 0.0

    if from_ckpt:
        checkpoint = torch.load(from_ckpt, map_location=device, weights_only=False)
        epoch = checkpoint["epoch"]
        step = checkpoint["step"]
        time_shift = checkpoint.get("elapsed_time", 0)
        if continue_training:
            print("Optimizer loaded. Training continues.")
            checkpoint_dir = checkpoint["checkpoint_dir"]

    model.to(device)
    if is_ode:
        model.dynamic_model.training_stage = training_stage
    model.train()

    wandb.init(project="MoSim", name=f"{task_name}_{train_time}", config=model_param)

    train_dataset = torch.utils.data.TensorDataset(
        state_train, action_train, target_train
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    state_test = state_test.to(device)
    action_test = action_test.to(device)
    target_test = target_test.to(device)

    while True:
        for x, a, target in train_dataloader:
            optimizer.zero_grad()
            x, a, target = x.to(device), a.to(device), target.to(device)
            output = model(x, a, 0.0, t) if is_ode else model(x, a)
            losses = criterion(output, target)
            loss = losses.mean()
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step(epoch + step / len(train_dataloader))

            step += batch_size
            elapsed_time = (
                time_shift + (datetime.now() - start_time).total_seconds() / 3600
            )

            log_data = {
                "Overall Training Loss": loss.item(),
                "Overall Training Loss per hour": loss.item(),
            }
            for dim in range(losses.shape[1]):
                name = f"Q/q{dim}" if dim < q_dim else f"V/v{dim - q_dim}"
                log_data[f"Training Loss {name}"] = losses[:, dim].mean().item()
                log_data[f"Training Loss per hour {name}"] = (
                    losses[:, dim].mean().item()
                )
            wandb.log(log_data, step=step)

        model.eval()
        with torch.no_grad():
            if not multistep_test:
                state_test_temp = state_test.flatten(0, 1)
                action_test_temp = action_test.flatten(0, 1)
                target_test_temp = target_test.flatten(0, 1)
                prediction_test = (
                    model(state_test_temp, action_test_temp, 0.0, t)
                    if is_ode
                    else model(state_test_temp, action_test_temp)
                )
                losses_test = criterion(prediction_test, target_test_temp)
                loss_test = losses_test.mean()
            else:
                prediction_test = []
                current_state = state_test[:, 0]
                for n in range(state_test.shape[1]):
                    next_state = model(current_state, action_test[:, n], 0.0, t)
                    prediction_test.append(next_state)
                    current_state = next_state
                prediction_test = torch.stack(prediction_test).permute(1, 0, 2)
                losses_test = torch.nn.functional.mse_loss(
                    prediction_test, target_test, reduction="none"
                )
                loss_test = losses_test.mean()

            test_log_data = {
                "Overall Test Loss": loss_test.item(),
                "Overall Test Loss per hour": loss_test.item(),
            }
            for dim in range(losses_test.shape[-1]):
                name = f"Q/q{dim}" if dim < q_dim else f"V/v{dim - q_dim}"
                test_log_data[f"Test Loss {name}"] = losses_test[..., dim].mean().item()
                test_log_data[f"Test Loss per hour {name}"] = (
                    losses_test[..., dim].mean().item()
                )
            wandb.log(test_log_data, step=step)

        epoch += 1
        model.train()

        checkpoint = {
            "task_name": task_name,
            "epoch": epoch,
            "step": step,
            "elapsed_time": elapsed_time,
            "training_stage": training_stage,
            "training_set_loss": loss.item(),
            "test_set_loss": loss_test.item(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "checkpoint_dir": checkpoint_dir,
            "model_param": model_param,
            "from_ckpt": from_ckpt,
            "dr_config": dr_config,
        }

        if checkpoint_dir == "":
            checkpoint_dir = os.path.join(save_path, f"{task_name}_{train_time}")
            os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_filename = f"ckpt_{int(step/1000000)}M.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
        torch.save(checkpoint, checkpoint_path)

        top_ckpt_list.append((loss_test.item(), checkpoint_path))
        top_ckpt_list.sort()
        top_ckpt_list = top_ckpt_list[:keep_top]

        latest_path = os.path.join(
            checkpoint_dir, f"latest_ckpt_{int(step/1000000)}M.pth"
        )
        torch.save(checkpoint, latest_path)

        keep_files = {x[1] for x in top_ckpt_list} | {latest_path}
        existing_files = set(
            os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)
        )
        for file in existing_files:
            if file not in keep_files and file.endswith(".pth"):
                os.remove(file)

        sys.stdout.write(f"\rModel saved at epoch {epoch}, step {step/1000000}M.")
        sys.stdout.flush()

        if epoch >= num_epochs:
            break


def scaled_sigmoid(log_prob_x, lower=-40, upper=0):
    center = (upper + lower) / 2
    scale = (upper - lower) / 4

    return 1 / (1 + torch.exp(-(log_prob_x - center) / scale))


def initialize_residual_flow(
    K,
    latent_size,
    hidden_units,
    hidden_layers,
    state_dim,
    device,
    data_path,
    batch_size,
    from_ckpt="",
    lipschitz_cons=0.9,
):
    flows = []
    for i in range(K):
        net = nf.nets.LipschitzMLP(
            [latent_size] + [hidden_units] * (hidden_layers - 1) + [latent_size],
            init_zeros=True,
            lipschitz_const=lipschitz_cons,
        )
        flows += [nf.flows.Residual(net, reduce_memory=True)]
        flows += [nf.flows.ActNorm(latent_size)]

    # Set prior and q0
    q0 = nf.distributions.DiagGaussian(state_dim, trainable=False)

    # Construct flow model
    nfm = nf.NormalizingFlow(q0=q0, flows=flows)
    nfm = nfm.to(device)
    if from_ckpt != "":
        checkpoint = torch.load(from_ckpt, map_location=device)
        nfm.load_state_dict(checkpoint["model_state_dict"])
    else:
        s, a, _, _ = load_data(data_path, flatten=True)
        x = torch.cat((s, a), dim=-1).float().to(device)[:,]
        num_samples = x.shape[0]
        idx = torch.randperm(num_samples).to(device)
        x = x[idx][:batch_size]
        _ = nfm.log_prob(x)
        nfm.inverse_and_log_det
    return nfm


def cal_prob(nfm, state, lower=-40, upper=0):
    x, log_det_J = nfm.inverse_and_log_det(state)

    log_prob_base = nfm.q0.log_prob(x)

    log_prob_x = log_prob_base + log_det_J

    normalized_prob = scaled_sigmoid(log_prob_x, lower=lower, upper=upper) - 1

    return normalized_prob


def train_residual_flow(
    nfm,
    batch_size,
    data_path,
    data_path_test,
    device,
    save_path,
    task_name,
    lr=3e-3,
    keep_top=5,
):

    # Load and prepare training data
    state_train, action_train, _, _ = load_data(data_path, flatten=True)
    train_data = torch.cat((state_train, action_train), dim=-1)
    train_dataset = TensorDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize optimizer
    optimizer = torch.optim.Adam(nfm.parameters(), lr=lr, weight_decay=1e-5)

    # Load test data
    state_test, action_test, _, _ = load_data(data_path_test, flatten=True)
    test_data = torch.cat((state_test, action_test), dim=-1).to(device)

    step = 0
    epoch = 0
    loss = torch.tensor(0.0)
    loss_test = torch.tensor(0.0)
    checkpoint_dir = ""
    top_ckpt_list = []
    train_time = datetime.now(pytz.timezone("US/Pacific")).strftime("%m%d%H%M")

    wandb.init(project="residual-flow", name=f"{task_name}_{train_time}")

    while True:
        for target in train_dataloader:
            optimizer.zero_grad()
            x = target[0].to(device)

            # Compute loss
            loss = nfm.forward_kld(x)

            # Backpropagation only if loss is valid
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
                optimizer.step()

            # Update Lipschitz constants
            nf.utils.update_lipschitz(nfm, 50)

            step += batch_size
            wandb.log({"Overall Training Loss": loss.item()}, step=step)

        # Evaluation
        nfm.eval()
        with torch.no_grad():
            loss_test = nfm.forward_kld(test_data)
            wandb.log({"Overall Test Loss": loss_test.item()}, step=step)
        nfm.train()

        epoch += 1

        # Save checkpoint
        if checkpoint_dir == "":
            checkpoint_dir = os.path.join(save_path, f"{task_name}_{train_time}")
            os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "task_name": task_name,
            "epoch": epoch,
            "step": step,
            "training_set_loss": loss.item(),
            "test_set_loss": loss_test.item(),
            "model_state_dict": nfm.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "checkpoint_dir": checkpoint_dir,
        }

        checkpoint_filename = f"ckpt_{int(step/1000000)}M.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
        torch.save(checkpoint, checkpoint_path)

        top_ckpt_list.append((loss_test.item(), checkpoint_path))
        top_ckpt_list.sort()
        top_ckpt_list = top_ckpt_list[:keep_top]

        latest_path = os.path.join(
            checkpoint_dir, f"latest_ckpt_{int(step/1000000)}M.pth"
        )
        torch.save(checkpoint, latest_path)

        keep_files = {x[1] for x in top_ckpt_list} | {latest_path}
        existing_files = set(
            os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)
        )
        for file in existing_files:
            if file not in keep_files and file.endswith(".pth"):
                os.remove(file)

        sys.stdout.write(f"\rModel saved at epoch {epoch}, step {step/1000000}M.")
        sys.stdout.flush()
        nfm.train()


def test(
    ode,
    data_path,
    start,
    end,
    step_length,
    device,
    single_step=False,
    ground_truth_interval=-1,
):
    ode.to(device)
    ode.eval()
    prob_list = []

    state, action, target, _ = load_data(data_path, device=device)
    with torch.no_grad():
        if single_step == False:
            prediction = []
            current_state = state[:, start]
            for n in range(start, end):
                if (
                    n != start
                    and ground_truth_interval > 0
                    and n % ground_truth_interval == 0
                ):
                    current_state = state[:, n]

                next_state = ode(current_state, action[:, n], 0.0, step_length)
                prediction.append(next_state.cpu().numpy())
                print(next_state.shape)
                current_state = next_state  # prepare for the next step

            prediction = np.array(prediction)
            prediction = np.transpose(prediction, (1, 0, 2))

        else:
            prediction = (
                ode(
                    state[:, start:end, :].flatten(0, 1),
                    action[:, start:end, :].flatten(0, 1),
                    0.0,
                    step_length,
                )
                .reshape(state[:, start:end, :].shape)
                .cpu()
                .numpy()
            )

        ground_truth = target[:, start:end, :].cpu().numpy()
    criterion = torch.nn.MSELoss(reduction="none")
    loss = criterion(torch.tensor(prediction).to(target), target[:, start:end, :])
    return ground_truth, prediction, loss, prob_list


def test_multistep(ode, data_path, step_length, device, ground_truth_interval=-1):
    state, action, target, metadata = load_data(data_path, device=device)
    horizon = metadata["horizon"]
    ode.to(device)
    ode.eval()
    with torch.no_grad():
        prediction = []
        current_state = state[:, 0]
        for n in range(horizon):
            if n != 0 and ground_truth_interval > 0 and n % ground_truth_interval == 0:
                current_state = state[:, n]
            next_state = ode(current_state, action[:, n], 0.0, step_length)
            prediction.append(next_state.cpu())
            current_state = next_state
    prediction = np.array(prediction)
    prediction = np.transpose(prediction, (1, 0, 2))
    criterion = torch.nn.MSELoss()
    loss = criterion(torch.tensor(prediction).to(target), target)
    ground_truth = target.cpu().numpy()
    return ground_truth, prediction, loss.item(), metadata


def AddNoise(data_path, saved_path, device="cuda:6", std=0.01):
    state, action, target, metadata = load_data(data_path, device=device)
    target = target.cpu().numpy()
    action = action.cpu().numpy()
    state = state.cpu().numpy()
    fixed_std = std
    data = np.concatenate((state, np.expand_dims(target[:, -1, :], axis=1)), axis=1)
    noise = np.random.normal(loc=0, scale=fixed_std, size=data.shape)

    noisy_data = data + noise
    state = noisy_data[:, :-1, :]
    target = noisy_data[:, 1:, :]
    noisy_data = np.concatenate((state, action, target), axis=2)
    np.savez_compressed(saved_path, data=noisy_data, metadata=metadata)

    print(f"Data saved to {saved_path}")

    return noisy_data


def draw(
    name,
    *xs,
    plt_type="plot",
    size=0.5,
    marker="o",
    labels=None,
    settings="",
    alpha=1,
    monochrome=False,
):
    plt.figure(figsize=(10, 5))
    time_steps = list(range(len(xs[0])))
    for i in range(len(xs)):
        if plt_type == "plot":
            if labels:
                plt.plot(
                    time_steps,
                    xs[i],
                    marker=marker,
                    label=labels[i],
                    alpha=alpha,
                    color=monochrome if monochrome else None,
                )
            else:
                plt.plot(
                    time_steps,
                    xs[i],
                    marker=marker,
                    alpha=alpha,
                    color=monochrome if monochrome else None,
                )
        elif plt_type == "scatter":
            if labels:
                plt.scatter(
                    time_steps,
                    xs[i],
                    marker=marker,
                    label=labels[i],
                    alpha=alpha,
                    color=monochrome if monochrome else None,
                )
            else:
                print("hello", len(time_steps), len(xs[i]))
                plt.scatter(
                    time_steps,
                    xs[i],
                    marker=marker,
                    s=size,
                    alpha=alpha,
                    color=monochrome if monochrome else None,
                )
    plt.title(f"{name} Over Time" + settings)
    plt.xlabel("Time Step")
    plt.ylabel(f"{name}")
    plt.grid(True)
    if labels:
        plt.legend()
    plt.show()


def data_processing(data_path, start, pred_horizen, device):
    inputs, actions, targets, _ = load_data(data_path, flatten=False)
    inputs = inputs[:, start : start + pred_horizen, :].to(device).float()
    actions = actions[:, start : start + pred_horizen, :].to(device).float()
    targets = targets[:, start : start + pred_horizen, :].to(device).float()
    is_first = torch.zeros(inputs.shape[0], pred_horizen, 1, dtype=torch.bool).to(
        device
    )
    is_first[:, 0, :] = True

    data_dict = {
        "inputs": inputs,
        "actions": actions,
        "targets": targets,
        "is_first": is_first,
    }
    return data_dict


def render_frames_comparison(
    ode,
    data_path,
    sequence_index,
    start,
    end,
    step_length,
    env,
    save_path,
    ground_truth_interval=-1,
    device="cuda:0",
    render_interval=1,
    pred_folder=str(Path(__file__).resolve().parents[2] / 'videos/temp/prediction_frames') + '/',
    gt_folder=str(Path(__file__).resolve().parents[2] / 'videos/temp/ground_truth_frames') + '/',
    delet_pred_folder=False,
    delet_gt_folder=False,
    is_Dreamer=False,
):
    if delet_pred_folder and os.path.exists(pred_folder):
        shutil.rmtree(pred_folder)
        print(f"Cleared contents of prediction folder: {pred_folder}")
    if delet_gt_folder and os.path.exists(gt_folder):
        shutil.rmtree(gt_folder)
        print(f"Cleared contents of ground truth folder: {gt_folder}")

    os.makedirs(pred_folder, exist_ok=True)
    os.makedirs(gt_folder, exist_ok=True)

    state, action, target, metadata = load_data(data_path, device=device)
    q_dim = metadata["nq"]
    v_dim = metadata["nv"]
    state, action, target = (
        state[sequence_index],
        action[sequence_index],
        target[sequence_index],
    )
    target = target.cpu().numpy()
    ode.to(device)
    ode.eval()

    if not delet_pred_folder:
        print("Skipping prediction as delet_pred_folder is set to False.")
        with torch.no_grad():
            for n in range(start, end):
                try:
                    env._env.physics.data.qpos = target[n, :q_dim]
                    env._env.physics.data.qvel = target[n, q_dim:]
                    env._env.physics.forward()
                    gt_frame = env._env.physics.render(
                        camera_id=0, width=640, height=480
                    )

                except ValueError as e:
                    print(f"Warning: {e}. Failed to render ground truth for step {n}.")
                    continue

                gt_image = Image.fromarray(gt_frame)
                gt_image_path = os.path.join(gt_folder, f"ground_truth_{n:04d}.png")
                gt_image.save(gt_image_path)
                print(f"Saved ground truth image: {gt_image_path}")
        return

    comparison_images = []
    overlay_images = []

    buffer_width = env._env.physics.model.vis.global_.offwidth or 640
    buffer_height = env._env.physics.model.vis.global_.offheight or 480

    render_width = min(1840, buffer_width)
    render_height = min(1024, buffer_height)

    with torch.no_grad():
        current_state = state[start]  # initialize at first step
        for n in range(start, end):
            if (
                n != start
                and ground_truth_interval > 0
                and n % ground_truth_interval == 0
            ):
                current_state = state[n]

            if n != start and n != end and n % render_interval != 0 and not is_Dreamer:
                action_ = (
                    action[n].unsqueeze(0)
                    if current_state.ndim == 2
                    else action.unsqueeze(0)
                )
                current_state = ode(current_state, action_, 0.0, step_length)
                continue

            prediction = current_state.cpu().numpy()
            if prediction.ndim == 2:
                prediction = prediction.squeeze(0)

            try:
                env._env.physics.data.qpos = prediction[:q_dim]
                env._env.physics.data.qvel = prediction[q_dim:]
                env._env.physics.forward()
                pred_frame = env._env.physics.render(
                    camera_id=0, width=render_width, height=render_height
                )
            except ValueError as e:
                print(f"Warning: {e}. Reducing resolution.")
                render_width = buffer_width
                render_height = buffer_height
                pred_frame = env._env.physics.render(
                    camera_id=0, width=render_width, height=render_height
                )

            pred_image = Image.fromarray(pred_frame)
            pred_image_path = os.path.join(pred_folder, f"prediction_{n:04d}.png")
            pred_image.save(pred_image_path)
            print(f"Saved prediction image: {pred_image_path}")

            try:
                env._env.physics.data.qpos = target[n, :q_dim]
                env._env.physics.data.qvel = target[n, q_dim:]
                env._env.physics.forward()
                gt_frame = env._env.physics.render(
                    camera_id=0, width=render_width, height=render_height
                )
            except ValueError as e:
                print(f"Warning: {e}. Reducing resolution.")
                render_width = buffer_width
                render_height = buffer_height
                gt_frame = env._env.physics.render(
                    camera_id=0, width=render_width, height=render_height
                )

            gt_image = Image.fromarray(gt_frame)
            gt_image_path = os.path.join(gt_folder, f"ground_truth_{n:04d}.png")
            gt_image.save(gt_image_path)

            comparison_image = Image.new(
                "RGB", (render_width, render_height * 2), (255, 255, 255)
            )
            comparison_image.paste(gt_image, (0, 0))
            comparison_image.paste(pred_image, (0, render_height))
            comparison_images.append(comparison_image)

            alpha = 0.5
            overlay_image = Image.blend(gt_image, pred_image, alpha)
            overlay_images.append(overlay_image)

            # ODE step for next round
            if current_state.ndim == 1:
                current_state = current_state.unsqueeze(0)
                action_ = action[n].unsqueeze(0)
            else:
                action_ = action[n].unsqueeze(0)

            current_state = ode(current_state, action_, 0.0, step_length)


def render_frames_comparison_myo(
    ode,
    data_path,
    sequence_index,
    start,
    end,
    step_length,
    env,
    save_path,
    ground_truth_interval=-1,
    device="cuda:0",
    render_interval=1,
    pred_folder=str(Path(__file__).resolve().parents[2] / 'videos/temp/prediction_frames') + '/',
    gt_folder=str(Path(__file__).resolve().parents[2] / 'videos/temp/ground_truth_frames') + '/',
    delet_pred_folder=False,
    delet_gt_folder=False,
    is_Dreamer=False,
):
    if delet_pred_folder and os.path.exists(pred_folder):
        shutil.rmtree(pred_folder)
        print(f"Cleared contents of prediction folder: {pred_folder}")
    if delet_gt_folder and os.path.exists(gt_folder):
        shutil.rmtree(gt_folder)
        print(f"Cleared contents of ground truth folder: {gt_folder}")

    os.makedirs(pred_folder, exist_ok=True)
    os.makedirs(gt_folder, exist_ok=True)

    state, action, target, metadata = load_data(data_path, device=device)
    q_dim = metadata["nq"]
    state, action, target = (
        state[sequence_index],
        action[sequence_index],
        target[sequence_index],
    )
    target = target.cpu().numpy()
    ode.to(device)
    ode.eval()

    if not delet_pred_folder:
        print("Skipping prediction as delet_pred_folder is set to False.")
        with torch.no_grad():
            for n in range(start, end):
                try:
                    env.unwrapped.sim.data.qpos = target[n, :q_dim]
                    env.unwrapped.sim.data.qvel = target[n, q_dim:]
                    env.unwrapped.sim.forward()
                    gt_frame = env.render()

                except ValueError as e:
                    print(f"Warning: {e}. Failed to render ground truth for step {n}.")
                    continue

                gt_image = Image.fromarray(gt_frame)
                gt_image_path = os.path.join(gt_folder, f"ground_truth_{n:04d}.png")
                gt_image.save(gt_image_path)
                print(f"Saved ground truth image: {gt_image_path}")
        return

    comparison_images = []
    overlay_images = []

    buffer_width = env.unwrapped.sim.model.vis.global_.offwidth or 640
    buffer_height = env.unwrapped.sim.model.vis.global_.offheight or 480

    render_width = min(1840, buffer_width)
    render_height = min(1024, buffer_height)
    print(render_width, render_height)
    with torch.no_grad():
        current_state = state[start]

        for n in range(start, end):
            if (
                n != start
                and ground_truth_interval > 0
                and n % ground_truth_interval == 0
            ):
                current_state = state[n]

            if n != start and n != end and n % render_interval != 0 and not is_Dreamer:
                action_ = (
                    action[n].unsqueeze(0)
                    if current_state.ndim == 2
                    else action.unsqueeze(0)
                )
                current_state = ode(current_state, action_, 0.0, step_length)
                continue

            prediction = current_state.cpu().numpy()
            if prediction.ndim == 2:
                prediction = prediction.squeeze(0)

            try:
                env.unwrapped.sim.data.qpos = prediction[:q_dim]
                env.unwrapped.sim.data.qvel = prediction[q_dim:]
                env.unwrapped.sim.forward()
                pred_frame = env.render()
            except ValueError as e:
                print(f"Warning: {e}. Reducing resolution.")
                render_width = buffer_width
                render_height = buffer_height
                pred_frame = env.render()

            pred_image = Image.fromarray(pred_frame)
            pred_image_path = os.path.join(pred_folder, f"prediction_{n:04d}.png")
            pred_image.save(pred_image_path)
            print(f"Saved prediction image: {pred_image_path}")

            try:
                env.unwrapped.sim.data.qpos = target[n, :q_dim]
                env.unwrapped.sim.data.qvel = target[n, q_dim:]
                env.unwrapped.sim.forward()
                gt_frame = env.render()
            except ValueError as e:
                print(f"Warning: {e}. Reducing resolution.")
                render_width = buffer_width
                render_height = buffer_height
                gt_frame = env.render()

            gt_image = Image.fromarray(gt_frame)
            gt_image_path = os.path.join(gt_folder, f"ground_truth_{n:04d}.png")
            gt_image.save(gt_image_path)
            print(f"Saved ground truth image: {gt_image_path}")

            comparison_image = Image.new(
                "RGB", (render_width, render_height * 2), (255, 255, 255)
            )
            comparison_image.paste(gt_image, (0, 0))
            comparison_image.paste(pred_image, (0, render_height))
            comparison_images.append(comparison_image)

            alpha = 0.5
            overlay_image = Image.blend(gt_image, pred_image, alpha)
            overlay_images.append(overlay_image)

            # Prepare next state via ODE model
            if current_state.ndim == 1:
                current_state = current_state.unsqueeze(0)
                action_ = action[n].unsqueeze(0)
            else:
                action_ = action[n].unsqueeze(0)

            current_state = ode(current_state, action_, 0.0, step_length)

        total_width = render_width * len(comparison_images)
        comparison_grid = Image.new(
            "RGB", (total_width, render_height * 2), (255, 255, 255)
        )
        for i, img in enumerate(comparison_images):
            comparison_grid.paste(img, (render_width * i, 0))
        comparison_grid_path = os.path.join(save_path, "comparison_grid.png")
        comparison_grid.save(comparison_grid_path)
        print(f"Saved comparison image: {comparison_grid_path}")

        overlay_grid = Image.new("RGB", (total_width, render_height), (255, 255, 255))
        for i, img in enumerate(overlay_images):
            overlay_grid.paste(img, (render_width * i, 0))
        overlay_grid_path = os.path.join(save_path, "overlay_grid.png")
        overlay_grid.save(overlay_grid_path)
        print(f"Saved overlay image: {overlay_grid_path}")


def blend_images(
    pred_folder,
    gt_folder,
    save_folder=str(Path(__file__).resolve().parents[2] / 'videos/temp/blended_frames') + '/',
    alpha=0.5,
):
    # Ensure the save folder exists, and clear its contents if it does
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)  # Remove all contents in the folder
    os.makedirs(save_folder, exist_ok=True)  # Recreate the empty folder

    # Get sorted list of prediction and ground truth images
    pred_images = sorted([f for f in os.listdir(pred_folder) if f.endswith(".png")])
    gt_images = sorted([f for f in os.listdir(gt_folder) if f.endswith(".png")])

    # Ensure matching number of files in both folders
    if len(pred_images) != len(gt_images):
        print(
            "Error: The number of images in the prediction and ground truth folders does not match."
        )
        return

    # Blend each pair of images
    for pred_img_name, gt_img_name in zip(pred_images, gt_images):
        pred_img_path = os.path.join(pred_folder, pred_img_name)
        gt_img_path = os.path.join(gt_folder, gt_img_name)

        # Open the images
        pred_img = Image.open(pred_img_path)
        gt_img = Image.open(gt_img_path)

        # Blend the images
        blended_img = Image.blend(gt_img, pred_img, alpha)

        # Save the blended image
        blended_img_name = f"blended_{pred_img_name}"
        blended_img_path = os.path.join(save_folder, blended_img_name)
        blended_img.save(blended_img_path)
        print(f"Saved blended image: {blended_img_path}")


def combine_images_sequence(
    pred_folder=str(Path(__file__).resolve().parents[2] / 'videos/temp/prediction_frames') + '/',
    gt_folder=str(Path(__file__).resolve().parents[2] / 'videos/temp/ground_truth_frames') + '/',
    save_path=str(Path(__file__).resolve().parents[2] / 'final_result_final') + '/',
    render_width=640,
    render_height=480,
):
    """
    Combine images from prediction and ground truth folders into a single sequence image.

    Args:
        pred_folder (str): Path to the folder containing prediction images.
        gt_folder (str): Path to the folder containing ground truth images.
        save_path (str): Path to save the combined image.
        render_width (int): Width of each image.
        render_height (int): Height of each image.
    """
    os.makedirs(save_path, exist_ok=True)

    pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith(".png")])
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith(".png")])

    if len(pred_files) != len(gt_files):
        raise ValueError(
            "The number of prediction images and ground truth images must be the same."
        )

    num_images = len(pred_files)
    total_width = render_width * num_images
    combined_height = render_height * 2
    comparison_grid = Image.new("RGB", (total_width, combined_height), (255, 255, 255))

    for i, (pred_file, gt_file) in enumerate(zip(pred_files, gt_files)):
        pred_image_path = os.path.join(pred_folder, pred_file)
        gt_image_path = os.path.join(gt_folder, gt_file)

        pred_image = Image.open(pred_image_path).resize((render_width, render_height))
        gt_image = Image.open(gt_image_path).resize((render_width, render_height))

        combined_image = Image.new(
            "RGB", (render_width, render_height * 2), (255, 255, 255)
        )
        combined_image.paste(gt_image, (0, 0))
        combined_image.paste(pred_image, (0, render_height))

        x_offset = render_width * i
        comparison_grid.paste(combined_image, (x_offset, 0))

    final_image_path = os.path.join(save_path, "combined_comparison_sequence.png")
    comparison_grid.save(final_image_path)
    print(f"Saved combined sequence image: {final_image_path}")
