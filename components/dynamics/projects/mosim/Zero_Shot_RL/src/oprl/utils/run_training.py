import logging
from multiprocessing import Process

from oprl.trainers.base_trainer import BaseTrainer
from oprl.trainers.safe_trainer import SafeTrainer
from oprl.utils.utils import set_seed


def run_training(
    make_algo, make_env, make_env_eval, make_logger, nfm, transition_model, config, seeds: int = 1, start_seed: int = 0
):
    if seeds == 1:
        _run_training_func(make_algo, make_env, make_env_eval, make_logger,transition_model, config,nfm, 0)
    else:
        processes = []
        for seed in range(start_seed, start_seed + seeds):
            processes.append(
                Process(
                    target=_run_training_func,
                    args=(make_algo, make_env, make_env_eval, make_logger,transition_model, config,nfm, seed),
                )
            )

        for i, p in enumerate(processes):
            p.start()
            logging.info(f"Starting process {i}...")

        for p in processes:
            p.join()

        logging.info("Training OK.")


def _run_training_func(make_algo, make_env, make_env_eval, make_logger,transition_model, config, nfm,seed: int):
    set_seed(seed)
    env = make_env(seed=seed,  ckpt_path=config["ckpt_path"], dt=config["dt"], step_limit=config["step_limit"], DEVICE=config["device"], nfm=nfm)
    logger = make_logger(seed)

    if env.env_family == "dm_control" or env.env_family == "MoSim":
        trainer_class = BaseTrainer
    elif env.env_family == "safety_gymnasium":
        trainer_class = SafeTrainer
    else:
        raise ValueError(f"Unsupported env family: {env.env_family}")

    trainer = trainer_class(
        transition_model=transition_model,
        real_t=config["dt"],
        horizen = config["horizen"],
        state_dim=config["state_dim"],
        action_dim=config["action_dim"],
        env=env,
        make_env_test=make_env_eval,
        algo=make_algo(logger),
        num_steps=config["num_steps"],
        eval_interval=config["eval_every"],
        device=config["device"],
        save_buffer_every=config["save_buffer"],
        visualise_every=config["visualise_every"],
        estimate_q_every=config["estimate_q_every"],
        stdout_log_every=config["log_every"],
        seed=seed,
        logger=logger,
        max_episode_len=config["step_limit"]
    )

    trainer.train()
