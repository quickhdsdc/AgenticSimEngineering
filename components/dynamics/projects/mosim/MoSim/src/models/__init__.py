from .modules import ResidualNet, MModule, MLP
from .dynamics import Dynamic_model_rigid_body, Dynamic_model_resnet
from .bijectors import ODEBijector


def initialize_model(v_dim, action_dim, real_t, activation, is_atten=False, **kwargs):
    q_dim = kwargs.get("q_dim", v_dim)
    quat_config = kwargs.get("quat_config", None)
    manually_int_q = kwargs.get("manually_int_q", False)
    dynamic_model_type = kwargs.get("dynamic_model_type", "rigid_body")
    crt_num = kwargs.get("crt_num", 0)
    integrator = kwargs.get("integrator", "dopri5")
    is_norm = kwargs.get("is_norm", False)
    correctors = []
    if crt_num > 0:
        crt_hidden_block_num, crt_network_width = (
            kwargs["crt_hidden_block_num"],
            kwargs["crt_network_width"],
        )
        correctors = [
            ResidualNet(
                q_dim + v_dim + action_dim,
                v_dim,
                crt_hidden_block_num,
                crt_network_width,
                activation,
                is_atten,
            )
            for _ in range(crt_num)
        ]

    if dynamic_model_type == "rigid_body":
        dynamic_model_rb_params = [
            "m_hidden_block_num",
            "m_network_width",
            "b_hidden_block_num",
            "b_network_width",
            "act_network_type",
            "act_hidden_block_num",
            "act_network_width",
        ]
        (
            m_hidden_block_num,
            m_network_width,
            b_hidden_block_num,
            b_network_width,
            act_network_type,
            act_hidden_block_num,
            act_network_width,
        ) = map(kwargs.get, dynamic_model_rb_params)
        M = MModule(
            q_dim,
            v_dim,
            m_hidden_block_num,
            m_network_width,
            activation,
            is_atten,
            is_norm,
        )
        B = ResidualNet(
            q_dim + v_dim,
            v_dim,
            b_hidden_block_num,
            b_network_width,
            activation,
            is_norm=is_norm,
        )
        if act_network_type == "MLP":
            action_module = MLP(
                action_dim,
                v_dim,
                act_hidden_block_num,
                act_network_width,
                activation,
                norm=is_norm,
            )
        elif act_network_type == "ResidualNetwork":
            action_module = ResidualNet(
                action_dim,
                v_dim,
                act_hidden_block_num,
                act_network_width,
                activation,
                is_norm=is_norm,
            )
        else:
            action_module = None
        dynamic_model = Dynamic_model_rigid_body(
            q_dim,
            v_dim,
            action_dim,
            M,
            B,
            action_module,
            correctors,
            quat_config=quat_config,
        )
    elif dynamic_model_type == "resnet":
        dynamic_model_rn_params = [
            "dynamics_hidden_block_num",
            "dynamics_network_width",
        ]
        dynamics_hidden_block_num, dynamics_network_width = map(
            kwargs.get, dynamic_model_rn_params
        )
        dynamic_model = Dynamic_model_resnet(
            q_dim,
            v_dim,
            action_dim,
            dynamics_hidden_block_num,
            dynamics_network_width,
            activation,
            is_atten,
            correctors,
            quat_config=quat_config,
        )
    else:
        raise ValueError(
            f"Unknown dynamic model type: {dynamic_model_type}. "
            "Please choose from ['rigid_body', 'resnet']."
        )

    model = ODEBijector(
        dynamic_model,
        q_dim,
        v_dim,
        real_t,
        integrator=integrator,
        manually_int_q=manually_int_q,
    )
    return model
