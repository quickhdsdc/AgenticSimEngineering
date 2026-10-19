import torch
from torch import nn
from .modules import ResidualNet


class Dynamic_model_rigid_body(nn.Module):
    def __init__(
        self,
        q_dim,
        v_dim,
        action_dim,
        M_module,
        B_module,
        action_module,
        correctors=[],
        quat_config=None,
    ):
        """
        quat_config: int tuple (q_quat_beging_idx, v_omega_begin_idx)
        """
        super().__init__()
        self.q_dim = q_dim
        self.v_dim = v_dim
        self.action_dim = action_dim
        self.M = M_module
        self.B = B_module
        if self.action_dim == 0:
            self.action_module = None
        else:
            self.action_module = action_module
        self.correctors = nn.ModuleList(correctors)
        self.quat_config = quat_config
        self.action = None
        self.training_stage = -1

    def forward(self, t, x):
        q = x[:, : self.q_dim]
        v = x[:, self.q_dim : -self.action_dim]
        action = x[:, -self.action_dim :]
        x = x[:, : -self.action_dim]
        if action == None:
            raise Exception("Action not set!")
        if self.action_module == None:
            kinetic_intermediate_term = self.B(x)
        else:
            kinetic_intermediate_term = self.B(x) + self.action_module(action)
        kinetic_intermediate_term = kinetic_intermediate_term.unsqueeze(2)
        v_rate = torch.bmm(self.M(q), kinetic_intermediate_term).squeeze(-1)

        if self.training_stage != 0:
            corrector_input = torch.cat((x, action), dim=1)
            if self.training_stage == -1:  # integral training.
                for i in range(len(self.correctors)):
                    v_rate += self.correctors[i](corrector_input)
            elif self.training_stage == -1000:
                v_rate = v_rate.detach() + self.correctors[-1](corrector_input)
            else:  # train the n-th part.
                v_rate = v_rate.detach()
                for i in range(self.training_stage):
                    if i != self.training_stage - 1:
                        v_rate += self.correctors[i](corrector_input).detach()
                    else:
                        v_rate += self.correctors[i](corrector_input)
        if self.quat_config:
            q_quat_beging_idx, v_omega_begin_idx = self.quat_config
            new_part = Quaternion(
                x[:, q_quat_beging_idx : q_quat_beging_idx + 4]
            ).clone()
            x = torch.cat(
                [x[:, :q_quat_beging_idx], new_part, x[:, q_quat_beging_idx + 4 :]],
                dim=1,
            )

            q_quat = x[:, q_quat_beging_idx : q_quat_beging_idx + 4]
            v_omega = x[:, v_omega_begin_idx : v_omega_begin_idx + 3]
            q_quat_rate = quaternion_derivative_batch(q_quat, v_omega)
            q_rate = torch.cat(
                (
                    x[:, self.q_dim : v_omega_begin_idx],
                    q_quat_rate,
                    x[:, v_omega_begin_idx + 3 :],
                ),
                dim=1,
            ).detach()
        else:
            if self.training:
                q_rate = v.detach()
            else:
                q_rate = v
        return torch.cat((q_rate, v_rate, torch.zeros_like(action)), dim=1)


class Dynamic_model_resnet(nn.Module):
    def __init__(
        self,
        q_dim,
        v_dim,
        action_dim,
        hidden_block_num,
        network_width,
        activation,
        is_atten,
        correctors=[],
        quat_config=None,
    ):
        """
        quat_config: int tuple (q_quat_beging_idx, v_omega_begin_idx)
        """
        super().__init__()
        self.q_dim = q_dim
        self.residual_module = ResidualNet(
            q_dim + v_dim + action_dim,
            v_dim,
            hidden_block_num,
            network_width,
            activation,
            is_atten,
        )
        self.correctors = nn.ModuleList(correctors)
        self.quat_config = quat_config
        self.action = None
        self.training_stage = -1

    def forward(self, t, x):
        q = x[:, : self.q_dim]
        v = x[:, self.q_dim :]
        if self.action == None:
            raise Exception("Action not set!")

        module_input = torch.cat((x, self.action), dim=1)
        v_rate = self.residual_module(module_input)

        if self.training_stage != 0:
            corrector_input = module_input
            if self.training_stage == -1:  # integral training.
                for i in range(len(self.correctors)):
                    v_rate += self.correctors[i](corrector_input)
            else:  # train the n-th part.
                v_rate = v_rate.detach()
                for i in range(self.training_stage):
                    if i != self.training_stage - 1:
                        v_rate += self.correctors[i](corrector_input).detach()
                    else:
                        v_rate += self.correctors[i](corrector_input)
        if self.quat_config:
            q_quat_beging_idx, v_omega_begin_idx = self.quat_config
            new_part = Quaternion(
                x[:, q_quat_beging_idx : q_quat_beging_idx + 4]
            ).clone()
            x = torch.cat(
                [x[:, :q_quat_beging_idx], new_part, x[:, q_quat_beging_idx + 4 :]],
                dim=1,
            )

            q_quat = x[:, q_quat_beging_idx : q_quat_beging_idx + 4]
            v_omega = x[:, v_omega_begin_idx : v_omega_begin_idx + 3]
            q_quat_rate = quaternion_derivative_batch(q_quat, v_omega)
            q_rate = torch.cat(
                (
                    x[:, self.q_dim : v_omega_begin_idx],
                    q_quat_rate,
                    x[:, v_omega_begin_idx + 3 :],
                ),
                dim=1,
            ).detach()
        else:
            q_rate = v.detach()

        return torch.cat((q_rate, v_rate), dim=1)


class Quaternion(torch.Tensor):
    def __new__(cls, quaternions):
        if quaternions.shape[-1] != 4:
            raise ValueError(
                "Input must be a quaternion tensor of shape (batch_size, 4)."
            )

        instance = torch.as_tensor(quaternions).to(torch.float32)

        if not cls.is_unit(instance):
            instance = cls._fast_normalize(instance)

        return instance.view_as(instance)

    @staticmethod
    def _normalize(quaternions):
        """Normalize a batch of input quaternions."""
        norm = torch.norm(quaternions, dim=-1, keepdim=True)
        # Prevent division by zero
        norm = torch.where(norm > 0, norm, torch.tensor(1.0, device=quaternions.device))
        return quaternions / norm  # Normalize and return

    @staticmethod
    def _fast_normalize(quaternions):
        """
        Fast normalization using Pade approximation,
        a quicker method when the error is small.
        """
        mag_squared = torch.sum(
            quaternions**2, dim=-1, keepdim=True
        )  # Compute squared magnitude
        # Return original quaternions if magnitude is close to 0
        mag_squared = torch.where(
            mag_squared == 0, torch.tensor(1.0, device=quaternions.device), mag_squared
        )

        # Use fast approximation or exact square root
        approx_condition = torch.abs(1.0 - mag_squared) < 2.107342e-08
        mag = torch.where(
            approx_condition, (1.0 + mag_squared) / 2.0, torch.sqrt(mag_squared)
        )

        return quaternions / mag  # Return normalized quaternions

    @staticmethod
    def is_unit(quaternions, tolerance=1e-5):
        """Check whether each quaternion in the batch is a unit quaternion, within a tolerance."""
        norm = torch.norm(quaternions, dim=-1)  # Compute L2 norm of each quaternion
        return torch.all(
            torch.abs(norm - 1.0) < tolerance
        )  # Return boolean if norms are close to 1


def quaternion_multiply_batch(q1, q2):
    """
    Compute the Hamilton product (quaternion multiplication) for a batch of quaternions.
    :param q1: torch tensor [batch_size, 4] - First batch of quaternions
    :param q2: torch tensor [batch_size, 4] - Second batch of quaternions
    :return: torch tensor [batch_size, 4] - Resulting batch of multiplied quaternions
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=1)


def quaternion_derivative_batch(q, omega):
    """
    Compute the derivative of a batch of quaternions.
    :param q: torch tensor [batch_size, 4] - Current batch of quaternions
    :param omega: torch tensor [batch_size, 3] - Batch of angular velocities (rad/s)
    :return: torch tensor [batch_size, 4] - Derivative of each quaternion in the batch
    """
    # Represent angular velocity as quaternion: [0, omega_x, omega_y, omega_z]
    omega_quat = torch.cat((torch.zeros(q.size(0), 1).to(omega), omega), dim=1)

    # Quaternion rate of change: 1/2 * q ⊗ omega
    q_rate = quaternion_multiply_batch(q, omega_quat) * 0.5
    return q_rate
