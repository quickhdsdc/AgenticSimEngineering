import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint


# rtol=1e-3, atol=1e-4 method='dopri5'
class ODEBijector(nn.Module):
    def __init__(
        self,
        dynamic_model,
        q_dim,
        v_dim,
        real_t,
        integrator="dopri5",
        manually_int_q=False,
        mid_point_int=False,
    ):
        super().__init__()
        self.dynamic_model = dynamic_model
        self.q_dim = q_dim
        self.v_dim = v_dim
        self.real_t = real_t
        self.integrator = integrator
        self.manually_int_q = manually_int_q
        self.mid_point_int = mid_point_int
        self.if_fast = False

    def forward(self, state, action, t0, t1, **kwargs):
        state = torch.cat((state, action), dim=1)
        if self.if_fast == True:
            xs = odeint(
                self.dynamic_model,
                state,
                torch.tensor([t0, t1]).to(state),
                rtol=1e-3,
                atol=1e-4,
                method=self.integrator,
                **kwargs
            )
        else:
            xs = odeint(
                self.dynamic_model,
                state,
                torch.tensor([t0, t1]).to(state),
                method=self.integrator,
                **kwargs
            )

        if self.manually_int_q:
            if not self.mid_point_int:
                q_rate = xs[-1][:, self.q_dim : self.q_dim + self.v_dim]
            else:
                q_rate = (
                    xs[-1][:, self.q_dim : self.q_dim + self.v_dim]
                    + state[:, self.q_dim : self.q_dim + self.v_dim]
                ) / 2
            q0 = state[:, : self.q_dim]
            q1 = q0 + q_rate * self.real_t
            if self.training:
                q1 = q1.detach()
            result = torch.cat(
                (q1, xs[-1][:, self.q_dim : self.q_dim + self.v_dim]), dim=1
            )
        else:
            result = xs[-1][:, : self.q_dim + self.v_dim]
        return result
