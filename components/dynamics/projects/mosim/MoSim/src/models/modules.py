import torch
from torch import nn
import torch.nn.modules.loss as Loss
import torch.nn.functional as F

import MoSim.src.tools as tool


class Atan(nn.Module):
    def forward(self, x):
        return torch.atan(x)


class Softplus(nn.Module):
    def forward(self, x):
        return F.softplus(x)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.as_tensor(0.0))

    def forward(self, x):
        x = x * nn.functional.sigmoid(self.beta * x)
        return x


def select_activation_function(function_type):
    if function_type == "atan":
        activation_func = Atan()
    elif function_type == "relu":
        activation_func = nn.ReLU()
    elif function_type == "softplus":
        activation_func = Softplus()
    elif function_type == "swish":
        activation_func = Swish()
    else:
        raise ValueError(f"Unsupported activation function: {function_type}")
    return activation_func


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, activation_function):
        super(AttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            select_activation_function(activation_function),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        x = self.attn(x, x, x)[0] + x
        x = self.ffn(x) + x
        return x


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_block_num,
        hidden_dim,
        activation_function,
        norm=False,
    ):
        super(MLP, self).__init__()
        self.fc_input_hidden = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.hidden_layers_list = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_block_num)]
        )
        self.fc_hidden_output = nn.Linear(hidden_dim, output_dim)
        self.activation_function = select_activation_function(activation_function)
        self.norm = norm

    def forward(self, x):
        x = self.fc_input_hidden(x)
        if self.norm:
            x = self.layer_norm(x)
        x = self.activation_function(x)
        for hidden_layer in self.hidden_layers_list:
            x = hidden_layer(x)
            if self.norm:
                x = self.layer_norm(x)
            x = self.activation_function(x)
        x = self.fc_hidden_output(x)
        return x


class FeatureProcessor(nn.Module):
    def __init__(self, inp_dim, hidden_dim, norm=True, act=nn.SiLU):
        super().__init__()
        obs_out_layers = []
        obs_out_layers.append(nn.Linear(inp_dim, hidden_dim, bias=False))
        if norm:
            obs_out_layers.append(nn.LayerNorm(hidden_dim, eps=1e-03))
        obs_out_layers.append(act())
        self._obs_out_layers = nn.Sequential(*obs_out_layers)

    def forward(self, x):
        return self._obs_out_layers(x)


class LinearEmbeddingLayer(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(LinearEmbeddingLayer, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.embedding = nn.Linear(input_dim, embed_dim * input_dim)

    def forward(self, x):
        x = self.embedding(x)
        return x.reshape(-1, self.input_dim, self.embed_dim)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, activation_function, is_norm=False):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation_function = select_activation_function(activation_function)
        self.is_norm = is_norm
        if is_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        if self.is_norm:
            x = self.layer_norm(x)
        x = self.activation_function(x)
        x = self.fc2(x)
        x += residual
        if self.is_norm:
            x = self.layer_norm(x)
        x = self.activation_function(x)
        return x


class ResidualNet(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_block_num,
        hidden_dim,
        activation_function,
        is_atten=False,
        is_norm=False,
    ):
        super(ResidualNet, self).__init__()
        self.fc_input_hidden = nn.Linear(input_dim, hidden_dim)
        if is_atten == True:
            self.hidden_layers_list = nn.ModuleList(
                [
                    ResidualBlock(hidden_dim, activation_function)
                    for _ in range(hidden_block_num)
                ]
            )
        else:
            self.hidden_layers_list = nn.ModuleList(
                [
                    ResidualBlock(hidden_dim, activation_function)
                    for _ in range(hidden_block_num)
                ]
            )
        self.fc_hidden_output = nn.Linear(hidden_dim, output_dim)
        self.activation_function = select_activation_function(activation_function)
        if is_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        self.is_norm = is_norm

    def forward(self, x):
        x = self.fc_input_hidden(x)
        if self.is_norm:
            x = self.layer_norm(x)
        x = self.activation_function(x)
        for hidden_layer in self.hidden_layers_list:
            x = hidden_layer(x)
        x = self.fc_hidden_output(x)
        return x


class MModule(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_block_num,
        hidden_dim,
        activation_function,
        is_atten,
        is_norm,
    ):
        super(MModule, self).__init__()
        self.output_dim = output_dim
        self.resNet = ResidualNet(
            input_dim,
            output_dim * (output_dim + 1) // 2,
            hidden_block_num,
            hidden_dim,
            activation_function,
            is_atten,
            is_norm,
        )

    def forward(self, x):
        L_elements = self.resNet(x)
        L = torch.zeros(x.size(0), self.output_dim, self.output_dim, device=x.device)
        tril_indices = torch.tril_indices(
            row=self.output_dim, col=self.output_dim, offset=0
        )
        L[:, tril_indices[0], tril_indices[1]] = L_elements
        M = torch.bmm(L, L.transpose(1, 2))
        return M


class Relative_MSELoss(Loss._Loss):
    __constants__ = ["reduction"]

    def __init__(
        self,
        if_norm=False,
        if_relative=False,
        epsilon=1e-5,
        reduction: str = "mean",
        max=None,
        min=None,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.if_relative = if_relative
        self.reduction = reduction
        self.if_norm = if_norm
        self.max = max
        self.min = min

    def forward(self, input, target):
        if self.if_relative:
            safe_target = torch.abs(target) + self.epsilon
            input = torch.abs(input)
            # safe_target.requires_grad=False
            loss = F.mse_loss(
                input / safe_target, torch.ones_like(input), reduction=self.reduction
            )
        elif self.if_norm:
            target_std = torch.std(target, dim=0)
            loss = F.mse_loss(input, target, reduction="none") / target_std
        else:
            loss = F.mse_loss(input, target, reduction="none")
        return loss
