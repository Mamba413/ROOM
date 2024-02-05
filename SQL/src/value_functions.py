import torch
import torch.nn as nn
from .util import mlp

class TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = mlp(dims, squeeze_output=True)
        self.q2 = mlp(dims, squeeze_output=True)

    def both(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state, action):
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = mlp(dims, squeeze_output=True)

    def forward(self, state):
        return self.v(state)


class PB_Q(nn.Module):
    def __init__(self, q_list, scale=2.0) -> None:
        super().__init__()
        self.q_list = nn.ModuleList(q_list)
        self.q_num = len(q_list)
        self.ensemble = True
        self.scale = scale
        pass

    def forward(self, state, action):
        predict_q_value = torch.zeros(size=(state.shape[0], self.q_num))
        for k in range(self.q_num):
            predict_q_value[:, k] = self.q_list[k](state, action)
        predict_q_value_mean = torch.mean(predict_q_value, 1)
        predict_q_value_std = torch.std(predict_q_value, 1)
        predict_q_value = predict_q_value_mean - self.scale * predict_q_value_std
        return predict_q_value.detach()


class PB_V(nn.Module):
    def __init__(self, v_list, scale=2.0) -> None:
        super().__init__()
        self.v_list = nn.ModuleList(v_list)
        self.v_num = len(v_list)
        self.ensemble = True
        self.scale = scale
        pass

    def forward(self, state):
        predict_v_value = torch.zeros(size=(state.shape[0], self.v_num))
        for k in range(self.v_num):
            predict_v_value[:, k] = self.v_list[k](state)
        predict_v_value_mean = torch.mean(predict_v_value, 1)
        predict_v_value_std = torch.std(predict_v_value, 1)
        predict_v_value = predict_v_value_mean - self.scale * predict_v_value_std
        return predict_v_value.detach()


class MM_Q(nn.Module):
    def __init__(self, q_list, quantile=0.5) -> None:
        super().__init__()
        self.q_list = nn.ModuleList(q_list)
        self.q_num = len(q_list)
        self.ensemble = True
        self.quantile = quantile
        pass

    def forward(self, state, action):
        predict_q_value = torch.zeros(size=(state.shape[0], self.q_num))
        for k in range(self.q_num):
            predict_q_value[:, k] = self.q_list[k](state, action)
        # if self.quantile == 0.5:
        #     predict_q_value = torch.median(predict_q_value, 1)[0]
        # else:
        #     predict_q_value = torch.quantile(predict_q_value, self.quantile, 1)
        predict_q_value = torch.quantile(predict_q_value, self.quantile, 1)
        return predict_q_value.detach()


class MM_V(nn.Module):
    def __init__(self, v_list, quantile=0.5) -> None:
        super().__init__()
        self.v_list = nn.ModuleList(v_list)
        self.v_num = len(v_list)
        self.ensemble = True
        self.quantile = quantile
        pass

    def forward(self, state):
        predict_v_value = torch.zeros(size=(state.shape[0], self.v_num))
        for k in range(self.v_num):
            predict_v_value[:, k] = self.v_list[k](state)
        # if self.quantile == 0.5:
        #     predict_v_value = torch.median(predict_v_value, 1)[0]
        # else:
        #     predict_v_value = torch.quantile(predict_v_value, self.quantile, 1)
        predict_v_value = torch.quantile(predict_v_value, self.quantile, 1)
        return predict_v_value.detach()
    

class Mean_Q(nn.Module):
    def __init__(self, q_list) -> None:
        super().__init__()
        self.q_list = nn.ModuleList(q_list)
        self.q_num = len(q_list)
        self.ensemble = True
        pass

    def forward(self, state, action):
        predict_q_value = torch.zeros(size=(state.shape[0], self.q_num))
        for k in range(self.q_num):
            predict_q_value[:, k] = self.q_list[k](state, action)
        predict_q_value = torch.mean(predict_q_value, 1)
        return predict_q_value.detach()


class Mean_V(nn.Module):
    def __init__(self, v_list) -> None:
        super().__init__()
        self.v_list = nn.ModuleList(v_list)
        self.v_num = len(v_list)
        self.ensemble = True
        pass

    def forward(self, state):
        predict_v_value = torch.zeros(size=(state.shape[0], self.v_num))
        for k in range(self.v_num):
            predict_v_value[:, k] = self.v_list[k](state)
        predict_v_value = torch.mean(predict_v_value, 1)
        return predict_v_value.detach()