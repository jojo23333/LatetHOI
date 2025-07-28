import torch
from utils import persistence

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, batch_data, labels, augment_pipe=None):
        n_shape = [batch_data.shape[0]] + (len(batch_data.shape) - 1) * [1]
        rnd_uniform = torch.rand(n_shape, device=batch_data.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(batch_data) if augment_pipe is not None else (batch_data, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, batch_data, labels, augment_pipe=None):
        n_shape = [batch_data.shape[0]] + (len(batch_data.shape) - 1) * [1]
        rnd_uniform = torch.rand(n_shape, device=batch_data.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(batch_data) if augment_pipe is not None else (batch_data, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def get_sigma(self, bsz, device):
        rnd_normal = torch.randn(bsz, device=device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        return sigma, weight

    def __call__(self, net, batch_data, condition):
        # Get Noise Level
        data = list(batch_data.items())
        sigma, weight = self.get_sigma(data[0][1].shape[0], data[0][1].device)
        # Add noise
        yn = {}
        for key, y in data:
            view_shape = [-1] + [1] * (len(y.shape) - 1) 
            sigma = sigma.view(*view_shape)
            n = torch.randn_like(y) * sigma
            yn[key] = y + n
        # Denoise

        yn = net.model.dict_2_tensor(yn)
        # print(sigma.mean(), sigma.max(), sigma.min())
        D_yn = net(yn, sigma, condition)
        D_yn = net.model.tensor_2_dict(D_yn)
        # Calculate Loss
        loss = {}
        for key, yn in D_yn.items():
            view_shape = [-1] + [1] * (len(yn.shape) - 1) 
            weight = weight.view(*view_shape)
            assert yn.shape == batch_data[key].shape, (key, yn.shape, batch_data[key].shape)
            loss[key] = weight * ((yn - batch_data[key]) ** 2)
        return loss

    # def __call__(self, net, batch_data, condition=None, augment_pipe=None):
        # if isinstance
        # return loss

    # def __call__(self, net, batch_data, condition):
    #     sigma, weight = self.get_sigma(batch_data.shape[0], batch_data.device)
    #     view_shape = [-1] + [1] * (len(batch_data.shape) - 1) 
    #     sigma = sigma.view(*view_shape)
    #     weight = weight.view(*view_shape)

    #     y = batch_data
    #     n = torch.randn_like(y) * sigma
    #     D_yn = net(y+n, sigma, condition)
    #     loss = weight * ((D_yn - y) ** 2)
    #     return loss, D_yn, weight