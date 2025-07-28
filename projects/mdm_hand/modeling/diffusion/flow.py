# Convert diffusion model's output ("eps" or "v") to flow matching's velocity flow ("u").
# Source: https://github.com/ylipman/CNF_REG/blob/3d80ce51bca868fa2dbe75b8af0b37a0a560b136/lib/models/transform.py#L152

import torch


# Subclassing `nn.Module` just for it to follow parent's `.to()`
class DiffusionToFlowMatchingConverter(torch.nn.Module):
    def __init__(self, alphas_cumprod, parametrization):
        super().__init__()
        self.fp32_buffers = set()

        self._extract_params_for_u_mapping(alphas_cumprod)
        self.parametrization = parametrization

    def forward(self, x_recon, t, x_noisy):
        dmu_t, mu_t, dsigma_t, sigma_t = self._get_terms(t, method="linear")

        if self.parametrization == "eps":
            mu_t = mu_t.clamp(min=1e-6)
            term1 = dmu_t / mu_t
            term2 = dmu_t * sigma_t / mu_t - dsigma_t

            x_recon = (
                term1.view(-1, 1, 1, 1) * x_noisy - term2.view(-1, 1, 1, 1) * x_recon
            )
        elif self.parametrization == "v":
            sigma_t = sigma_t.clamp(min=1e-6)
            term1 = (dsigma_t / sigma_t) * (mu_t**2 - 1) - mu_t * dmu_t
            term2 = dsigma_t * mu_t - dmu_t * sigma_t

            x_recon = (
                -term1.view(-1, 1, 1, 1) * x_noisy + term2.view(-1, 1, 1, 1) * x_recon
            )
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} and parametrization: '{self.parametrization}'"
            )

        return x_recon

    def register_fp32_buffer(self, name, *args, **kwargs):
        self.register_buffer(name, *args, **kwargs)
        self.fp32_buffers.add(name)

    def _apply(self, *args, **kwargs):
        # This function is used by `nn.Module`'s `.to()`, `.half()`, `.cuda()` etc.
        # To keep some buffers' dtype, we have to override it, `.to()` isn't enough:
        # https://stackoverflow.com/questions/75400657/overriding-nn-modules-to-does-not-work-when-module-is-child

        # Backup the always-fp32 tensors
        fp32_backup = {name: getattr(self, name).clone() for name in self.fp32_buffers}

        retval = super()._apply(*args, **kwargs)

        # Always-fp32 tensors possibly got converted to other type. Revert that
        for name, tensor in fp32_backup.items():
            assert (
                tensor.dtype == torch.float32
            ), f"Someone has previously converted an always-fp32 buffer {name!r} to {tensor.dtype} manually! Please don't do that"
            new_device = getattr(self, name).device
            self.register_buffer(name, tensor.to(new_device))

        return retval

    def _extract_params_for_u_mapping(self, alphas_cumprod):
        num_timesteps = len(alphas_cumprod)

        _mu = torch.sqrt(alphas_cumprod)
        _sigma = torch.sqrt(1 - alphas_cumprod)

        _dmu = torch.cat(
            [
                torch.zeros(1, device=_mu.device, dtype=_mu.dtype),
                -torch.diff(_mu) * num_timesteps,
            ]
        )
        _dsigma = torch.cat(
            [
                torch.zeros(1, device=_sigma.device, dtype=_sigma.dtype),
                -torch.diff(_sigma) * num_timesteps,
            ]
        )

        self.register_fp32_buffer("_mu", _mu)
        self.register_fp32_buffer("_sigma", _sigma)
        self.register_fp32_buffer("_dmu", _dmu)
        self.register_fp32_buffer("_dsigma", _dsigma)

    @staticmethod
    def _piecewise_linear_1d_interp(input, t_queries):
        # `input` is a tensor of shape [num_points] containing the values to interpolate
        # `t_queries` is a tensor of shape [num_queries] containing the interpolation queries, which are floats in the interval [0, num_points - 1]
        # The function returns a tensor of shape [num_queries] containing the interpolated values, computed via piecewise linear interpolation
        # First, let's find the two closest indices for each query
        # `floors` is a tensor of shape [num_queries] containing the largest integer <= each query
        # `ceils` is a tensor of shape [num_queries] containing the smallest integer >= each query
        floors = torch.floor(t_queries).long()
        ceils = torch.ceil(t_queries).long()
        # Let's also compute the distance between each query and the two closest indices
        # `deltas` is a tensor of shape [num_queries] containing the distance between each query and the two closest indices
        deltas = (
            t_queries - floors.float()
        )  # <------ This is where the gradients flow for the variable `t_queries`
        # Let's clamp the indices so that they are within the range [0, num_points - 1]
        floors = torch.clamp(floors, 0, input.shape[0] - 1)
        ceils = torch.clamp(ceils, 0, input.shape[0] - 1)
        # Let's gather the values for the two closest indices for each query
        # `lower` is a tensor of shape [num_queries] containing the value of the closest index smaller than each query
        # `upper` is a tensor of shape [num_queries] containing the value of the closest index larger than each query
        lower = input[floors]
        upper = input[ceils]
        # Let's compute the interpolated values
        output = lower + deltas * (upper - lower)
        return output

    def _get_terms(self, t, method="linear"):
        t_in_range = t
        if method == "floor":  # This was the default method
            # We find the index of a close element in the tensor, via the long() cast
            t_ind = t_in_range.long()  # The indices that index the mu and sigma tensors
            sel_dmu_t = self._dmu[t_ind]
            sel_mu_t = self._mu[t_ind]
            sel_dsigma_t = self._dsigma[t_ind]
            sel_sigma_t = self._sigma[t_ind]
        elif method == "linear":
            # Need to interpolate the values of mu_t and sigma_t at the given t_in_range
            sel_dmu_t = self._piecewise_linear_1d_interp(self._dmu, t_in_range)
            sel_mu_t = self._piecewise_linear_1d_interp(self._mu, t_in_range)
            sel_dsigma_t = self._piecewise_linear_1d_interp(self._dsigma, t_in_range)
            sel_sigma_t = self._piecewise_linear_1d_interp(self._sigma, t_in_range)

        return sel_dmu_t, sel_mu_t, sel_dsigma_t, sel_sigma_t