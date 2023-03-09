import torch
import torch.nn as nn
import numpy as np


class GFA(nn.Module):

    def __init__(self, p=0.5, eps=1e-6):
        super(GFA, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1e-1

        self.phase_memory = [torch.zeros(1)] * 4
        self.init = [False] * 4
        self.alpha = 0.9

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x, index):
        # B C H W

        if (not self.training) or (np.random.random()) > self.p:
            return x

        frequency = torch.fft.fft2(x, dim=[2, 3], norm="ortho")
        amplitude = torch.abs(frequency)
        phase = torch.angle(frequency)
        if self.init[index]:
            phase = self.alpha * phase + (1 - self.alpha) * self.phase_memory[index]
        else:
            self.init[index] = True

        self.phase_memory[index] = phase.detach()
        amplitude_mean = amplitude.mean(dim=[2, 3], keepdim=False)
        amplitude_std = (amplitude.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        amplitude_sqrtvar_mu = self.sqrtvar(amplitude_mean)
        amplitude_sqrtvar_std = self.sqrtvar(amplitude_std)

        amplitude_beta = self._reparameterize(amplitude_mean, amplitude_sqrtvar_mu)
        amplitude_gamma = self._reparameterize(amplitude_std, amplitude_sqrtvar_std)

        amplitude = (amplitude - amplitude_mean.reshape(amplitude.shape[0], amplitude.shape[1], 1, 1))/ \
                    amplitude_std.reshape(amplitude.shape[0], amplitude.shape[1], 1, 1)
        amplitude = amplitude * amplitude_gamma.reshape(amplitude.shape[0], amplitude.shape[1], 1, 1) + \
                    amplitude_beta.reshape(amplitude.shape[0], amplitude.shape[1], 1, 1)

        x = torch.fft.ifft2(amplitude * torch.exp(1j * phase), dim=[2, 3], norm="ortho").real

        return x