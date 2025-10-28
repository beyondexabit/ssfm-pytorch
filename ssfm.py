import torch
import numpy as np
import math

import utils

# Speed of light :D
C = 299792458

class FibreParameters():
    def __init__(self):
        self.attenuation = 0.16 # dB/km
        self.pmd = 0.1 # ps/sqrt(km)
        self.disp = 17 # ps/nm/km
        self.gamma = 1.3 # /W/km
        self.length = 50e3 # m
        self.dz = 1000 # m
        self.carrier_wavelength = 1550e-9 # m
        self.sampling_freq = 60e9 # Hz


class SSFM():
    def __init__(self, params: FibreParameters, n_samples: int, device=torch.device("cpu")):
        """
        Initialised parameters for SSFM transmission

        Params: 
            params: FibreParameters
                Contains the main parameters for the SSFM, including dispersion, nonlinearities, and PMD
            n_samples: int
                The number of samples in the incoming signal
            device: torch.device
                The device that the data is stored and simulated on
        """
        self.params = params
        self.device = device

        att = params.attenuation
        pmd = params.pmd
        disp = params.disp
        gamma = params.gamma
        z = params.length
        dz = params.dz
        carrier_wavelength = params.carrier_wavelength
        sampling_freq = params.sampling_freq
        self.steps = round(z // dz)

        # Convert from dB/km to Np/m
        attenuation = torch.tensor(att / (10*np.log10(np.e)) / 1000, device=device, dtype=torch.float32)

        # Convert from ps/sqrt(km) to s 
        mean_total_dgd = pmd * math.sqrt(z) * 1e-12
        # DGD per section
        beta1 = np.sqrt(3 * np.pi / (8 * self.steps)) * mean_total_dgd
        # Polarisation-based delay
        dgd = torch.tensor([beta1, -beta1], device=device, dtype=torch.float32) / 2

        # Chromatic dispersion
        beta2 = -disp * (carrier_wavelength**2) / (2 * torch.pi * C) / 1e6
        beta2 = torch.tensor(beta2, dtype=torch.float32, device=device)

        # Construct unitary coupling matrix for each step via QR decomposition
        a = torch.randn((self.steps, 2, 2), device=device, dtype=torch.complex64)
        self.c, _ = torch.linalg.qr(a)

        # Define the linear step in the frequency domain
        freq_bins = torch.fft.fftfreq(n_samples, device=device) * sampling_freq
        w = 2 * torch.pi * freq_bins
        D = (-attenuation/2 - 1j*(beta2.unsqueeze(0)/2 * w**2)).unsqueeze(-1) * dz/2 - 1j*dgd.unsqueeze(0)*w.unsqueeze(-1)/2 
        # Shape (1024, 2)
        self.half_step = torch.exp(D)

        # Nonlinearity matrix, where each polarisation is affected by its own power and 2/3 the power of the other polarisation
        self.nl_mat = torch.tensor([[gamma, 2*gamma/3], [2*gamma/3, gamma]], dtype=torch.float32, device=device) / 1000 * dz


    def simulate(self, x):
        """
        Implements the coupled Nonlinear Schrodinger Equation using the SSFM

        Params:
            x: torch.tensor, shape (batch_size, n_samples, 2)
                The input signal to the fibre
        
        Returns
            x: torch.tensor, shape (batch_size, n_samples, 2)
                The normalised output signal of the fibre
        """
        for step in range(self.steps):
            # Coupling
            x = torch.matmul(x, self.c[step])

            # Dispersion
            X = torch.fft.fft(x, dim=1)
            X = self.half_step.unsqueeze(0) * X
            x = torch.fft.ifft(X, dim=1)

            # Nonlinearities
            theta = torch.matmul(x.abs()**2, self.nl_mat)
            x = x * torch.exp(1j * theta)

            # Dispersion
            X = torch.fft.fft(x, dim=1)
            X = self.half_step.unsqueeze(0) * X
            x = torch.fft.ifft(X, dim=1)

        x = utils.normalise(x, dim=1)

        return x

        
        