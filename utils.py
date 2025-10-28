import torch
import torch.nn.functional as F
import numpy as np
import math

def normalise(samples, dim=0):
    """
    Normalise input signal to unitary power across given dimension

    Params:
        samples: torch.tensor, shape (...)
            Samples to normalise
        dim: int
            Dimension to normalise across
    
    Returns:
        samples_norm: torch.tensor, shape (...)
            The normalised samples
    """
    power = torch.sqrt(torch.mean(samples.abs()**2, dim=dim, keepdim=True))
    samples_norm = samples / power
    return samples_norm


def generate_qam(M, n_symbs, batch_size=1, device=torch.device("cpu")):
    """
    Generate complex QAM symbols

    Params:
        M: int
            QAM modulation order. Must be a power of 4 to be valid (to maintain
            a square constellation)
        n_symbs: int
            Number of symbols to generate
        batch_size: int
            Size of the batch
        device: torch.device
            The device to generate the symbols on
    
    Returns:
        symbs: torch.tensor, shape (batch_size, n_symbs, 2), dtype torch.complex64
            Dual-polarisation QAM symbols, where the last dimension represents the two polarisations
    """
    assert math.log(M, 4).is_integer(), "QAM modulation order must be a power of 4, i.e. 4, 16, 64, ..."

    sqrt_M = round(math.sqrt(M))
    x = 2 * torch.randint(0, sqrt_M, (batch_size, n_symbs, 2, 2), device=device, dtype=torch.float32) - (sqrt_M-1)
    symbs = torch.view_as_complex(x)
    symbs = normalise(symbs, dim=1)
    return symbs


def upsample_time(samples, upsample_factor, dim=0):
    """
    Upsamples the signal via zero-packing.

    Params:
        samples: torch.tensor, shape (..., N, ...)
            Samples to upsample
        upsample_rate: int
            The upsampling factor, which defines how many output samples per input sample
        dim: int
            The dimensions to upsample across
    
    Returns:
        resampled_samples: torch.tensor, shape (..., N*uf, ...)
            The zero-packed samples
    """
    # Skip upsampling if no upsampling is present
    if upsample_factor == 1:
        return samples

    num = samples.shape[dim] * upsample_factor
    resampled_shape = list(samples.shape)
    resampled_shape[dim] = num
    resampled_samples = torch.zeros(resampled_shape, device=samples.device, dtype=samples.dtype)

    resampled_idx = [slice(None)] * samples.ndim
    resampled_idx[dim] = slice(0, -1, upsample_factor)
    resampled_samples[tuple(resampled_idx)] = samples

    return resampled_samples


def time_domain_rrc(sps, alpha, n_taps, device=torch.device("cpu")):
    """
    Defines an RRC filter for pulse shaping

    Params:
        sps: int
            Samples per symbol in the incoming signal
        alpha: float
            RRC roll-off factor. Usual values are 0.1 or 0.01, where values closer to 0 provide a 
            more rectangular spectrum but require more filter taps to represent.
        n_taps: int
            Number of taps in the RRC filter
        device: torch.device
            The device to generate the RRC on

    Returns:
        b: torch.tensor, shape (n_taps, ), dtype=torch.complex64
            RRC filter. Dtype is complex to allow for convolution with complex signals, even though the
            coefficients themselves are real


    """
    assert n_taps % 2 == 1, "Number of RRC taps must be an odd number"
    n = torch.arange(-(n_taps//2), n_taps//2+1, device=device).to(dtype=torch.complex64)

    eps = torch.abs(n[0]-n[1])/4
    idx1 = torch.abs(n) < eps
    idx2 = torch.abs(torch.abs(n)-abs(sps/(4*alpha))) < eps
    
    b = 1/sps*((torch.sin(torch.pi*n/sps*(1-alpha)) +  4*alpha*n/sps*torch.cos(torch.pi*n/sps*(1+alpha)))/(torch.pi*n/sps*(1-(4*alpha*n/sps)**2)))
    b[idx1] = 1/sps*(1+alpha*(4/np.pi-1))
    b[idx2] = alpha/(sps*np.sqrt(2))*((1+2/torch.pi)*np.sin(torch.pi/(4*alpha))+(1-2/torch.pi)*np.cos(torch.pi/(4*alpha)))
    
    return b


def apply_filter_1d(signal, filt, dim=0):
    """
    Convolve a 1D filter with a signal along a given dimension using zero-padding.
    
    Args:
        signal (torch.Tensor): Input tensor of arbitrary shape (..., L, ...).
        filt (torch.Tensor): 1D filter tensor of shape (K,).
        dim (int): Dimension along which to perform the convolution.
    
    Returns:
        torch.Tensor: Convolved tensor of the same shape as `signal`.
    """
    # Ensure filter is 1D
    assert filt.ndim == 1, "Filter must be 1D"
    
    # Move the target dimension to the last position
    signal = signal.transpose(dim, -1)
    orig_shape = signal.shape
    L = orig_shape[-1]

    # Flatten all other dimensions into the batch dimension
    signal_flat = signal.reshape(-1, 1, L)  # (batch, channels=1, length)
    
    # Prepare filter
    filt = filt.to(signal.device, signal.dtype).view(1, 1, -1)
    
    # Compute padding (same output length as input)
    pad = filt.shape[-1] // 2
    out = F.conv1d(signal_flat, filt, padding=pad)
    
    # Reshape back to original shape
    out = out.reshape(*orig_shape)
    
    # Move dimension back to original position
    out = out.transpose(dim, -1)
    
    return out


def set_power(signal, power, dim=0):
    """
    Sets the power of a signal

    Parameters:
        signal: torch.tensor
        power: float
            The power in dBms to set the signal to
        dim: int
            The dimensions over which to set power
    
    Returns:
        signal: torch.tensor
            The signal set to the given power
    """
    # Divide power amongst polarisations
    pol_power = power - 10*np.log10(2)
    signal_power = 1000*torch.mean((torch.abs(signal))**2, dim=dim, keepdim=True)
    power_shape = [1 for _ in range(signal_power.ndim)]
    coef = torch.sqrt(10**(pol_power/10) / signal_power)
    signal *= coef
    return signal


def awgn(signal, snr):
    """
    Applies AWGN according to a specific SNR to the signal

    Params:
        signal: torch.tensor, dtype=torch.complex64
            The signal to apply noise to. This assumes a normalised signal.
        snr: float
            The signal-to-noise ratio which controls the variance of the noise
    
    Returns:
        signal_awgn: torch.tensor, dtype=torch.complex64
            The signal with AWGN applied to it
    """
    noise = torch.randn(signal.shape, device=signal.device, dtype=torch.complex64)
    noise *= 10**(-snr/20)
    signal_awgn = signal + noise
    return signal_awgn