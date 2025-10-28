import torch
import matplotlib.pyplot as plt

import utils
import ssfm

device = torch.device("cpu")

batch_size = 16
n_symbs = 4096
sps = 2
M = 16
power = 6 # dBm
snr = 20 # dB

# Set parameters of the SSFM
fp = ssfm.FibreParameters()
fp.pmd = 0.1
fp.disp = 20.0
fp.gamma = 1.4
fp.attenuation = 0.2

# Initialise SSFM channel
ssfm_channel = ssfm.SSFM(fp, n_symbs * sps, device=device)

# Generate symbols, upsample, pulse shape
symbs = utils.generate_qam(M, n_symbs, batch_size=batch_size, device=device)
samples = utils.upsample_time(symbs, sps, dim=1)
rrc = utils.time_domain_rrc(2, 0.1, 201, device=device)
samples_ps = utils.apply_filter_1d(samples, rrc, dim=1)
samples_ps = utils.set_power(samples_ps, power, dim=1)
plt.plot(samples_ps[0,:,0].detach().cpu().abs())
plt.show()

# SSFM channel simulating coupled NLSE
out = ssfm_channel.simulate(samples_ps)
out = utils.awgn(out, snr)

# Matched filter
out = utils.apply_filter_1d(out, rrc, dim=1)
out = utils.normalise(out, dim=1)

plt.scatter(out[0, ::sps].detach().cpu().real, out[0, ::sps].detach().cpu().imag, s=1)
plt.show()
