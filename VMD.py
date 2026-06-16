import numpy as np
import pandas as pd
from vmdpy import VMD


import os
import glob

data = pd.read_csv('wind_data_cleaned.csv',index_col='timestamps')
data.drop(data.columns[0],axis=1,inplace = True)

signal = data['wspd100'].dropna().values

def try_vmd(signal, K, alpha=2000, tau=0, DC=0, init=1, tol=1e-7):
    u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
    recon_error = np.abs(signal - u.sum(axis=0)).max()
    return u, omega, recon_error

K_candidates = [4, 6, 8, 10, 12, 14, 16]

## Uncomment for plots
# for ax, K in zip(axes, K_candidates):
#     u, omega, err = try_vmd(signal, K)
#     for i in range(K):
#         ax.plot(u[i, :500], alpha=0.7, linewidth=0.8, label=f'IMF {i+1}')
#     ax.set_title(f'K={K}  |  Reconstruction error: {err:.6f}')
#     ax.set_xlabel('Sample')
#     ax.legend(loc='upper right', fontsize=7, ncol=K)
#     ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

for K in K_candidates:
    u, omega, err = try_vmd(signal, K)
    print("Number of modes: ",f"{K} ","Reconstruction error: ",f"{err}")

K_MODES  = 14
heights  = ['40', '60', '80', '100', '120', '140']

os.makedirs('vmd_series', exist_ok=True)

for height in heights:
    signal = data[f'wspd{height}'].dropna().values
    u, omega, err = try_vmd(signal, K_MODES)
    np.savez(f'vmd_series/vmd_results_wspd{height}.npz', u=u, omega=omega)
    print(f"Saved vmd_results_wspd{height}.npz  |  reconstruction error: {err:.6f}")

