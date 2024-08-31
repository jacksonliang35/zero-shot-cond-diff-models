import numpy as np
import glob

n_value = np.array([20,50,100,200,500,1000])

kl1 = np.zeros(n_value.shape)
kl2 = np.zeros(n_value.shape)
kl3 = np.zeros(n_value.shape)
comp1 = np.zeros(n_value.shape)
comp2 = np.zeros(n_value.shape)
comp3 = np.zeros(n_value.shape)

n = 0
for p in glob.glob("*/"):
    if p == 'out/':
        continue
    n += 1
    kl1 += np.load(p + "kl_mixture_reg.npy")
    comp1 += np.load(p + "kl_mixture_reg_comp.npy")
    kl2 += np.load(p + "kl_mixture_accl.npy")
    comp2 += np.load(p + "kl_mixture_accl_comp.npy")
    kl3 += np.load(p + "kl_mixture_accl_genli.npy")
    comp3 += np.load(p + "kl_mixture_accl_genli_comp.npy")

np.save("merged_kl_mixture_reg.npy", kl1 / n)
np.save("merged_kl_mixture_reg_comp.npy", comp1 / n)
np.save("merged_kl_mixture_accl.npy", kl2 / n)
np.save("merged_kl_mixture_accl_comp.npy", comp2 / n)
np.save("merged_kl_mixture_accl_genli.npy", kl3 / n)
np.save("merged_kl_mixture_accl_genli_comp.npy", comp3 / n)
