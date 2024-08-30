import numpy as np
from scipy.stats import norm,multivariate_normal,gaussian_kde
from scipy.linalg import fractional_matrix_power
from scipy.integrate import nquad
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from tqdm import tqdm
import time, os, sys

N = 4
d = 4
Nkern = 200000

I = np.eye(d)
# n_value = np.linspace(100,250,5, dtype=int)
# n_value = np.array([50,100,150,200,250])
# n_value = np.array([20,50,80,150,300,500])
n_value = np.array([20,50,100,200,500,1000])
# n_value = np.array([100,500,1000,2000,5000])

SEED = 100
FILE_NUM = str(int(sys.argv[-1]))
FOLDER = FILE_NUM + "/"

if not os.path.exists(FOLDER):
    os.mkdir(FOLDER)

np.random.seed(SEED)

mu_0 = (np.random.rand(N,d)-0.5)*2
# Sigma_0 = np.diagflat([0.1,1])
# Std_0 = np.sqrt(np.diag(Sigma_0))
# Corr_0 = np.diagflat(1/Std_0).dot(Sigma_0).dot(np.diagflat(1/Std_0))
# aa = 0.6
# Corr_0[:,:] = aa
# Corr_0[:dc,dc:] = aa
# var_0 = np.diagflat(Std_0).dot(Corr_0).dot(np.diagflat(Std_0))
A = np.random.rand(d, d) * 2
var_0 = np.dot(A, A.T) / d
pi_0 = np.array((1./N,)*N)
pi_0 = pi_0 / np.sum(pi_0)
print(mu_0)
print(var_0)
print(pi_0)

class Mixture(object):
    def __init__(self, mu_0, var_0, pi_0):
        super(Mixture, self).__init__()
        self.mu = mu_0
        self.pi = pi_0
        if len(var_0.shape) == 1:
            # independent components
            self.dist = [multivariate_normal(mu_0[i],var_0[i] * I) for i in range(N)]
            self.var = np.array([v * I for v in var_0])
        elif len(var_0.shape) == 2:
            # same correlated matrix among components
            self.dist = [multivariate_normal(mu_0[i],var_0) for i in range(N)]
            self.var = np.tile(var_0,(N,1,1))
        elif len(var_0.shape) == 3:
            # general correlated matrix among components
            self.dist = [multivariate_normal(mu_0[i],var_0[i]) for i in range(N)]
            self.var = var_0

    def pdf(self, x):
        pdf_each = np.array([dd.pdf(x) for dd in self.dist])
        return np.average(pdf_each, axis=0, weights=self.pi)

    def score(self, x, alpha_bar):
        mu_t = np.sqrt(alpha_bar) * self.mu
        var_t = alpha_bar * self.var + (1-alpha_bar) * I
        dist_t = [multivariate_normal(mu_t[i],var_t[i]) for i in range(N)]
        pdf_each = np.array([dd.pdf(x) for dd in dist_t])
        deriv_exp_each = np.array([ -(x-mu_t[i]).dot(np.linalg.inv(var_t[i])) for i in range(N)])
        return np.average(deriv_exp_each * pdf_each[:,:,np.newaxis], axis=0, weights=self.pi) / np.average(pdf_each, axis=0, weights=self.pi)[:, np.newaxis]

    def score_and_hess(self, x, alpha_bar):
        n, d = x.shape
        mu_t = np.sqrt(alpha_bar) * self.mu
        var_t = alpha_bar * self.var + (1-alpha_bar) * I
        var_inv_t = np.array([np.linalg.inv(var_t[i]) for i in range(N)])
        dist_t = [multivariate_normal(mu_t[i],var_t[i]) for i in range(N)]
        pdf_each = np.array([dd.pdf(x) for dd in dist_t])
        deriv_exp_each = np.array([ -(x-mu_t[i]).dot(var_inv_t[i]) for i in range(N)])
        score_t = np.average(deriv_exp_each * pdf_each[:,:,None], axis=0, weights=self.pi) / np.average(pdf_each, axis=0, weights=self.pi)[:, None]
        sec_deriv_each = np.empty((N, n, d, d))
        for i in range(N):
            temp = (x-mu_t[i]).dot(var_inv_t[i])
            sec_deriv_each[i] = temp[:,:,None] * temp[:,None,:] - var_inv_t[i]
        hess_t = np.average(sec_deriv_each * pdf_each[:,:,None,None], axis=0, weights=self.pi) / np.average(pdf_each, axis=0, weights=self.pi)[:, None, None]
        hess_t -= score_t[:,:,None] * score_t[:,None,:]
        return score_t, hess_t

    def score_and_hess_noise(self, x, alpha_bar, noise):
        n, d = x.shape
        mu_t = np.sqrt(alpha_bar) * self.mu
        var_t = alpha_bar * self.var + (1-alpha_bar) * I
        var_inv_t = np.array([np.linalg.inv(var_t[i]) for i in range(N)])
        dist_t = [multivariate_normal(mu_t[i],var_t[i]) for i in range(N)]
        pdf_each = np.array([dd.pdf(x) for dd in dist_t])
        deriv_exp_each = np.array([ -(x-mu_t[i]).dot(var_inv_t[i]) for i in range(N)])
        score_t = np.average(deriv_exp_each * pdf_each[:,:,None], axis=0, weights=self.pi) / np.average(pdf_each, axis=0, weights=self.pi)[:, None]
        sec_deriv_noise_each = np.empty((N, n, d))
        for i in range(N):
            temp = (x-mu_t[i]).dot(var_inv_t[i])
            sec_deriv_noise_each[i] = temp * np.sum(temp * noise, axis=1, keepdims=True) - noise.dot(var_inv_t[i])
        hess_noise_t = np.average(sec_deriv_noise_each * pdf_each[:,:,None], axis=0, weights=self.pi) / np.average(pdf_each, axis=0, weights=self.pi)[:, None]
        hess_noise_t -= score_t * np.sum(score_t * noise, axis=1, keepdims=True)
        return score_t, hess_noise_t

    def rvs(self, size):
        S = size
        palette = np.zeros((N,S,d))
        y = np.zeros((S,d))
        for i in range(N):
            palette[i,:,:] = self.dist[i].rvs(size=S)
        y = palette[np.random.choice(N, p=self.pi), np.arange(S), :]
        # for i in range(S):
        #     y[i,:] = palette[np.random.choice(N, p=self.pi), i, :]
        return y

def calc_samples(dist0, alpha_t, S=10000):
    # Note: alpha_t[i] = \alpha_{i+1}, alpha_t_bar[i] = \bar{\alpha}_{i+1}
    assert(alpha_t[0] > 0)
    alpha_t_bar = np.cumprod(alpha_t)
    X = np.random.normal(size=(S,d))
    for t in range(len(alpha_t)-1,-1,-1):
        # print(t, end=" ")
        # DDPM update
        X = (X + (1-alpha_t[t]) * dist0.score(X, alpha_t_bar[t])) / np.sqrt(alpha_t[t]) + np.sqrt((1-alpha_t[t]) / alpha_t[t])* np.random.normal(size=(S,d))
        # deterministic update
        # X = (X + (1-alpha_t[t])/2 * dist0.score(X, alpha_t_bar[t])) / np.sqrt(alpha_t[t])
    # print("")
    return X

def calc_samples_accl(dist0, alpha_t, S=10000):
    # Note: alpha_t[i] = \alpha_{i+1}, alpha_t_bar[i] = \bar{\alpha}_{i+1}
    assert(alpha_t[0] > 0)
    alpha_t_bar = np.cumprod(alpha_t)
    X = np.random.normal(size=(S,d))
    for t in range(len(alpha_t)-1,-1,-1):
        # print(t, end=" ")
        noise = np.random.normal(size=(S,d))
        score, hess_noise = dist0.score_and_hess_noise(X, alpha_t_bar[t], noise)
        mean = (X + (1-alpha_t[t]) * score) / np.sqrt(alpha_t[t])
        noise_scaled = np.sqrt((1-alpha_t[t]) / alpha_t[t]) * (noise + .5 * (1-alpha_t[t]) * hess_noise)
        X = mean + noise_scaled
    # print("")
    return X

def calc_samples_accl_genli(dist0, alpha_t, S=10000):
    # Note: alpha_t[i] = \alpha_{i+1}, alpha_t_bar[i] = \bar{\alpha}_{i+1}
    assert(alpha_t[0] > 0)
    alpha_t_bar = np.cumprod(alpha_t)
    X = np.random.normal(size=(S,d))
    for t in range(len(alpha_t)-1,-1,-1):
        # print(t, end=" ")
        Y = X + np.sqrt((1-alpha_t[t])/2) * np.random.normal(size=(S,d))
        X = (Y + (1-alpha_t[t]) * dist0.score(X, alpha_t_bar[t]) + np.sqrt((1-alpha_t[t])/2) * np.random.normal(size=(S,d))) / np.sqrt(alpha_t[t])
    # print("")
    return X

def calc_kl_from_mix_samp(dist0, X, S=10000):
    Y = dist0.rvs(S)
    numer = np.log(dist0.pdf(Y))
    denom = np.log(gaussian_kde(X.T, bw_method='silverman')(Y.T))
    return np.maximum(np.mean(numer - denom), 0.)

kl1 = np.zeros(n_value.shape)
kl2 = np.zeros(n_value.shape)
kl3 = np.zeros(n_value.shape)
comp1 = np.zeros(n_value.shape)
comp2 = np.zeros(n_value.shape)
comp3 = np.zeros(n_value.shape)
dist0 = Mixture(mu_0, var_0, pi_0)

# gm = GaussianMixture(n_components=N,
#                      covariance_type='spherical',
#                      weights_init=dist0.pi,
#                      means_init=dist0.mu,
#                      precisions_init=1/dist0.var
#                      ).fit(dist0.rvs(10000))
# print(gm.weights_)
# print(gm.means_)
# print(gm.covariances_)
# exit(0)

# x,y = np.mgrid[-5:5:.1, -5:5:.1]
# plt.contourf(x, y, dist0.pdf(np.dstack((x,y))))
# X = dist0.rvs(1000)
# plt.scatter(X[:,0],X[:,1],s=4,c='k')

np.random.seed(None)

# for i, n in tqdm(enumerate(n_value), total=len(n_value)):
for i, n in enumerate(n_value):
    print("")
    # x,y = np.mgrid[-10:10:.1, -10:10:.1]
    # plt.contourf(x, y, dist0.pdf(np.dstack((x,y))))

    # Sampling
    t_values = np.arange(1, n + 1)
    c = 4
    delta = 0.001
    inner = delta * (1 + c * np.log(n) / n)**(t_values)
    alpha_t = 1 - c * np.log(n) / n * np.minimum(inner,1)
    alpha_t[0] = 1-delta

    # beta_start = 1e-4
    # beta_end = 3. * np.log(n) / n
    # # beta_end = 0.02
    # beta_t = np.linspace(beta_start, beta_end, n)
    # alpha_t = 1 - beta_t

    # Evaluating KL
    # gm = GaussianMixture(n_components=N,
    #                      covariance_type='spherical',
    #                      weights_init=dist0.pi,
    #                      means_init=dist0.mu,
    #                      precisions_init=1/dist0.var
    #                      ).fit(X)
    # print(gm.weights_)
    # print(gm.means_)
    # print(gm.covariances_)
    # print("")

    # while kl1[i] == 0.:
    #     print("Estimating KL for n =", n)
    #     X = calc_samples(dist0, alpha_t, Nkern)
    #     kl1[i] = calc_kl_from_mix_samp(dist0, X, Nkern)
    # while kl2[i] == 0.:
    #     print("Estimating KL accl for n =", n)
    #     X = calc_samples_accl(dist0, alpha_t, Nkern)
    #     kl2[i] = calc_kl_from_mix_samp(dist0, X, Nkern)

    while True:
        # np.random.seed(seed_t)
        start_time = time.time()
        X = calc_samples(dist0, alpha_t, Nkern)
        comp1[i] = time.time() - start_time
        print("Sec for reg:", comp1[i])
        print("Estimating KL for n =", n)
        kl1[i] = calc_kl_from_mix_samp(dist0, X, Nkern)
        if kl1[i] == 0.:
            continue
        break
    print(kl1[i])

    while True:
        # np.random.seed(seed_t)
        start_time = time.time()
        X = calc_samples_accl(dist0, alpha_t, Nkern)
        comp2[i] = time.time() - start_time
        print("Sec for accl:", comp2[i])
        print("Estimating KL accl for n =", n)
        kl2[i] = calc_kl_from_mix_samp(dist0, X, Nkern)
        if kl2[i] == 0.:
            continue
        break
    print(kl2[i])

    while True:
        # np.random.seed(seed_t)
        start_time = time.time()
        X = calc_samples_accl_genli(dist0, alpha_t, Nkern)
        comp3[i] = time.time() - start_time
        print("Sec for accl genli:", comp3[i])
        print("Estimating KL accl genli for n =", n)
        kl3[i] = calc_kl_from_mix_samp(dist0, X, Nkern)
        if kl3[i] == 0.:
            continue
        break
    print(kl3[i])

np.save(FOLDER + "kl_mixture_reg.npy", kl1)
np.save(FOLDER + "kl_mixture_reg_comp.npy", comp1)
np.save(FOLDER + "kl_mixture_accl.npy", kl2)
np.save(FOLDER + "kl_mixture_accl_comp.npy", comp2)
np.save(FOLDER + "kl_mixture_accl_genli.npy", kl3)
np.save(FOLDER + "kl_mixture_accl_genli_comp.npy", comp3)
# print(kl)

## Plots
# plt.plot(n_value, kl1, 'bo--', label="regular")
# plt.plot(n_value, kl2, 'ro--', label="accelerated")
# plt.xlabel(r"$T$")
# plt.ylabel(r"$KL(Q_0||P_0)$")
# plt.yscale("log")
# plt.legend(loc='upper left')
# plt.title(r"$Q_0$ Mixture Gaussian")
# # plt.show()




###
