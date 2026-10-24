"""Expected settlement value given the current state, by Monte Carlo over a library of past storms."""
import numpy as np
from damage import SW, SC, NS
from value import value, company_closure, LOSS_B
def storm_effect(h, Fs, clos):
    """h[...,NS] health; Fs[K,NS] storm fractions; returns added company dmg [...,K,5], closure [...,K,5], new health"""
    add = h[..., None, :]*Fs
    d = 100*np.stack([np.sum((SW*add)[..., SC == j], -1) for j in range(5)], -1)
    return d, company_closure(clos), h[..., None, :] - add
class EV:
    def __init__(self, F, CLO, n_pairs=3000, seed=0):
        self.F = F.reshape(-1, NS); self.C = CLO.reshape(-1, NS)
        rng = np.random.default_rng(seed); K = len(self.F)
        self.pi = rng.integers(0, K, n_pairs); self.pj = rng.integers(0, K, n_pairs)
    def samples(self, h, cum, clo, L, n_future):
        """returns value samples [M,10] for one state"""
        if n_future == 0: return value(cum, clo, L)[None]
        d1, c1, h1 = storm_effect(h, self.F, self.C)
        if n_future == 1:
            c = cum + d1; return value(c, clo + c1, L + d1@LOSS_B)
        i, j = self.pi, self.pj
        dA, cA = d1[i], c1[i]; hA = h1[i]
        addB = hA*self.F[j]
        dB = 100*np.stack([np.sum((SW*addB)[:, SC == k], -1) for k in range(5)], -1)
        cB = company_closure(self.C[j])
        c = cum + dA + dB
        return value(c, clo + cA + cB, L + (dA + dB)@LOSS_B)
    def mean(self, *a): return self.samples(*a).mean(0)
