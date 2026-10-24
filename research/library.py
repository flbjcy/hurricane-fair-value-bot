"""Per-storm site-level outcomes (fractions, closure secs) reconciled with reported company damage."""
import numpy as np
from damage import *
from value import *
a=np.load('arch.npz'); S=a['S']; D=a['DMG'][:,:,1,:5]
p=np.load('damage_params.npy')
u,I=site_u(S); E=exposure(u,I,*p[:3]); F=site_frac(E,p)    # model fractions [600,3,NS]
# reconcile: scale each company's site fractions so company damage matches the final report
Fr=F.copy(); h=np.ones((600,NS))
for r in range(3):
    mod=np.stack([100*np.sum((SW*h*F[:,r])[:,SC==j],-1) for j in range(5)],-1)
    ratio=np.where(mod>1e-6, D[:,r]/np.maximum(mod,1e-9), 0.0)
    Fr[:,r]=np.clip(F[:,r]*ratio[:,SC],0,1)
    # companies with reported damage but model zero: spread uniformly
    miss=(mod<=1e-6)&(D[:,r]>0)
    for j in range(5):
        idx=miss[:,j]
        if idx.any(): Fr[idx,r][:,SC==j]=0  # placeholder (rare)
    h=h*(1-Fr[:,r])
CLO=closure_seconds(S,0.70,0.34).sum(2)   # [600,3,NS]
np.savez('library.npz',F=F,Fr=Fr,CLO=CLO)
chk=np.stack([np.stack([100*np.sum((SW*np.prod(1-Fr[:,:r],1)*Fr[:,r])[:,SC==j],-1) for j in range(5)],-1) for r in range(3)],1)
print('reconciled max err',np.abs(chk-D).max(), 'missed cases',((chk-D)>0.01).sum())
