"""Site-level physical damage model fitted from storm observations -> final damage reports."""
import numpy as np, json
from scipy.optimize import least_squares
import os
C=json.load(open(os.path.join(os.environ.get('HS_DATA','../data'),'case.json')))
PHYS=['PORT','ISLE','AQUA','GRID','FERRY']
SX=[];SY=[];SW=[];SC=[];SI=[]
ISL=['cascadia_prime','coral_key','azure_isle','palm_atoll']
for j,c in enumerate(PHYS):
    for s in C['companies'][c]['sites']:
        SX.append(s['x']);SY.append(s['y']);SW.append(s['weight']);SC.append(j);SI.append(ISL.index(s['island']))
SX,SY,SW,SC,SI=map(np.array,(SX,SY,SW,SC,SI))
NS=len(SX)
def site_u(S):
    """S[...,241,6] -> u [...,241,NS], I [...,241]"""
    d=np.hypot(S[...,0,None]-SX,S[...,1,None]-SY)
    return d/S[...,4,None], S[...,5]
def exposure(u,I,sig,thr,pw):
    w=I[...,None]*np.exp(-.5*(u/sig)**2)
    return np.nansum(np.maximum(w-thr,0)**pw,axis=-2)*2      # [...,NS]
def unpack(p):
    sig,thr,pw=p[:3]; k=p[3:8]; fmax=p[8:13]; return sig,thr,pw,k,fmax
def site_frac(E,p):
    sig,thr,pw,k,fmax=unpack(p)
    return fmax[SC]*(1-np.exp(-k[SC]*E))
def company_damage(F):
    """F [season,3,NS] storm fractions -> reported dmg% per storm [season,3,5] and health"""
    h=np.ones(F.shape[:-2]+(NS,)); out=np.zeros(F.shape[:-1]+(5,)) ; H=np.zeros(F.shape)
    for r in range(3):
        H[...,r,:]=h
        add=h*F[...,r,:]
        out[...,r,:]=100*np.stack([np.sum((SW*add)[...,SC==j],-1) for j in range(5)],-1)
        h=h-add
    return out,H
P0=np.array([0.72,0.01,0.9]+[0.01]*5+[1.0]*5)
LB=[.2,0,.2]+[1e-6]*5+[0.01]*5; UB=[5,2,4]+[10]*5+[1]*5
def fit(S,D,p0=P0):
    u,I=site_u(S)
    def res(p):
        E=exposure(u,I,*p[:3]); out,_=company_damage(site_frac(E,p)); return (out-D).ravel()
    r=least_squares(res,p0,bounds=(LB,UB),x_scale='jac')
    return r.x
if __name__=='__main__':
    a=np.load('arch.npz'); S=a['S']; D=a['DMG'][:,:,1,:5]
    p=fit(S[:350],D[:350]); print(np.round(p,4))
    np.save('damage_params.npy',p)
    u,I=site_u(S); E=exposure(u,I,*p[:3]); out,_=company_damage(site_frac(E,p))
    for nm,sl in [('train',slice(0,350)),('val',slice(350,450)),('test',slice(450,600))]:
        e=out[sl]-D[sl]; print(nm,'rmse by company',np.sqrt((e**2).mean((0,1))).round(2),'R2',(1-e.var((0,1))/D[sl].var((0,1))).round(3))
