from math import pi
import torch
from functools import wraps
from torch import acos, atan2, cos, sin
from einops import rearrange, repeat

# constants

THRES = 7e-2

# helper functions

def exists(val):
    return val is not None

def to(t):
    return {'device': t.device, 'dtype': t.dtype}

def taylor(thres):
    def outer(fn):
        @wraps(fn)
        def inner(x):
            usetaylor = x.abs() < THRES
            taylor_expanded, full = fn(x, x * x)
            return torch.where(usetaylor, taylor_expanded, full)
        return inner
    return outer

# Helper functions for analytic exponential maps. Uses taylor expansions near x=0
# See http://ethaneade.com/lie_groups.pdf for derivations.

@taylor(THRES)
def sinc(x, x2):
    """ sin(x)/x """
    texpand = 1-x2/6*(1-x2/20*(1-x2/42))
    full = sin(x) / x
    return texpand, full

@taylor(THRES)
def sincc(x, x2):
    """ (1-sinc(x))/x^2"""
    texpand = 1/6*(1-x2/20*(1-x2/42*(1-x2/72)))
    full = (x-sin(x)) / x**3
    return texpand, full

@taylor(THRES)
def cosc(x, x2):
    """ (1-cos(x))/x^2"""
    texpand = 1/2*(1-x2/12*(1-x2/30*(1-x2/56)))
    full = (1-cos(x)) / x2
    return texpand, full

@taylor(THRES)
def coscc(x, x2):
    #assert not torch.any(torch.isinf(x2)), f"infs in x2 log"
    texpand = 1/12*(1+x2/60*(1+x2/42*(1+x2/40)))
    costerm = (2*(1-cos(x))).clamp(min=1e-6)
    full = (1-x*sin(x)/costerm) / x2 #Nans can come up here when cos = 1
    return texpand, full

@taylor(THRES)
def sinc_inv(x, _):
    texpand = 1+(1/6)*x**2 +(7/360)*x**4
    full = x / sin(x)
    assert not torch.any(torch.isinf(texpand)|torch.isnan(texpand)),'sincinv texpand inf'+torch.any(torch.isinf(texpand))
    return texpand, full

## Lie Groups acting on R3

# Hodge star on R3
def cross_matrix(k):
    """Application of hodge star on R3, mapping Λ^1 R3 -> Λ^2 R3"""
    K = torch.zeros(*k.shape[:-1], 3, 3, **to(k))
    K[...,0,1] = -k[...,2]
    K[...,0,2] = k[...,1]
    K[...,1,0] = k[...,2]
    K[...,1,2] = -k[...,0]
    K[...,2,0] = -k[...,1]
    K[...,2,1] = k[...,0]
    return K

def uncross_matrix(K):
    """Application of hodge star on R3, mapping Λ^2 R3 -> Λ^1 R3"""
    k = torch.zeros(*K.shape[:-1], **to(K))
    k[...,0] = (K[...,2,1] - K[...,1,2])/2
    k[...,1] = (K[...,0,2] - K[...,2,0])/2
    k[...,2] = (K[...,1,0] - K[...,0,1])/2
    return k

class SO3:
    lie_dim = 3
    rep_dim = 3
    q_dim = 1

    def __init__(self, alpha = .2):
        super().__init__()
        self.alpha = alpha
    
    def exp(self,w):
        """ Computes (matrix) exponential Lie algebra elements (in a given basis).
            ie out = exp(\sum_i a_i A_i) where A_i are the exponential generators of G.
            Input: [a (*,lie_dim)] where * is arbitrarily shaped
            Output: [exp(a) (*,rep_dim,rep_dim)] returns the matrix for each."""

        """ Rodriguez's formula, assuming shape (*,3)
            where components 1,2,3 are the generators for xrot,yrot,zrot"""
        theta = w.norm(dim=-1)[..., None, None]
        K = cross_matrix(w)
        I = torch.eye(3, **to(K))
        Rs = I + K * sinc(theta) + (K @ K) * cosc(theta)
        return Rs
    
    def log(self,R):
        """ Computes components in terms of generators rx,ry,rz. Shape (*,3,3)"""

        """ Computes (matrix) logarithm for collection of matrices and converts to Lie algebra basis.
            Input [u (*,rep_dim,rep_dim)]
            Output [coeffs of log(u) in basis (*,d)] """
        trR = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        costheta = ((trR-1) / 2).clamp(max=1, min=-1).unsqueeze(-1)
        theta = acos(costheta)
        logR = uncross_matrix(R) * sinc_inv(theta)
        return logR

    def inv(self,g):
        """ We can compute the inverse of elements g (*,rep_dim,rep_dim) as exp(-log(g))"""
        return self.exp(-self.log(g))

    def elems2pairs(self,a):
        """ computes log(e^-b e^a) for all a b pairs along n dimension of input.
            inputs: [a (bs,n,d)] outputs: [pairs_ab (bs,n,n,d)] """
        vinv = self.exp(-a.unsqueeze(-3))
        u = self.exp(a.unsqueeze(-2))
        return self.log(vinv@u)    # ((bs,1,n,d) -> (bs,1,n,r,r))@((bs,n,1,d) -> (bs,n,1,r,r))

    def lift(self, x, nsamples, **kwargs):
        """assumes p has shape (*,n,2), vals has shape (*,n,c), mask has shape (*,n)
            returns (a,v) with shapes [(*,n*nsamples,lie_dim),(*,n*nsamples,c)"""
        p, v, m, e = x
        expanded_a = self.lifted_elems(p,nsamples,**kwargs) # (bs,n*ns,d), (bs,n*ns,qd)
        nsamples = expanded_a.shape[-2]//m.shape[-1]
        # expand v and mask like q
        expanded_v = repeat(v, 'b n c -> b (n m) c', m = nsamples) # (bs,n,c) -> (bs,n,1,c) -> (bs,n,ns,c) -> (bs,n*ns,c)
        expanded_mask = repeat(m, 'b n -> b (n m)', m = nsamples) # (bs,n) -> (bs,n,ns) -> (bs,n*ns)
        expanded_e = repeat(e, 'b n1 n2 c -> b (n1 m1) (n2 m2) c', m1 = nsamples, m2 = nsamples) if exists(e) else None

        # convert from elems to pairs
        paired_a = self.elems2pairs(expanded_a) #(bs,n*ns,d) -> (bs,n*ns,n*ns,d)
        embedded_locations = paired_a
        return (embedded_locations,expanded_v,expanded_mask, expanded_e)

class SE3(SO3):
    lie_dim = 6
    rep_dim = 4
    q_dim = 0

    def __init__(self, alpha=.2, per_point=True):
        super().__init__()
        self.alpha = alpha
        self.per_point = per_point

    def exp(self,w):
        dd_kwargs = to(w)
        theta = w[...,:3].norm(dim=-1)[...,None,None]
        K = cross_matrix(w[...,:3])
        R = super().exp(w[...,:3])
        I = torch.eye(3, **dd_kwargs)
        V = I + cosc(theta)*K + sincc(theta)*(K@K)
        U = torch.zeros(*w.shape[:-1],4,4, **dd_kwargs)
        U[...,:3,:3] = R
        U[...,:3,3] = (V@w[...,3:].unsqueeze(-1)).squeeze(-1)
        U[...,3,3] = 1
        return U
    
    def log(self,U):
        w = super().log(U[..., :3, :3])
        I = torch.eye(3, **to(w))
        K = cross_matrix(w[..., :3])
        theta = w.norm(dim=-1)[..., None, None]#%(2*pi)
        #theta[theta>pi] -= 2*pi
        cosccc = coscc(theta)
        Vinv = I - K/2 + cosccc*(K@K)
        u = (Vinv @ U[..., :3, 3].unsqueeze(-1)).squeeze(-1)
        #assert not torch.any(torch.isnan(u)), f"nans in u log {torch.isnan(u).sum()}, {torch.where(torch.isnan(u))}"
        return torch.cat([w, u], dim=-1)

    def lifted_elems(self,pt,nsamples):
        """ pt (bs,n,D) mask (bs,n), per_point specifies whether to
            use a different group element per atom in the molecule"""
        #return farthest_lift(self,pt,mask,nsamples,alpha)
        # same lifts for each point right now
        bs,n = pt.shape[:2]
        dd_kwargs = to(pt)

        q = torch.randn(bs, (n if self.per_point else 1), nsamples, 4, **dd_kwargs)
        q /= q.norm(dim=-1).unsqueeze(-1)

        theta_2 = atan2(q[..., 1:].norm(dim=-1),q[..., 0])[..., None]
        so3_elem = 2 * sinc_inv(theta_2) * q[...,1:] # (sin(x/2)u -> xu) for x angle and u direction
        se3_elem = torch.cat([so3_elem, torch.zeros_like(so3_elem)], dim=-1)
        R = self.exp(se3_elem)

        T = torch.zeros(bs, n, nsamples, 4, 4, **dd_kwargs) # (bs,n,nsamples,4,4)
        T[..., :, :] = torch.eye(4, **dd_kwargs)
        T[..., :3, 3] = pt[..., None, :] # (bs,n,1,3)

        a = self.log(T @ R) # bs, n, nsamples, 6
        return a.reshape(bs, n * nsamples, 6)

    def distance(self,abq_pairs):
        dist_rot = abq_pairs[...,:3].norm(dim=-1)
        dist_trans = abq_pairs[...,3:].norm(dim=-1)
        return dist_rot * self.alpha + (1-self.alpha) * dist_trans
