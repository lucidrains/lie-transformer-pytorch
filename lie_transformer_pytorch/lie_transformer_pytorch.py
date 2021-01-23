import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum

from lie_transformer_pytorch.se3 import SE3
from einops import rearrange, repeat

# constants

TOKEN_SELF_ATTN_VALUE = -5e4 # carefully set for half precision to work

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# helper classes

class Pass(nn.Module):
    def __init__(self, fn, dim = 1):
        super().__init__()
        self.fn = fn
        self.dim = dim

    def forward(self,x):
        dim = self.dim
        xs = list(x)
        xs[dim] = self.fn(xs[dim])
        return xs

class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class GlobalPool(nn.Module):
    """computes values reduced over all spatial locations (& group elements) in the mask"""
    def __init__(self,mean=False):
        super().__init__()
        self.mean = mean

    def forward(self,x):
        """x [xyz (bs,n,d), vals (bs,n,c), mask (bs,n)]"""
        if len(x)==2: return x[1].mean(1)
        coords, vals, mask = x
        summed = vals.masked_fill_(~mask[..., None], 0.).sum(dim = 1)
        if self.mean:
            summed /= mask.sum(-1).unsqueeze(-1)
        return summed

# subsampling code

def FPSindices(dists,frac,mask):
    """ inputs: pairwise distances DISTS (bs,n,n), downsample_frac (float), valid atom mask (bs,n)
        outputs: chosen_indices (bs,m) """
    m = int(round(frac * dists.shape[1]))
    device = dists.device
    bs,n = dists.shape[:2]
    chosen_indices = torch.zeros(bs, m, dtype=torch.long,device=device)
    distances = torch.ones(bs, n,device=device) * 1e8
    a = torch.randint(0, n, (bs,), dtype=torch.long,device=device) #choose random start
    idx = a%mask.sum(-1) + torch.cat([torch.zeros(1,device=device).long(), torch.cumsum(mask.sum(-1),dim=0)[:-1]],dim=0)
    farthest = torch.where(mask)[1][idx]
    B = torch.arange(bs, dtype=torch.long,device=device)\

    for i in range(m):
        chosen_indices[:, i] = farthest # add point that is farthest to chosen
        dist = dists[B,farthest].masked_fill(mask, -100) # (bs,n) compute distance from new point to all others
        closer = dist < distances      # if dist from new point is smaller than chosen points so far
        distances[closer] = dist[closer] # update the chosen set's distance to all other points
        farthest = torch.max(distances, -1)[1] # select the point that is farthest from the set

    return chosen_indices


class FPSsubsample(nn.Module):
    def __init__(self, ds_frac, cache=False, group=None):
        super().__init__()
        self.ds_frac = ds_frac
        self.cache=cache
        self.cached_indices = None
        self.group = group

    def forward(self,inp,withquery=False):
        abq_pairs,vals,mask = inp
        device = vals.device

        dist = self.group.distance if self.group else lambda ab: ab.norm(dim=-1)

        if self.ds_frac!=1:
            if self.cache and self.cached_indices is None:
                query_idx = self.cached_indices = FPSindices(dist(abq_pairs),self.ds_frac,mask).detach()
            elif self.cache:
                query_idx = self.cached_indices
            else:
                query_idx = FPSindices(dist(abq_pairs),self.ds_frac,mask)
            B = torch.arange(query_idx.shape[0], device = device).long()[:,None]
            subsampled_abq_pairs = abq_pairs[B,query_idx][B,:,query_idx]
            subsampled_values = vals[B,query_idx]
            subsampled_mask = mask[B,query_idx]
        else:
            subsampled_abq_pairs = abq_pairs
            subsampled_values = vals
            subsampled_mask = mask
            query_idx = None

        ret = (subsampled_abq_pairs,subsampled_values,subsampled_mask)

        if withquery:
            ret = (*ret, query_idx)

        return ret

# lie attention

class LieSelfAttention(nn.Module):
    def __init__(
        self,
        chin,
        mc_samples=32,
        xyz_dim=3,
        ds_frac=1,
        loc_attn = False,
        knn_channels=None,
        mean=False,
        group=SE3,
        fill=1/3,
        cache=False,
        knn=False,
        dim_head = 64,
        heads = 8,
        attend_self = True,
        **kwargs
    ):
        super().__init__()
        self.chin = chin # input channels
        self.cmco_ci = 16 # a hyperparameter controlling size and bottleneck compute cost of weightnet
        self.xyz_dim = xyz_dim # dimension of the space on which convolution operates
        self.knn_channels = knn_channels # number of xyz dims on which to compute knn
        self.mc_samples = mc_samples # number of samples to use to estimate convolution

        self.mean=mean  # Whether or not to divide by the number of mc_samples

        self.group = group # Equivariance group for LieConv
        self.register_buffer('r',torch.tensor(2.)) # Internal variable for local_neighborhood radius, set by fill
        self.fill_frac = min(fill,1.) # Average Fraction of the input which enters into local_neighborhood, determines r
        self.knn=knn            # Whether or not to use the k nearest points instead of random samples for conv estimator

        self.subsample = FPSsubsample(ds_frac, cache=cache, group=self.group)
        self.coeff = .5  # Internal coefficient used for updating r
        self.fill_frac_ema = fill # Keeps track of average fill frac, used for logging only

        inner_dim = dim_head * heads
        self.heads = heads
        self.attend_self = attend_self

        self.to_q = nn.Linear(chin, inner_dim, bias = False)
        self.to_k = nn.Linear(chin, inner_dim, bias = False)
        self.to_v = nn.Linear(chin, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, chin)

        self.loc_attn_mlp = nn.Sequential(
            nn.Linear(self.group.lie_dim, self.group.lie_dim * 4),
            nn.ReLU(),
            nn.Linear(self.group.lie_dim * 4, 1),
        ) if loc_attn else None

    def extract_neighborhood(self,inp,query_indices):
        """ inputs: [pairs_abq (bs,n,n,d), inp_vals (bs,n,c), mask (bs,n), query_indices (bs,m)]
            outputs: [neighbor_abq (bs,m,mc_samples,d), neighbor_vals (bs,m,mc_samples,c)]"""

        # Subsample pairs_ab, inp_vals, mask to the query_indices
        pairs_abq, inp_vals, mask = inp
        device = inp_vals.device

        if query_indices is not None:
            B = torch.arange(inp_vals.shape[0], device = device).long()[:,None]
            abq_at_query = pairs_abq[B,query_indices]
            mask_at_query = mask[B,query_indices]
        else:
            abq_at_query = pairs_abq
            mask_at_query = mask
        vals_at_query = inp_vals
        dists = self.group.distance(abq_at_query) #(bs,m,n,d) -> (bs,m,n)
        dists = dists.masked_fill(mask[:,None,:].expand(*dists.shape), 1e8)

        k = min(self.mc_samples,inp_vals.shape[1])

        # Determine ids (and mask) for points sampled within neighborhood (A4)
        if self.knn: # NBHD: KNN
            nbhd_idx = torch.topk(dists,k,dim=-1,largest=False,sorted=False)[1] #(bs,m,nbhd)
            valid_within_ball = (nbhd_idx>-1)&mask[:,None,:]&mask_at_query[:,:,None]
            assert not torch.any(nbhd_idx>dists.shape[-1]), f"error with topk,\
                        nbhd{k} nans|inf{torch.any(torch.isnan(dists)|torch.isinf(dists))}"
        else: # NBHD: Sampled Distance Ball
            bs,m,n = dists.shape
            within_ball = (dists < self.r)&mask[:,None,:]&mask_at_query[:,:,None] # (bs,m,n)
            B = torch.arange(bs)[:,None,None]
            M = torch.arange(m)[None,:,None]
            noise = torch.zeros(bs,m,n, device = device)
            noise.uniform_(0,1)
            valid_within_ball, nbhd_idx =torch.topk(within_ball+noise,k,dim=-1,largest=True,sorted=False)
            valid_within_ball = (valid_within_ball>1)

        # Retrieve abq_pairs, values, and mask at the nbhd locations
        B = torch.arange(inp_vals.shape[0], device = device).long()[:,None,None].expand(*nbhd_idx.shape)
        M = torch.arange(abq_at_query.shape[1], device = device).long()[None,:,None].expand(*nbhd_idx.shape)
        nbhd_abq = abq_at_query[B,M,nbhd_idx]     #(bs,m,n,d) -> (bs,m,mc_samples,d)
        nbhd_vals = vals_at_query[B,nbhd_idx]   #(bs,n,c) -> (bs,m,mc_samples,c)
        nbhd_mask = mask[B,nbhd_idx]            #(bs,n) -> (bs,m,mc_samples)

        if self.training and not self.knn: # update ball radius to match fraction fill_frac inside
            navg = (within_ball.float()).sum(-1).sum()/mask_at_query[:,:,None].sum()
            avg_fill = (navg/mask.sum(-1).float().mean()).cpu().item()
            self.r +=  self.coeff*(self.fill_frac - avg_fill)
            self.fill_frac_ema += .1*(avg_fill-self.fill_frac_ema)
        return nbhd_abq, nbhd_vals, (nbhd_mask&valid_within_ball.bool()), nbhd_idx

    def forward(self, inp):
        """inputs: [pairs_abq (bs,n,n,d)], [inp_vals (bs,n,ci)]), [query_indices (bs,m)]
           outputs [subsampled_abq (bs,m,m,d)], [convolved_vals (bs,m,co)]"""
        sub_abq, sub_vals, sub_mask, query_indices = self.subsample(inp, withquery = True)
        nbhd_abq, nbhd_vals, nbhd_mask, nbhd_indices = self.extract_neighborhood(inp, query_indices)

        h, b, n, d, device = self.heads, *sub_vals.shape, sub_vals.device

        q, k, v = self.to_q(sub_vals), self.to_k(nbhd_vals), self.to_v(nbhd_vals)

        q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        k, v = map(lambda t: rearrange(t, 'b n m (h d) -> b h n m d', h = h), (k, v))

        sim = einsum('b h i d, b h i j d -> b h i j', q, k) * (q.shape[-1] ** -0.5)

        if exists(self.loc_attn_mlp):
            loc_attn = self.loc_attn_mlp(nbhd_abq)
            loc_attn = rearrange(loc_attn, 'b i j () -> b () i j')
            sim = sim + loc_attn

        mask_value = -torch.finfo(sim.dtype).max

        if not self.attend_self:
            seq = torch.arange(n, device = device)
            seq = rearrange(seq, 'n -> () n ()')
            mask = (seq == nbhd_indices)
            sim.masked_fill_(mask, TOKEN_SELF_ATTN_VALUE)

        sim.masked_fill_(~rearrange(nbhd_mask, 'b n m -> b () n m'), mask_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h i j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        combined = self.to_out(out)

        return sub_abq, combined, sub_mask

class LieSelfAttentionWrapper(nn.Module):
    """ A bottleneck residual block as described in figure 5"""
    def __init__(self,chin, attn, fill=None):
        super().__init__()
        self.attn = attn()

        self.net = nn.Sequential(
            Pass(nn.LayerNorm(chin)),
            self.attn
        )

        self.chin = chin

    def forward(self,inp):
        sub_coords, sub_values, mask = self.attn.subsample(inp)
        new_coords, new_values, mask = self.net(inp)
        new_values[...,:self.chin] += sub_values
        return new_coords, new_values, mask

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.dim = dim

        self.net = nn.Sequential(
            Pass(nn.LayerNorm(dim)),
            Pass(nn.Linear(dim, mult * dim)),
            Pass(nn.GELU()),
            Pass(nn.Linear(mult * dim, dim)),
        )

    def forward(self,inp):
        sub_coords, sub_values, mask = inp
        new_coords, new_values, mask = self.net(inp)
        new_values = new_values + sub_values
        return new_coords, new_values, mask

# transformer class

class LieTransformer(nn.Module):
    """
        [Fill] specifies the fraction of the input which is included in local neighborhood.
                (can be array to specify a different value for each layer)
        [nbhd] number of samples to use for Monte Carlo estimation (p)
        [chin] number of input channels: 1 for MNIST, 3 for RGB images, other for non images
        [ds_frac] total downsampling to perform throughout the layers of the net. In (0,1)
        [num_layers] number of BottleNeck Block layers in the network
        [k] channel width for the network. Can be int (same for all) or array to specify individually.
        [liftsamples] number of samples to use in lifting. 1 for all groups with trivial stabilizer. Otherwise 2+
        [Group] Chosen group to be equivariant to.
        """
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        depth = 2,
        loc_attn = False,
        ds_frac = 1,
        dim_out = None,
        k = 1536,
        nbhd = 128,
        mean = True,
        per_point = True,
        liftsamples = 4,
        fill = 1/4,
        knn = False,
        cache = False,
        **kwargs
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        if isinstance(fill,(float,int)):
            fill = [fill] * depth

        group = SE3()
        self.group = group
        self.liftsamples = liftsamples

        block = lambda dim, fill: nn.Sequential(
            LieSelfAttentionWrapper(dim, attn = partial(LieSelfAttention, dim, heads = heads, dim_head = dim_head, loc_attn = loc_attn, mc_samples=nbhd, ds_frac=ds_frac, mean=mean, group=group,fill=fill,cache=cache,knn=knn,**kwargs), fill=fill),
            FeedForward(dim)
        )

        self.net = nn.Sequential(
            Pass(nn.Linear(dim, dim)), #embedding layer
            *[block(dim, fill[i]) for i in range(depth)],
            Pass(nn.LayerNorm(dim)),
            Pass(nn.Linear(dim, dim_out))
        )

        self.pool = GlobalPool(mean=mean)

    def forward(self, x, pool = False):
        lifted_x = self.group.lift(x, self.liftsamples)
        out = self.net(lifted_x)

        if not pool:
            _, features, _ = out
            return features

        return self.pool(out)
