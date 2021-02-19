import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum

from lie_transformer_pytorch.se3 import SE3
from einops import rearrange, repeat

from lie_transformer_pytorch.reversible import SequentialSequence, ReversibleSequence

# helpers

def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else ((val,) * depth)

def default(val, d):
    return val if exists(val) else d

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

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
    def __init__(self, mean = False):
        super().__init__()
        self.mean = mean

    def forward(self, x):
        coords, vals, mask = x

        if not exists(mask):
            return val.mean(dim = 1)

        masked_vals = vals.masked_fill_(~mask[..., None], 0.)
        summed = masked_vals.sum(dim = 1)

        if not self.mean:
            return summed

        count = mask.sum(-1).unsqueeze(-1)
        return summed / count

# subsampling code

def FPSindices(dists, frac, mask):
    """ inputs: pairwise distances DISTS (bs,n,n), downsample_frac (float), valid atom mask (bs,n)
        outputs: chosen_indices (bs,m) """
    m = int(round(frac * dists.shape[1]))
    bs, n, device = *dists.shape[:2], dists.device
    dd_kwargs = {'device': device, 'dtype': torch.long}
    B = torch.arange(bs, **dd_kwargs)

    chosen_indices = torch.zeros(bs, m, **dd_kwargs)
    distances = torch.ones(bs, n, device=device) * 1e8
    a = torch.randint(0, n, (bs,), **dd_kwargs)            # choose random start
    idx = a % mask.sum(-1) + torch.cat([torch.zeros(1, **dd_kwargs), torch.cumsum(mask.sum(-1), dim=0)[:-1]], dim=0)
    farthest = torch.where(mask)[1][idx]

    for i in range(m):
        chosen_indices[:, i] = farthest                    # add point that is farthest to chosen
        dist = dists[B, farthest].masked_fill(~mask, -100) # (bs,n) compute distance from new point to all others
        closer = dist < distances                          # if dist from new point is smaller than chosen points so far
        distances[closer] = dist[closer]                   # update the chosen set's distance to all other points
        farthest = torch.max(distances, -1)[1]             # select the point that is farthest from the set

    return chosen_indices


class FPSsubsample(nn.Module):
    def __init__(self, ds_frac, cache = False, group = None):
        super().__init__()
        self.ds_frac = ds_frac
        self.cache = cache
        self.cached_indices = None
        self.group = default(group, SE3())

    def get_query_indices(self, abq_pairs, mask):
        if self.cache and exists(self.cached_indices):
            return self.cached_indices

        dist = self.group.distance if self.group else lambda ab: ab.norm(dim=-1)
        value = FPSindices(dist(abq_pairs), self.ds_frac, mask).detach()

        if self.cache:
            self.cached_indices = value

        return value

    def forward(self, inp, withquery=False):
        abq_pairs, vals, mask, edges = inp
        device = vals.device

        if self.ds_frac != 1:
            query_idx = self.get_query_indices(abq_pairs, mask)

            B = torch.arange(query_idx.shape[0], device = device).long()[:,None]
            subsampled_abq_pairs = abq_pairs[B, query_idx][B, :, query_idx]
            subsampled_values = batched_index_select(vals, query_idx, dim = 1)
            subsampled_mask = batched_index_select(mask, query_idx, dim = 1)
            subsampled_edges = edges[B, query_idx][B, :, query_idx] if exists(edges) else None
        else:
            subsampled_abq_pairs = abq_pairs
            subsampled_values = vals
            subsampled_mask = mask
            subsampled_edges = edges
            query_idx = None

        ret = (
            subsampled_abq_pairs,
            subsampled_values,
            subsampled_mask,
            subsampled_edges
        )

        if withquery:
            ret = (*ret, query_idx)

        return ret

# lie attention

class LieSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        edge_dim = None,
        group = None,
        mc_samples = 32,
        ds_frac = 1,
        fill = 1 / 3,
        dim_head = 64,
        heads = 8,
        cache = False
    ):
        super().__init__()
        self.dim = dim

        self.mc_samples = mc_samples # number of samples to use to estimate convolution
        self.group = default(group, SE3()) # Equivariance group for LieConv
        self.register_buffer('r',torch.tensor(2.)) # Internal variable for local_neighborhood radius, set by fill
        self.fill_frac = min(fill, 1.) # Average Fraction of the input which enters into local_neighborhood, determines r

        self.subsample = FPSsubsample(ds_frac, cache = cache, group = self.group)
        self.coeff = .5  # Internal coefficient used for updating r

        self.fill_frac_ema = fill # Keeps track of average fill frac, used for logging only

        # attention related parameters

        inner_dim = dim_head * heads
        self.heads = heads

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        edge_dim = default(edge_dim, 0)
        edge_dim_in = self.group.lie_dim + edge_dim

        self.loc_attn_mlp = nn.Sequential(
            nn.Linear(edge_dim_in, edge_dim_in * 4),
            nn.ReLU(),
            nn.Linear(edge_dim_in * 4, 1),
        )

    def extract_neighborhood(self, inp, query_indices):
        """ inputs: [pairs_abq (bs,n,n,d), inp_vals (bs,n,c), mask (bs,n), query_indices (bs,m)]
            outputs: [neighbor_abq (bs,m,mc_samples,d), neighbor_vals (bs,m,mc_samples,c)]"""

        # Subsample pairs_ab, inp_vals, mask to the query_indices
        pairs_abq, inp_vals, mask, edges = inp
        device = inp_vals.device

        if exists(query_indices):
            abq_at_query = batched_index_select(pairs_abq, query_indices, dim = 1)
            mask_at_query = batched_index_select(mask, query_indices, dim = 1)
            edges_at_query = batched_index_select(edges, query_indices, dim = 1) if exists(edges) else None
        else:
            abq_at_query = pairs_abq
            mask_at_query = mask
            edges_at_query = edges

        mask_at_query = mask_at_query[..., None]

        vals_at_query = inp_vals
        dists = self.group.distance(abq_at_query) #(bs,m,n,d) -> (bs,m,n)
        mask_value = torch.finfo(dists.dtype).max
        dists = dists.masked_fill(mask[:,None,:], mask_value)

        k = min(self.mc_samples, inp_vals.shape[1])

        # NBHD: Sampled Distance Ball
        bs, m, n = dists.shape
        within_ball = (dists < self.r) & mask[:,None,:] & mask_at_query # (bs,m,n)
        noise = torch.zeros((bs, m, n), device = device).uniform_(0, 1)
        valid_within_ball, nbhd_idx = torch.topk(within_ball + noise, k, dim=-1, sorted=False)
        valid_within_ball = (valid_within_ball > 1)

        # Retrieve abq_pairs, values, and mask at the nbhd locations

        nbhd_abq = batched_index_select(abq_at_query, nbhd_idx, dim = 2)
        nbhd_vals = batched_index_select(vals_at_query, nbhd_idx, dim = 1)
        nbhd_mask = batched_index_select(mask, nbhd_idx, dim = 1)
        nbhd_edges = batched_index_select(edges_at_query, nbhd_idx, dim = 2) if exists(edges) else None

        if self.training: # update ball radius to match fraction fill_frac inside
            navg = (within_ball.float()).sum(-1).sum() / mask_at_query.sum()
            avg_fill = (navg / mask.sum(-1).float().mean()).cpu().item()
            self.r +=  self.coeff * (self.fill_frac - avg_fill)
            self.fill_frac_ema += .1 * (avg_fill-self.fill_frac_ema)

        nbhd_mask &= valid_within_ball.bool()

        return nbhd_abq, nbhd_vals, nbhd_mask, nbhd_edges, nbhd_idx

    def forward(self, inp):
        """inputs: [pairs_abq (bs,n,n,d)], [inp_vals (bs,n,ci)]), [query_indices (bs,m)]
           outputs [subsampled_abq (bs,m,m,d)], [convolved_vals (bs,m,co)]"""
        sub_abq, sub_vals, sub_mask, sub_edges, query_indices = self.subsample(inp, withquery = True)
        nbhd_abq, nbhd_vals, nbhd_mask, nbhd_edges, nbhd_indices = self.extract_neighborhood(inp, query_indices)

        h, b, n, d, device = self.heads, *sub_vals.shape, sub_vals.device

        q, k, v = self.to_q(sub_vals), self.to_k(nbhd_vals), self.to_v(nbhd_vals)

        q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        k, v = map(lambda t: rearrange(t, 'b n m (h d) -> b h n m d', h = h), (k, v))

        sim = einsum('b h i d, b h i j d -> b h i j', q, k) * (q.shape[-1] ** -0.5)

        edges = nbhd_abq
        if exists(nbhd_edges):
            edges = torch.cat((nbhd_abq, nbhd_edges), dim = -1)

        loc_attn = self.loc_attn_mlp(edges)
        loc_attn = rearrange(loc_attn, 'b i j () -> b () i j')
        sim = sim + loc_attn

        mask_value = -torch.finfo(sim.dtype).max

        sim.masked_fill_(~rearrange(nbhd_mask, 'b n m -> b () n m'), mask_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h i j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        combined = self.to_out(out)

        return sub_abq, combined, sub_mask, sub_edges

class LieSelfAttentionWrapper(nn.Module):
    def __init__(self, dim, attn):
        super().__init__()
        self.dim = dim
        self.attn = attn

        self.net = nn.Sequential(
            Pass(nn.LayerNorm(dim)),
            self.attn
        )

    def forward(self, inp):
        sub_coords, sub_values, mask, edges = self.attn.subsample(inp)
        new_coords, new_values, mask, edges = self.net(inp)
        new_values[..., :self.dim] += sub_values
        return new_coords, new_values, mask, edges

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
        sub_coords, sub_values, mask, edges = inp
        new_coords, new_values, mask, edges = self.net(inp)
        new_values = new_values + sub_values
        return new_coords, new_values, mask, edges

# transformer class

class LieTransformer(nn.Module):
    """
    [Fill] specifies the fraction of the input which is included in local neighborhood.
            (can be array to specify a different value for each layer)
    [nbhd] number of samples to use for Monte Carlo estimation (p)
    [dim] number of input channels: 1 for MNIST, 3 for RGB images, other for non images
    [ds_frac] total downsampling to perform throughout the layers of the net. In (0,1)
    [num_layers] number of BottleNeck Block layers in the network
    [k] channel width for the network. Can be int (same for all) or array to specify individually.
    [liftsamples] number of samples to use in lifting. 1 for all groups with trivial stabilizer. Otherwise 2+
    [Group] Chosen group to be equivariant to.
    """
    def __init__(
        self,
        dim,
        num_tokens = None,
        num_edge_types = None,
        edge_dim = None,
        heads = 8,
        dim_head = 64,
        depth = 2,
        ds_frac = 1.,
        dim_out = None,
        k = 1536,
        nbhd = 128,
        mean = True,
        per_point = True,
        liftsamples = 4,
        fill = 1 / 4,
        cache = False,
        reversible = False,
        **kwargs
    ):
        super().__init__()
        assert not (ds_frac < 1 and reversible), 'must not downsample if network is reversible'

        dim_out = default(dim_out, dim)
        self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None
        self.edge_emb = nn.Embedding(num_edge_types, edge_dim) if exists(num_edge_types) else None

        group = SE3()
        self.group = group
        self.liftsamples = liftsamples

        layers_fill = cast_tuple(fill, depth)
        layers = nn.ModuleList([])

        for _, layer_fill in zip(range(depth), layers_fill):
            layers.append(nn.ModuleList([
                LieSelfAttentionWrapper(dim, LieSelfAttention(dim, heads = heads, dim_head = dim_head, edge_dim = edge_dim, mc_samples = nbhd, ds_frac = ds_frac, group = group, fill = fill, cache = cache,**kwargs)),
                FeedForward(dim)
            ]))

        execute_class = ReversibleSequence if reversible else SequentialSequence
        self.net = execute_class(layers)

        self.to_logits = nn.Sequential(
            Pass(nn.LayerNorm(dim)),
            Pass(nn.Linear(dim, dim_out))
        )

        self.pool = GlobalPool(mean = mean)

    def forward(self, feats, coors, edges = None, mask = None, return_pooled = False):
        b, n, *_ = feats.shape

        if exists(self.token_emb):
            feats = self.token_emb(feats)

        if exists(self.edge_emb):
            assert exists(edges), 'edges must be passed in on forward'
            assert edges.shape[1] == edges.shape[2] and edges.shape[1] == n, f'edges must be of the shape ({b}, {n}, {n})'
            edges = self.edge_emb(edges)

        inps = (coors, feats, mask, edges)

        lifted_x = self.group.lift(inps, self.liftsamples)
        out = self.net(lifted_x)

        out = self.to_logits(out)

        if not return_pooled:
            features = out[1]
            return features

        return self.pool(out)
