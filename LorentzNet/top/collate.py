import torch
import numpy as np
from math import sqrt
from . import randomSO13pTransform, randomLieGroupTransform

def batch_stack_general(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Unlike :batch_stack:, this will automatically stack scalars, vectors,
    and matrices. It will also automatically convert Numpy Arrays to
    Torch Tensors.

    Parameters
    ----------
    props : list or tuple of Pytorch Tensors, Numpy ndarrays, ints or floats.
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if type(props[0]) in [int, float]:
        # If batch is list of floats or ints, just create a new Torch Tensor.
        return torch.tensor(props)

    if type(props[0]) is np.ndarray:
        # Convert numpy arrays to tensors
        props = [torch.from_numpy(prop) for prop in props]

    shapes = [prop.shape for prop in props]

    if all(shapes[0] == shape for shape in shapes):
        # If all shapes are the same, stack along dim=0
        return torch.stack(props)

    elif all(shapes[0][1:] == shape[1:] for shape in shapes):
        # If shapes differ only along first axis, use the RNN pad_sequence to pad/stack.
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)

    elif all((shapes[0][2:] == shape[2:]) for shape in shapes):
        # If shapes differ along the first two axes, (shuch as a matrix),
        # pad/stack first two axes

        # Ensure that input features are matrices
        assert all((shape[0] == shape[1]) for shape in shapes), 'For batch stacking matrices, first two indices must match for every data point'

        max_atoms = max([len(p) for p in props])
        max_shape = (len(props), max_atoms, max_atoms) + props[0].shape[2:]
        padded_tensor = torch.zeros(max_shape, dtype=props[0].dtype, device=props[0].device)

        for idx, prop in enumerate(props):
            this_atoms = len(prop)
            padded_tensor[idx, :this_atoms, :this_atoms] = prop

        return padded_tensor
    else:
        ValueError('Input tensors must have the same shape on all but at most the first two axes!')




def batch_stack(props, edge_mat=False, nobj=None):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack
    edge_mat : bool
        The included tensor refers to edge properties, and therefore needs
        to be stacked/padded along two axes instead of just one.

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """

    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    elif not edge_mat:
        props = [p[:nobj, ...] for p in props]
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)
    else:
        max_atoms = max([len(p) for p in props])
        max_shape = (len(props), max_atoms, max_atoms) + props[0].shape[2:]
        padded_tensor = torch.zeros(max_shape, dtype=props[0].dtype, device=props[0].device)

        for idx, prop in enumerate(props):
            this_atoms = len(prop)
            padded_tensor[idx, :this_atoms, :this_atoms] = prop

        return padded_tensor


def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0 or props[0].shape[0] != to_keep.shape[0]:
        return props
    else:
        return props[:, to_keep, ...]

def normsq4(p):
    # Quick hack to calculate the norms of the four-vectors
    # The last dimension of the input gets eaten up
    psq = torch.pow(p, 2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)

from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import coo_matrix
import numpy as np

def get_adj_matrix(n_nodes, batch_size, edge_mask):
    rows, cols = [], []
    for batch_idx in range(batch_size):
        nn = batch_idx*n_nodes
        x = coo_matrix(edge_mask[batch_idx])
        rows.append(nn + x.row)
        cols.append(nn + x.col)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)

    edges = [torch.LongTensor(rows), torch.LongTensor(cols)]
    return edges

enc = OneHotEncoder().fit([[-1],[1]])

def collate_fn(data, scale=1., nobj=None, edge_features=[], add_beams=False, beam_mass=1, aug=False, aug_group=None, aug_var=1.0):
    """
    Collation function that collates datapoints into the batch format for lgn

    Parameters
    ----------
    data : list of datapoints
        The data to be collated.
    edge_features : list of strings
        Keys of properties that correspond to edge features, and therefore are
        matrices of shapes (num_atoms, num_atoms), which when forming a batch
        need to be padded along the first two axes instead of just the first one.

    Returns
    -------
    batch : dict of Pytorch tensors
        The collated data.
    """
    data = {prop: batch_stack([mol[prop] for mol in data], nobj=nobj) for prop in data[0].keys()}
    data['label'] = data['label'].to(torch.bool)

    # to_keep = batch['Nobj'].to(torch.uint8)
    to_keep = torch.any(data['label'],dim = 0)
    data = {key: drop_zeros(prop, to_keep) for key, prop in data.items()}

    if add_beams:
        beams = torch.tensor([[[sqrt(1+beam_mass**2),0,0,1], [sqrt(1+beam_mass**2),0,0,-1]]], dtype=data['Pmu'].dtype).expand(data['Pmu'].shape[0], 2, 4)
        s = data['Pmu'].shape
        data['Pmu'] = torch.cat([beams * scale, data['Pmu'] * scale], dim=1)
        labels = torch.cat((torch.ones(s[0], 2), -torch.ones(s[0], s[1])), dim=1)
        if 'scalars' not in data.keys():
            data['scalars'] = labels.to(dtype=data['Pmu'].dtype).unsqueeze(-1)
        else:
            data['scalars'] = torch.stack((data['scalars'], labels.to(dtype=data['Pmu'].dtype)))
    else:
        data['Pmu'] = data['Pmu'] * scale

    if aug:
        # data['Pmu']: (B, N or N+2, 4)
        if aug_group is None:
            data['Pmu'] = randomSO13pTransform(data['Pmu'], var=aug_var)
        else:
            data['Pmu'] = randomLieGroupTransform(aug_group, data['Pmu'], var=aug_var)

    # batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}
    atom_mask = data['Pmu'][...,0] != 0.
    #Obtain edges
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

    #mask diagonal
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask

    data['atom_mask'] = atom_mask.to(torch.bool)
    data['edge_mask'] = edge_mask.to(torch.bool)

    batch_size, n_nodes, _ = data['Pmu'].size()


    # Centralize Data
    # data['Pmu'] = data['Pmu'] - data['Pmu'].sum(dim=1, keepdim=True) / data['Nobj'][:,None,None]
    
    if add_beams:
        beamlabel = data['scalars']
        one_hot = enc.transform(beamlabel.reshape(-1,1)).toarray().reshape(batch_size,n_nodes,-1)
        one_hot = torch.tensor(one_hot)

        mass = normsq4(data['Pmu']).abs().sqrt().unsqueeze(-1) # [B,N,1]
        mass_tensor = mass.view(mass.shape + (1,))
        nodes = (one_hot.unsqueeze(-1) * mass_tensor).view(mass.shape[:2] + (-1,)) # [B,N,2]
    else:
        mass = normsq4(data['Pmu']).unsqueeze(-1)
        nodes = mass

    edges = get_adj_matrix(n_nodes, batch_size, data['edge_mask'])
    data['nodes'] = nodes
    data['edges'] = edges

    return data
