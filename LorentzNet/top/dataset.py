from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from . import collate_fn, initialize_datasets

def retrieve_dataloaders(batch_size, num_workers = 4, num_train = -1, datadir = './data', aug=False, aug_group=None, aug_var=1.0):
    # Initialize dataloader
    datasets = initialize_datasets(datadir, num_pts={'train':num_train,'test':-1,'valid':-1})
    # distributed training
    train_sampler = DistributedSampler(datasets['train'])
    # Construct PyTorch dataloaders from datasets
    collate_train = lambda data: collate_fn(data, scale=1, add_beams=True, beam_mass=1, aug=aug, aug_group=aug_group, aug_var=aug_var)
    collate_test = lambda data: collate_fn(data, scale=1, add_beams=True, beam_mass=1)
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=batch_size if (split == 'train') else batch_size, # prevent CUDA memory exceeded
                                     sampler=train_sampler if (split == 'train') else DistributedSampler(dataset, shuffle=False),
                                     pin_memory=True,
                                     persistent_workers=True,
                                     drop_last= True if (split == 'train') else False,
                                     num_workers=num_workers,
                                     collate_fn=collate_train if split == 'train' else collate_test)
                        for split, dataset in datasets.items()}

    return train_sampler, dataloaders