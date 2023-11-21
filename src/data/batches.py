import torch
from torch.utils import data

from src.utils.distributed import is_distributedSetup

from torch.utils.data.distributed import DistributedSampler


def _create_dataLoader(pytorch_dataset, batch_size, should_shuffle, world_size, device):
    if is_distributedSetup(world_size):
        sampler = DistributedSampler(
            pytorch_dataset,
            num_replicas=world_size,
            rank=device,
            shuffle=should_shuffle,
        )

        data_loader = data.DataLoader(
            pytorch_dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            sampler=sampler,
            collate_fn=pytorch_dataset.collate_fn,
        )
        return sampler, data_loader
    else:
        data_loader = data.DataLoader(
            pytorch_dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=should_shuffle,
            collate_fn=pytorch_dataset.collate_fn,
        )

        return None, data_loader


def getMultipleEpochs_ofBatches(
    pytorch_dataset, batch_size, should_shuffle, world_size, device
):
    sampler, data_loader = _create_dataLoader(
        pytorch_dataset, batch_size, should_shuffle, world_size, device
    )
    current_epoch = 0

    while True:
        if is_distributedSetup(world_size):
            sampler.set_epoch(current_epoch)

        for x in data_loader:
            yield x

        if is_distributedSetup(world_size):
            current_epoch += 1


def getSingleEpoch_OfBatches(pytorch_dataset, batch_size, world_size, device):
    _, data_loader = _create_dataLoader(
        pytorch_dataset, batch_size, False, world_size, device
    )

    for x in data_loader:
        yield x
