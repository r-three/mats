import torch
import copy
from torch.utils import data

from src.data.dataset_readers import NULL_ANSWER_CHOICE
from src.utils.utils import flatten_list


class PytorchDataset(data.Dataset):
    def __init__(
        self,
        dataset,
        dataset_config,
        tokenize_fn,
        device,
    ):
        self.dataset = dataset
        self.dataset_config = dataset_config
        self.tokenize_fn = tokenize_fn
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, get_idx):
        return copy.deepcopy(self.dataset[get_idx])

    def collate_fn(self, batch_ofDatapoints):
        """
        Convert a batch of datapoints into a datapoint that is batched.  This is meant to
        override the default collate function in pytorch.

        Args:
            batch_ofDatapoints:

        Returns:

        """
        datapoint_batched = {}

        for datapoint in batch_ofDatapoints:
            for k, v in datapoint.items():
                if k in datapoint_batched:
                    datapoint_batched[k].append(v)
                else:
                    datapoint_batched[k] = [v]

        datapoint_batched = self.tokenize_fn(datapoint_batched, self.device)

        # Add a non null anwer choices mask to handle answer_choices that are null
        # and should be padded out.
        if "answer_choices" in datapoint_batched:
            nonNull_answerChoices = []
            for answer_choice in flatten_list(datapoint_batched["answer_choices"]):
                if answer_choice == NULL_ANSWER_CHOICE:
                    nonNull_answerChoices.append(0)
                else:
                    nonNull_answerChoices.append(1)

            datapoint_batched["non_null_answer_choices"] = torch.tensor(
                nonNull_answerChoices
            ).to(self.device)

        # Convert lbl to tensor
        if "lbl" in datapoint_batched:
            datapoint_batched["lbl"] = torch.tensor(datapoint_batched["lbl"]).to(
                self.device
            )

        return datapoint_batched
