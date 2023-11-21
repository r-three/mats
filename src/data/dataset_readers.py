import os
import logging
import copy
import random
import json

from promptsource.templates import DatasetTemplates, Template
from datasets import load_dataset
from src.data.DatasetConfig import DatasetConfig

SPLIT_MAPPING = {"train": "train", "validation": "train", "test": "validation"}
P3_VALIDATION_SET_SIZE = 1000
NULL_ANSWER_CHOICE = "NULL_ANSWER"


class P3_DatasetReader(object):
    """
    DatasetReader objects reads dataset and has all attributes specific to dataset
    """

    def __init__(self, dataset_stash, template_stash):
        self.dataset_stash = dataset_stash
        self.template_stash = template_stash

        self.all_templates = self._get_datasetTemplates(None, None)

        self.cached_origData = {}
        self.cached_datasets = {}

    def _get_origData(self, split):
        """

        Args:
            split:

        Returns:

        """
        real_split = SPLIT_MAPPING[split]

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=real_split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            if split == "train":
                orig_data = orig_data[:-P3_VALIDATION_SET_SIZE]
            elif split == "validation":
                orig_data = orig_data[-P3_VALIDATION_SET_SIZE:]

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        """
        Args:
            templateNames_toIgnore:
            metrics_toUse: specify the metric to use so that we only include templates which
                           match the metric we want to use

        Returns:

        """
        all_templates = []

        # Get original templates from promptsource
        for template in DatasetTemplates(*self.template_stash).templates.values():
            # Filter out templates that
            # 1) are not designed for original task
            # 2) have different metrics than we want to use
            # 3) are ones that we want to ignore based on the name
            if template.metadata.original_task:
                should_ignoreTemplate = False

                for metric in template.metadata.metrics:
                    if metric not in metrics_toUse:
                        should_ignoreTemplate = True

                for template_name in templateNames_toIgnore:
                    if template.name == template_name:
                        should_ignoreTemplate = True

                if not should_ignoreTemplate:
                    all_templates.append(template)

        return all_templates

    def _applyTemplate_toData(
        self, orig_data, num_templates, template_idx, use_answerChoices
    ):
        """
        Args:
            orig_data:
            num_templates:
            template_idx:
            use_answerChoices:

        Returns:

        """
        dataset = []

        for datapoint_idx, datapoint in enumerate(orig_data):
            # Use fixed template across entire dataset
            if template_idx >= 0:
                templateIdx_forDatapoint = template_idx

            # Use all templates across entire dataset, where different datapoints can get
            # different templates. However, a datapoint is always matched with the same template
            elif template_idx == -1:
                templateIdx_forDatapoint = datapoint_idx % num_templates

            else:
                raise ValueError(f"Invalid template index {templateIdx_forDatapoint}")

            template = self.all_templates[templateIdx_forDatapoint]
            new_datapoint = copy.deepcopy(datapoint)

            # Whether to use answer_choices or target
            if use_answerChoices:
                answer_choices = template.get_answer_choices_list(datapoint)
                if answer_choices is not None:
                    new_datapoint["answer_choices"] = answer_choices

            # We apply the template to datapoint instead of new_datapoint since the answer_choices
            # are added in the template function, and so applying the template to new_datapoint
            # will cause an error with the answer_choices key
            input_txt, target_txt = template.apply(datapoint)
            new_datapoint["input"] = input_txt

            # For non-evaluation or tasks where they are no answer_choices, we just add the target (
            # the correct answer_choice)
            if not use_answerChoices or "answer_choices" not in new_datapoint:
                new_datapoint["target"] = target_txt
            dataset.append(new_datapoint)

        return dataset

    def get_dataset(self, split, template_idx, use_answerChoices, num_samples):
        """
        Create dataset that includes the template

        Args:
            split:
            template_idx:
                if >=0, then we use the fixed template_idx across entire dataset
                if ==-1, then we use all template across entire the dataset, where different
                         datapoints can have different templates. A datapoint will always be
                         mapped to the same template though
                if ==-2, then we take the cross product of all templates and all datapoints.
            use_answerChoices: whether the split is for evaluation (where it will have answer_choices)
                            or for training (where it will only have the target)
            num_samples: how many samples to be selected for the dataset. samples are based on
                        datapoints in the original dataset and not datapoints with a template applied
                if == -1, then we use the entire dataset
        Returns:
            dataset:
        """
        if (
            split,
            template_idx,
            use_answerChoices,
            num_samples,
        ) not in self.cached_datasets:
            orig_data = self._get_origData(split)

            if num_samples != -1:
                orig_data = orig_data[:num_samples]

            logging.info(
                f"Loaded {split} which contains {len(orig_data)} original datapoints"
            )

            num_templates = self.get_numTemplates()

            # template_idx -2 means we do a cross product of each datapoint with each template
            if template_idx == -2:
                dataset = []
                templateIdx_toData = {}
                for iterate_templateIdx in range(num_templates):
                    templateIdx_toData[
                        iterate_templateIdx
                    ] = self._applyTemplate_toData(
                        orig_data, num_templates, iterate_templateIdx, use_answerChoices
                    )
                num_datapoints = len(templateIdx_toData[0])
                for datapoint_idx in range(num_datapoints):
                    for template_idx in range(len(templateIdx_toData)):
                        dataset.append(templateIdx_toData[template_idx][datapoint_idx])

            # otherwise apply template to dataset
            else:
                dataset = self._applyTemplate_toData(
                    orig_data, num_templates, template_idx, use_answerChoices
                )
            logging.info(
                f"Loaded {split} which contains {len(dataset)} datapoints with templates"
            )

            self.cached_datasets[
                (split, template_idx, use_answerChoices, num_samples)
            ] = dataset

        return self.cached_datasets[
            (split, template_idx, use_answerChoices, num_samples)
        ]

    def get_numTemplates(self):
        return len(self.all_templates)

    def get_datasetMetrics(self):
        return self.all_templates[0].metadata.metrics


class P3_RTEReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(
            dataset_stash=("super_glue", "rte"), template_stash=("super_glue", "rte")
        )

    def _get_origData(self, split):
        return super()._get_origData(split)

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_MNLIReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(
            dataset_stash=("glue", "mnli"), template_stash=("glue", "mnli")
        )

    def _get_origData(self, split):
        print(split)
        return super()._get_origData(split)

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_QNLIReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(
            dataset_stash=("glue", "qnli"), template_stash=("glue", "qnli")
        )

    def _get_origData(self, split):
        return super()._get_origData(split)

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_QQPReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(dataset_stash=("glue", "qqp"), template_stash=("glue", "qqp"))

    def _get_origData(self, split):
        return super()._get_origData(split)

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_HSwagReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(dataset_stash=("hellaswag",), template_stash=("hellaswag",))

    def _get_origData(self, split):
        real_split = SPLIT_MAPPING[split]

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=real_split,
                cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            if split == "train":
                orig_data = orig_data[:-P3_VALIDATION_SET_SIZE]
            elif split == "validation":
                orig_data = orig_data[-P3_VALIDATION_SET_SIZE:]

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        all_templates = super()._get_datasetTemplates(
            ["Randomized prompts template"], ["Accuracy"]
        )

        # Add each template from the several templates in the randomized prompt individually
        listOf_randomJinjas = [
            (
                "randomized prompt 1",
                "Can you pick the correct ending for the sentence: {{ctx}}|||{{answer_choices [label | int()]}}",
            ),
            (
                "randomized prompt 2",
                "The task is to generate the ending for the sentence: {{ctx}}|||{{answer_choices [label | int()]}}",
            ),
            (
                "randomized prompt 3",
                "How does this sentence end? {{ctx}}|||{{answer_choices [label | int()]}}",
            ),
            (
                "randomized prompt 4",
                "From the list of endings described below, what ending makes the most sense for the sentence {{ctx}}|||{{answer_choices [label | int()]}}",
            ),
        ]

        for name, jinja in listOf_randomJinjas:
            all_templates.append(
                Template(
                    name=name,
                    jinja=jinja,
                    reference="",
                    answer_choices='{{endings | join("|||")}}',
                )
            )

        return all_templates


class P3_WiCReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(
            dataset_stash=("super_glue", "wic"), template_stash=("super_glue", "wic")
        )

    def _get_origData(self, split):
        return super()._get_origData(split)

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_WinograndeReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(
            dataset_stash=("winogrande", "winogrande_xl"),
            template_stash=("winogrande", "winogrande_xl"),
        )

    def _get_origData(self, split):
        real_split = SPLIT_MAPPING[split]

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=real_split,
                cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["answer"]) - 1
                orig_data.append(example)

            if split == "train":
                orig_data = orig_data[:-P3_VALIDATION_SET_SIZE]
            elif split == "validation":
                orig_data = orig_data[-P3_VALIDATION_SET_SIZE:]

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_CBReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(
            dataset_stash=("super_glue", "cb"), template_stash=("super_glue", "cb")
        )

    def _get_origData(self, split):
        real_split = SPLIT_MAPPING[split]

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=real_split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            if split == "train":
                orig_data = orig_data[:-100]
            elif split == "validation":
                orig_data = orig_data[-100:]

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_BoolQReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(
            dataset_stash=("super_glue", "boolq"),
            template_stash=("super_glue", "boolq"),
        )

    def _get_origData(self, split):
        return super()._get_origData(split)

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_COPAReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(
            dataset_stash=("super_glue", "copa"), template_stash=("super_glue", "copa")
        )

    def _get_origData(self, split):
        real_split = SPLIT_MAPPING[split]

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=real_split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            if split == "train":
                orig_data = orig_data[:-100]
            elif split == "validation":
                orig_data = orig_data[-100:]

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates(
            [
                "…which may be caused by",
                "…What could happen next, C1 or C2?",
                "…As a result, C1 or C2?",
                "…why? C1 or C2",
            ],
            ["Accuracy"],
        )


class P3_MultiRCReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(
            dataset_stash=("super_glue", "multirc"),
            template_stash=("super_glue", "multirc"),
        )

    def _get_origData(self, split):
        return super()._get_origData(split)

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_ReCORDReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(
            dataset_stash=("super_glue", "record"),
            template_stash=("super_glue", "record"),
        )

    def _get_origData(self, split):
        real_split = SPLIT_MAPPING[split]

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=real_split,
                cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["id"] = str(idx)
                example["text"] = example["answers"]
                example["answer_start"] = [0] * len(example["answers"])
                orig_data.append(example)

            if split == "train":
                orig_data = orig_data[:-P3_VALIDATION_SET_SIZE]
            elif split == "validation":
                orig_data = orig_data[-P3_VALIDATION_SET_SIZE:]

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Squad"])


class P3_WiCReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(
            dataset_stash=("super_glue", "wic"),
            template_stash=("super_glue", "wic"),
        )

    def _get_origData(self, split):
        return super()._get_origData(split)

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_WSCReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(
            dataset_stash=("super_glue", "wsc.fixed"),
            template_stash=("super_glue", "wsc.fixed"),
        )

    def _get_origData(self, split):
        real_split = SPLIT_MAPPING[split]

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=real_split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            if split == "train":
                orig_data = orig_data[:-100]
            elif split == "validation":
                orig_data = orig_data[-100:]

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_CoLAReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(
            dataset_stash=("glue", "cola"), template_stash=("glue", "cola")
        )

    def _get_origData(self, split):
        return super()._get_origData(split)

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_STSBReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(
            dataset_stash=("glue", "stsb"), template_stash=("glue", "stsb")
        )

    def _get_origData(self, split):
        return super()._get_origData(split)

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_MRPCReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(
            dataset_stash=("glue", "mrpc"), template_stash=("glue", "mrpc")
        )

    def _get_origData(self, split):
        return super()._get_origData(split)

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_SST2Reader(P3_DatasetReader):
    def __init__(self):
        super().__init__(
            dataset_stash=("glue", "sst2"), template_stash=("glue", "sst2")
        )

    def _get_origData(self, split):
        return super()._get_origData(split)

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_WNLIReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(
            dataset_stash=("glue", "wnli"), template_stash=("glue", "wnli")
        )

    def _get_origData(self, split):
        real_split = SPLIT_MAPPING[split]

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=real_split,
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            if split == "train":
                orig_data = orig_data[:-200]
            elif split == "validation":
                orig_data = orig_data[-200:]

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_StoryClozeReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(
            dataset_stash=("story_cloze", "2016"),
            template_stash=("story_cloze", "2016"),
        )

    def _get_origData(self, split):
        # We use the test set of StoryCloze for validation and the validation set of StoryCloze
        # for train - following GPT3
        split_mapping = {
            "train": "validation",
            "validation": "validation",
            "test": "test",
        }
        real_split = split_mapping[split]

        if split not in self.cached_origData:
            # Do not use default method for loading dataset since the story_cloze dataset must be
            # downloaded manually and then we have to set data_dir to point to it.
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=real_split,
                data_dir=os.path.join(
                    os.environ["HUGGINGFACE_HUB_CACHE"], "datasets", "story_cloze"
                ),
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["answer_right_ending"]) - 1
                orig_data.append(example)

            if split == "train":
                orig_data = orig_data[:-P3_VALIDATION_SET_SIZE]
            elif split == "validation":
                orig_data = orig_data[-P3_VALIDATION_SET_SIZE:]

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_ANLIR1Reader(P3_DatasetReader):
    def __init__(self):
        super().__init__(dataset_stash=("anli",), template_stash=("anli",))

    def _get_origData(self, split):
        split_mapping = {
            "train": "train",
            "validation": "train",
            "test": "dev",
        }
        real_split = split_mapping[split]

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=f"{real_split}_r1",
                cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            if split == "train":
                orig_data = orig_data[:-P3_VALIDATION_SET_SIZE]
            elif split == "validation":
                orig_data = orig_data[-P3_VALIDATION_SET_SIZE:]

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_ANLIR2Reader(P3_DatasetReader):
    def __init__(self):
        super().__init__(dataset_stash=("anli",), template_stash=("anli",))

    def _get_origData(self, split):
        split_mapping = {
            "train": "train",
            "validation": "train",
            "test": "dev",
        }
        real_split = split_mapping[split]

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=f"{real_split}_r2",
                cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            if split == "train":
                orig_data = orig_data[:-P3_VALIDATION_SET_SIZE]
            elif split == "validation":
                orig_data = orig_data[-P3_VALIDATION_SET_SIZE:]

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_ANLIR3Reader(P3_DatasetReader):
    def __init__(self):
        super().__init__(dataset_stash=("anli",), template_stash=("anli",))

    def _get_origData(self, split):
        split_mapping = {
            "train": "train",
            "validation": "train",
            "test": "dev",
        }
        real_split = split_mapping[split]

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=f"{real_split}_r3",
                cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            if split == "train":
                orig_data = orig_data[:-P3_VALIDATION_SET_SIZE]
            elif split == "validation":
                orig_data = orig_data[-P3_VALIDATION_SET_SIZE:]

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_CosmosQAReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(dataset_stash=("cosmos_qa",), template_stash=("cosmos_qa",))

    def _get_origData(self, split):
        real_split = SPLIT_MAPPING[split]

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=real_split,
                cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            if split == "train":
                orig_data = orig_data[:-P3_VALIDATION_SET_SIZE]
            elif split == "validation":
                orig_data = orig_data[-P3_VALIDATION_SET_SIZE:]

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_SocialIQAReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(
            dataset_stash=("social_i_qa",), template_stash=("social_i_qa",)
        )

    def _get_origData(self, split):
        real_split = SPLIT_MAPPING[split]

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=real_split,
                cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"]) - 1
                orig_data.append(example)

            if split == "train":
                orig_data = orig_data[:-P3_VALIDATION_SET_SIZE]
            elif split == "validation":
                orig_data = orig_data[-P3_VALIDATION_SET_SIZE:]

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates(
            ["Check if a random answer is valid or not"], ["Accuracy"]
        )


class P3_PAWSReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(
            dataset_stash=("paws", "labeled_final"),
            template_stash=("paws", "labeled_final"),
        )

    def _get_origData(self, split):
        real_split = SPLIT_MAPPING[split]

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=real_split,
                cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = example["label"]
                orig_data.append(example)

            if split == "train":
                orig_data = orig_data[:-P3_VALIDATION_SET_SIZE]
            elif split == "validation":
                orig_data = orig_data[-P3_VALIDATION_SET_SIZE:]

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_QuAILReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(dataset_stash=("quail",), template_stash=("quail",))

    def _get_origData(self, split):
        real_split = SPLIT_MAPPING[split]

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=real_split,
                cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = example["correct_answer_id"]
                orig_data.append(example)
            if split == "train":
                orig_data = orig_data[:-P3_VALIDATION_SET_SIZE]
            elif split == "validation":
                orig_data = orig_data[-P3_VALIDATION_SET_SIZE:]

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_WikiQAReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(dataset_stash=("wiki_qa",), template_stash=("wiki_qa",))

    def _get_origData(self, split):
        real_split = SPLIT_MAPPING[split]

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=real_split,
                cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = int(example["label"])
                orig_data.append(example)

            if split == "train":
                orig_data = orig_data[:-P3_VALIDATION_SET_SIZE]
            elif split == "validation":
                orig_data = orig_data[-P3_VALIDATION_SET_SIZE:]

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_QuaRTzReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(dataset_stash=("quartz",), template_stash=("quartz",))

        self.string_toLabelIdx = {"A": 0, "B": 1}

    def _get_origData(self, split):
        real_split = SPLIT_MAPPING[split]

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=real_split,
                cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = self.string_toLabelIdx[example["answerKey"]]
                orig_data.append(example)

            validation_set_size = 200
            if split == "train":
                orig_data = orig_data[:-validation_set_size]
            elif split == "validation":
                orig_data = orig_data[-validation_set_size:]

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_QASCReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(dataset_stash=("qasc",), template_stash=("qasc",))

        self.string_toLabelIdx = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
            "G": 6,
            "H": 7,
        }

    def _get_origData(self, split):
        if split == "test":
            split = "validation"

        if split not in self.cached_origData:
            real_split = SPLIT_MAPPING[split]

            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=real_split,
                cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["lbl"] = self.string_toLabelIdx[example["answerKey"]]
                orig_data.append(example)

            validation_set_size = 500
            if split == "train":
                orig_data = orig_data[:-validation_set_size]
            elif split == "validation":
                orig_data = orig_data[-validation_set_size:]

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Accuracy"])


class P3_ROPESReader(P3_DatasetReader):
    def __init__(self):
        super().__init__(dataset_stash=("ropes",), template_stash=("ropes",))

    def _get_origData(self, split):
        real_split = SPLIT_MAPPING[split]

        if split not in self.cached_origData:
            huggingFace_data = load_dataset(
                *self.dataset_stash,
                split=real_split,
                cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
            )

            orig_data = []
            for idx, example in enumerate(huggingFace_data):
                example["idx"] = idx
                example["answers"]["answer_start"] = [0]
                orig_data.append(example)

            if split == "train":
                orig_data = orig_data[:-P3_VALIDATION_SET_SIZE]
            elif split == "validation":
                orig_data = orig_data[-P3_VALIDATION_SET_SIZE:]

            self.cached_origData[split] = orig_data

        return self.cached_origData[split]

    def _get_datasetTemplates(self, templateNames_toIgnore, metrics_toUse):
        return super()._get_datasetTemplates([], ["Squad"])


P3_DATASETS = {
    "rte": P3_RTEReader,
    "h-swag": P3_HSwagReader,
    "copa": P3_COPAReader,
    "wic": P3_WiCReader,
    "winogrande": P3_WinograndeReader,
    "cb": P3_CBReader,
    "story_cloze": P3_StoryClozeReader,
    "anli-r1": P3_ANLIR1Reader,
    "anli-r2": P3_ANLIR2Reader,
    "anli-r3": P3_ANLIR3Reader,
    "wsc": P3_WSCReader,
    "cosmos_qa": P3_CosmosQAReader,
    "social_iqa": P3_SocialIQAReader,
    "paws": P3_PAWSReader,
    "quail": P3_QuAILReader,
    "wiki_qa": P3_WikiQAReader,
    "quartz": P3_QuaRTzReader,
    "qasc": P3_QASCReader,
    "ropes": P3_ROPESReader,
    "qnli": P3_QNLIReader,
    "qqp": P3_QQPReader,
    "mnli": P3_MNLIReader,
    "cola": P3_CoLAReader,
    "stsb": P3_STSBReader,
    "wnli": P3_WNLIReader,
    "sst2": P3_SST2Reader,
    "mrpc": P3_MRPCReader,
    "boolq": P3_BoolQReader,
    "multirc": P3_MultiRCReader,
    "record": P3_ReCORDReader,
}

def get_singleDatasetReader(dataset_config, cached_singleDatasetReaders):
    """Get a dataset reader for asingle dataset

    Args:
        dataset_config (_type_): _description_
        cached_singleDatasetReaders (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        dataset_reader: _description_
        cached_singleDatasetReaders:
    """
    if cached_singleDatasetReaders is None:
        cached_singleDatasetReaders = {}
    if (
        dataset_config.instruction_format,
        dataset_config.dataset,
    ) not in cached_singleDatasetReaders:
        if dataset_config.instruction_format == "p3":
            dataset_reader = P3_DATASETS[dataset_config.dataset]()
        else:
            raise ValueError(
                f"invalid instruction format {dataset_config.instruction_format}"
            )
        cached_singleDatasetReaders[
            (dataset_config.instruction_format, dataset_config.dataset)
        ] = dataset_reader

    return (
        cached_singleDatasetReaders[
            (dataset_config.instruction_format, dataset_config.dataset)
        ],
        cached_singleDatasetReaders,
    )


class DatasetMixtureReader(object):
    def __init__(self, dataset_mixture, dataset_config, cached_singleDatasetReaders):
        self.dataset_mixture = dataset_mixture
        self.dataset_config = dataset_config
        self.cached_singleDatasetReaders = cached_singleDatasetReaders

        # Store the individual dataset readers so we don't have to rely on the cache to store it,
        # even if the cache also has the dataset readers
        self.mixture_ofDatasetReaders = []

        # Add dataset_readers to the cache of single dataset readers
        for dataset in getDatasets_inMixture(self.dataset_mixture):
            new_updateDict = self.dataset_config.get_key_values()
            new_updateDict.update({"dataset": dataset})
            newDataset_config = DatasetConfig(
                None,
                update_dict=new_updateDict,
            )

            dataset_reader, self.cached_singleDatasetReaders = get_singleDatasetReader(
                newDataset_config, self.cached_singleDatasetReaders
            )
            self.mixture_ofDatasetReaders.append(dataset_reader)

    def get_cacheSingleDatasetReaders(self):
        return self.cached_singleDatasetReaders

    def get_dataset(self, split, template_idx, use_answerChoices, num_samples):
        """
        Create dataset that includes the template

        Args:
            split:
            template_idx:
                if >=0, then we use the fixed template_idx across entire dataset
                if ==-1, then we use all template across entire the dataset, where different
                         datapoints can have different templates. A datapoint will always be
                         mapped to the same template though
                if ==-2, then we take the cross product of all templates and all datapoints.
            use_answerChoices:

        Returns:
            dataset:
        """

        dataset = []

        for dataset_name in getDatasets_inMixture(self.dataset_mixture):
            dataset_reader = self.cached_singleDatasetReaders[
                (self.dataset_config.instruction_format, dataset_name)
            ]

            # If we only use 1 template, then get the dataset for each template
            if template_idx >= -1:
                single_dataset = dataset_reader.get_dataset(
                    split, template_idx, use_answerChoices, num_samples=-1
                )
                dataset.extend(
                    single_dataset[
                        : self.dataset_config.max_examples_per_dataset_for_mixture
                    ]
                )

            else:
                assert template_idx == -2

                num_templates = dataset_reader.get_numTemplates()

                maximumDatapoints_perDatasetAndTemplate = (
                    self.dataset_config.max_examples_per_dataset_for_mixture
                    // num_templates
                )
                # To ensure each dataset gets the same number of datapoints, if the maximum
                # number of datapoints is not divisble by the number of templates, we compute
                # the remainder and add back in this many datapoints
                remainderDatasets_withExtraDatapoint = (
                    self.dataset_config.max_examples_per_dataset_for_mixture
                    % num_templates
                )

                # Add the same number of datapoints per template
                for iterated_templateIdx in range(num_templates):
                    single_dataset = dataset_reader.get_dataset(
                        split, iterated_templateIdx, use_answerChoices, num_samples=-1
                    )
                    # Fix the seed so that the examples chosen in the mixture of dataset is
                    # deterministic
                    random.seed(0)
                    random.shuffle(single_dataset)
                    dataset.extend(
                        single_dataset[:maximumDatapoints_perDatasetAndTemplate]
                    )

                    if iterated_templateIdx < remainderDatasets_withExtraDatapoint:
                        if (
                            len(single_dataset)
                            >= maximumDatapoints_perDatasetAndTemplate
                        ):
                            dataset.append(
                                single_dataset[maximumDatapoints_perDatasetAndTemplate]
                            )
        return dataset

    def get_numTemplates(self):
        raise ValueError("Cannot get number of templates for mixture of datasets")

    def get_datasetMetrics(self):
        raise ValueError("Cannot get metrics for mixture of datasets")


DATASET_MIXTURES = {
    "p3_eight_qa": [
        "cosmos_qa",
        "social_iqa",
        "paws",
        "quail",
        "wiki_qa",
        "quartz",
        "qasc",
        "ropes",
    ],
    "p3_ties": [
        "paws",
        "qasc",
        "quartz",
        "story_cloze",
        "wiki_qa",
        "winogrande",
        "wsc",
    ],
    "p3_mnli_rte": ["mnli", "rte"],
    "p3_qqp_rte": ["qqp", "rte"],
    "p3_qnli_rte": ["qnli", "rte"],
}


def get_allDatasets():
    setOf_datasets = set()
    for _, listOf_datasets in DATASET_MIXTURES.items():
        setOf_datasets.update(set(listOf_datasets))
    return list(setOf_datasets)


def getDatasets_inMixture(dataset_mixture):
    """
    dataset_mixture can also just be a single dataset

    Args:
        dataset_mixture:

    Returns:

    """
    # If dataset_mixture is a list, then check that each dataset in list is valid.
    if isinstance(dataset_mixture, list):
        for dataset in dataset_mixture:
            assert dataset in P3_DATASETS.keys()
        return dataset_mixture
    # If dataset_mixture is a string, then we look up the dataset mixture.
    elif dataset_mixture in DATASET_MIXTURES.keys():
        return DATASET_MIXTURES[dataset_mixture]
    # dataset_mixture might be just one dataset
    else:
        assert (
            dataset_mixture in P3_DATASETS.keys()
        )
        return [dataset_mixture]


def get_datasetMixtureReader(
    dataset_mixture, dataset_config, cached_singleDatasetReaders
):
    """

    Args:
        dataset_mixture:
        dataset_config:
        cached_singleDatasetReaders:

    Returns:
        datasetMixture_reader:
        cached_singleDatasetReaders:
    """

    datasetMixture_reader = DatasetMixtureReader(
        dataset_mixture, dataset_config, cached_singleDatasetReaders
    )
    return datasetMixture_reader, datasetMixture_reader.get_cacheSingleDatasetReaders()


def get_datasetReader(dataset_mixture, dataset_config, cached_singleDatasetReaders):
    """Get a dataset readers that either reads a single dataset or a mixture of datasets

    Args:
        dataset_mixture (_type_): _description_
        dataset_config (_type_): _description_
        cached_singleDatasetReaders (_type_): _description_

    Returns:
        _type_: _description_
    """
    if dataset_mixture is None:
        return get_singleDatasetReader(dataset_config, cached_singleDatasetReaders)
    else:
        return get_datasetMixtureReader(
            dataset_mixture, dataset_config, cached_singleDatasetReaders
        )
