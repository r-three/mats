from src.utils.utils import format_modelName

from src.data.dataset_readers import getDatasets_inMixture


MODEL_CHECKPOINTS = {
    "google-t5-large-lm-adapt": {
        "p3": {
            "full_model": {
                "cosmos_qa": "exp_out/p3/cosmos_qa/google-t5-large-lm-adapt/2023-04-30-10-16-25/checkpoints/checkpoint_399.pt",
                "paws": "exp_out/p3/paws/google-t5-large-lm-adapt/2023-04-30-00-43-21/checkpoints/checkpoint_1299.pt",
                "qasc": "exp_out/p3/qasc/google-t5-large-lm-adapt/2023-04-29-21-15-10/checkpoints/checkpoint_199.pt",
                "quail": "exp_out/p3/quail/google-t5-large-lm-adapt/2023-04-29-21-11-24/checkpoints/checkpoint_999.pt",
                "quartz": "exp_out/p3/quartz/google-t5-large-lm-adapt/2023-04-29-21-13-32/checkpoints/checkpoint_399.pt",
                "ropes": "exp_out/p3/ropes/google-t5-large-lm-adapt/2023-04-29-21-11-51/checkpoints/checkpoint_499.pt",
                "social_iqa": "exp_out/p3/social_iqa/google-t5-large-lm-adapt/2023-04-29-21-13-12/checkpoints/checkpoint_799.pt",
                "wiki_qa": "exp_out/p3/wiki_qa/google-t5-large-lm-adapt/2023-04-29-21-15-46/checkpoints/checkpoint_499.pt",
                "mnli": "exp_out/p3/mnli/google-t5-large-lm-adapt/full_model/2023-09-24-01-05-51/checkpoints/checkpoint_599.pt",
                "rte": "exp_out/p3/rte/google-t5-large-lm-adapt/full_model/2023-09-23-23-19-06/checkpoints/checkpoint_399.pt",
                "qqp": "exp_out/p3/qqp/google-t5-large-lm-adapt/full_model/2023-09-23-23-18-55/checkpoints/checkpoint_999.pt",
                "qnli": "exp_out/p3/qnli/google-t5-large-lm-adapt/full_model/2023-09-24-00-13-28/checkpoints/checkpoint_899.pt",
            },
            "ia3": {
                "cosmos_qa": "exp_out/p3/cosmos_qa/google-t5-large-lm-adapt/ia3/2023-05-01-10-25-08/checkpoints/checkpoint_1299.pt",
                "paws": "exp_out/p3/paws/google-t5-large-lm-adapt/ia3/2023-04-30-19-54-04/checkpoints/checkpoint_2699.pt",
                "qasc": "exp_out/p3/qasc/google-t5-large-lm-adapt/ia3/2023-05-01-10-23-07/checkpoints/checkpoint_499.pt",
                "quail": "exp_out/p3/quail/google-t5-large-lm-adapt/ia3/2023-04-30-18-25-27/checkpoints/checkpoint_1399.pt",
                "quartz": "exp_out/p3/quartz/google-t5-large-lm-adapt/ia3/2023-04-30-19-53-11/checkpoints/checkpoint_1799.pt",
                "ropes": "exp_out/p3/ropes/google-t5-large-lm-adapt/ia3/2023-04-30-18-26-14/checkpoints/checkpoint_799.pt",
                "social_iqa": "exp_out/p3/social_iqa/google-t5-large-lm-adapt/ia3/2023-04-30-18-26-30/checkpoints/checkpoint_1099.pt",
                "wiki_qa": "exp_out/p3/wiki_qa/google-t5-large-lm-adapt/ia3/2023-04-30-23-57-20/checkpoints/checkpoint_99.pt",
                "mnli": "exp_out/p3/mnli/google-t5-large-lm-adapt/ia3/2023-09-24-08-23-23/checkpoints/checkpoint_2099.pt",
                "qnli": "exp_out/p3/qnli/google-t5-large-lm-adapt/ia3/2023-09-24-08-24-36/checkpoints/checkpoint_1199.pt",
                "qqp": "exp_out/p3/qqp/google-t5-large-lm-adapt/ia3/2023-09-24-04-50-08/checkpoints/checkpoint_1399.pt",
                "rte": "exp_out/p3/rte/google-t5-large-lm-adapt/ia3/2023-09-24-04-49-45/checkpoints/checkpoint_799.pt",
            },
            "full_model_ties": {
                "winogrande": "exp_out/p3/winogrande/google-t5-large-lm-adapt/full_model/2023-10-06-08-45-28/checkpoints/checkpoint_1399.pt",
                "wsc": "exp_out/p3/wsc/google-t5-large-lm-adapt/full_model/2023-10-06-09-50-48/checkpoints/checkpoint_99.pt",
                "story_cloze": "exp_out/p3/story_cloze/google-t5-large-lm-adapt/full_model/2023-10-05-21-20-36/checkpoints/checkpoint_299.pt",
                "qasc": "exp_out/p3/qasc/google-t5-large-lm-adapt/2023-04-29-21-15-10/checkpoints/checkpoint_199.pt",
                "quartz": "exp_out/p3/quartz/google-t5-large-lm-adapt/2023-04-29-21-13-32/checkpoints/checkpoint_399.pt",
                "paws": "exp_out/p3/paws/google-t5-large-lm-adapt/2023-04-30-00-43-21/checkpoints/checkpoint_1299.pt",
                "wiki_qa": "exp_out/p3/wiki_qa/google-t5-large-lm-adapt/2023-04-29-21-15-46/checkpoints/checkpoint_499.pt",
            },
            "ia3_ties": {
                "winogrande": "exp_out/p3/winogrande/google-t5-large-lm-adapt/ia3/2023-10-06-12-55-09/checkpoints/checkpoint_2099.pt",
                "wsc": "exp_out/p3/wsc/google-t5-large-lm-adapt/ia3/2023-10-06-12-55-13/checkpoints/checkpoint_99.pt",
                "story_cloze": "exp_out/p3/story_cloze/google-t5-large-lm-adapt/ia3/2023-10-06-12-59-38/checkpoints/checkpoint_1099.pt",
                "quartz": "exp_out/p3/quartz/google-t5-large-lm-adapt/ia3/2023-04-30-19-53-11/checkpoints/checkpoint_1799.pt",
                "qasc": "exp_out/p3/qasc/google-t5-large-lm-adapt/ia3/2023-05-01-10-23-07/checkpoints/checkpoint_499.pt",
                "paws": "exp_out/p3/paws/google-t5-large-lm-adapt/ia3/2023-04-30-19-54-04/checkpoints/checkpoint_2699.pt",
                "wiki_qa": "exp_out/p3/wiki_qa/google-t5-large-lm-adapt/ia3/2023-04-30-23-57-20/checkpoints/checkpoint_99.pt",
            },
        }
    }
}


def get_modelCheckpointFilepath(
    pretrained_model, checkpoint_descriptor, instruction_format, dataset
):
    """

    Args:
        pretrained_model:
        checkpoint_descriptor:
        dataset:

    Returns:

    """
    pretrained_model = format_modelName(pretrained_model)
    return MODEL_CHECKPOINTS[pretrained_model][instruction_format][
        checkpoint_descriptor
    ][dataset]


def get_datasetMixtureCheckpointFilepaths(
    pretrained_model, checkpoint_descriptor, instruction_format, dataset_mixture
):
    """

    Args:
        pretrained_model (str):
        checkpoint_descriptor (str):
        dataset_mixture (list or str):

    Returns:
        checkpoint_fps (list):
    """

    checkpoint_fps = []
    for dataset in getDatasets_inMixture(dataset_mixture):
        checkpoint_fps.append(
            get_modelCheckpointFilepath(
                pretrained_model, checkpoint_descriptor, instruction_format, dataset
            )
        )

    return checkpoint_fps


def getDataset_fromCheckpointFilepath(filepath):
    dataset = "_".join(filepath.split("/")[1:3])
    return dataset
