import torch
import numpy as np


def normalize_metadata(stored_metadata, count):
    normalized_metadata = {}
    for parameter_name, parameter in stored_metadata.items():
        normalized_metadata[parameter_name] = parameter / count
    return normalized_metadata


def detach_metadata(stored_metadata):
    detached_metadata = {}
    for parameter_name, parameter in stored_metadata.items():
        detached_metadata[parameter_name] = parameter.detach().contiguous().cpu()
    return detached_metadata


def sample_label(model, batch, evaluation_config, metrics):
    with torch.no_grad():
        if "Accuracy" in metrics:
            (_, score_ofChoices, _, _) = model.predict_mulChoice(
                batch, evaluation_config.length_normalization
            )
            # batch_size must be 1
            assert len(score_ofChoices) == 1
            probs = np.exp(score_ofChoices[0])
            normalized_probs = probs / np.sum(probs)
            # Sample lbl from predicted distribution
            sampled_lbl = np.random.choice(len(normalized_probs), p=normalized_probs)

            assert (
                len(batch["answer_choices"][0]) == batch["answer_choices_ids"].shape[0]
            )

            # Get the answer_ids from the corresponding sample lbl
            max_targetLen = batch["answer_choices_ids"].shape[1]
            target_idx = (
                torch.tensor([sampled_lbl])
                .to(batch["answer_choices_ids"].device)[:, None]
                .repeat((1, max_targetLen))
            )

            target_ids = torch.gather(batch["answer_choices_ids"], 0, target_idx)
            target_mask = torch.gather(batch["answer_choices_mask"], 0, target_idx)
            batch["target_ids"] = target_ids
            batch["target_mask"] = target_mask

        if "Squad" in metrics or "F1" in metrics:
            sampled_ids, _ = model.generate(
                batch, evaluation_config.max_gen_len, sample_tokens=True
            )

            sampled_ids = torch.tensor(sampled_ids).to(batch["input_ids"].device)
            batch["target_ids"] = sampled_ids
            batch["target_mask"] = (
                (sampled_ids != model.get_tokenizer().pad_token_id)
                .int()
                .to(sampled_ids.device)
            )

    return batch


def scale_nonDiagonalElements(matrix, scale_lambda):
    scaled_matrix = scale_lambda * matrix + (1 - scale_lambda) * torch.diag_embed(
        torch.diagonal(matrix)
    )
    return scaled_matrix
