import torch
import re


def computeLogProb_perChoice(
    logProb_ofAnswerChoiceIds,
    mask_ofAnswerChoices,
    nonNullAnswerChoices_mask,
    num_answerChoices,
    maxLen_ofAnswerChoices,
    length_normalization,
):
    """
    Get the log probs of each choice by summing or averaging the log probs of the ids

    Args:
        logProb_ofAnswerChoiceIds (_type_): [batch_size * num_answerChoices * max_answerChoiceLen]
        answerChoices_mask (_type_): [batch_size * num_answerChoices * max_answerChoiceLen, ]
        num_answerChoices (_type_):
        maxLen_ofAnswerChoices (_type_):
        length_normalization (_type_):

    Returns:
        logProb_ofAnswerChoices:
        logProb_ofAnswerChoiceIds_zeroOutPadIds:
        answerChoices_len:
    """

    # [batch_size, num_answerChoices, max_answerChoiceLen]
    logProb_ofAnswerChoiceIds = logProb_ofAnswerChoiceIds.reshape(
        -1, num_answerChoices, maxLen_ofAnswerChoices
    )

    mask_ofAnswerChoices = mask_ofAnswerChoices.reshape(
        -1, num_answerChoices, maxLen_ofAnswerChoices
    )
    # Zero out padded out tokens so we their log probability is not included
    logProb_ofAnswerChoiceIds_zeroOutPadIds = (
        logProb_ofAnswerChoiceIds * mask_ofAnswerChoices
    )

    # Sum the log_prob across ids per answer choice
    logProb_ofAnswerChoices = torch.sum(logProb_ofAnswerChoiceIds_zeroOutPadIds, dim=2)

    answerChoices_len = torch.sum(mask_ofAnswerChoices, dim=2)

    if length_normalization:
        logProb_ofAnswerChoices = logProb_ofAnswerChoices / answerChoices_len

    nonNullAnswerChoices_mask = nonNullAnswerChoices_mask.reshape(-1, num_answerChoices)

    # For answer choices which are null, we mask them out by setting them to the smallest value
    logProb_ofAnswerChoices = (1 - nonNullAnswerChoices_mask) * torch.finfo(
        logProb_ofAnswerChoices.dtype
    ).min + nonNullAnswerChoices_mask * logProb_ofAnswerChoices

    return (
        logProb_ofAnswerChoices,
        logProb_ofAnswerChoiceIds_zeroOutPadIds,
        answerChoices_len,
    )


def get_trainableParameters(model, trainableParameter_regex):
    """
    Gets the trainable parameters

    Args:
        model:
        trainableParameter_regex:

    Returns:

    """
    trainable_parameters = {}
    num_parameters = 0

    for parameter_name, parameter_value in model.named_parameters():
        # Ignore the embed_token weight since it is not a trainable parameter but shared between
        # the encoder and decoder in T5
        if re.fullmatch(trainableParameter_regex, parameter_name) and not re.fullmatch(
            "transformer.*.embed_tokens.weight", parameter_name
        ):
            if parameter_value.requires_grad:
                trainable_parameters[parameter_name] = parameter_value
                num_parameters += parameter_value.numel()

    return trainable_parameters, num_parameters


def expand_lora(checkpoint):
    """
    Multiply out lora_a and lora_b parameters and use those as the lora weights
    or use expanded out lora weights already in checkpoint
    """
    expanded_checkpoint = {}
    for parameter_name, parameter in checkpoint.items():
        if "lora_a" in parameter_name:
            lora_b_parameterName = parameter_name.replace("lora_a", "lora_b")
            lora_b_parameter = checkpoint[lora_b_parameterName]

            lora_parameter = torch.matmul(lora_b_parameter.T, parameter.T).T
            lora_parameterName = parameter_name.replace("lora_a", "lora")
            expanded_checkpoint[lora_parameterName] = lora_parameter
        elif "lora_b" in parameter_name:
            continue
        else:
            assert "lora" in parameter_name
            expanded_checkpoint[parameter_name] = parameter
    return expanded_checkpoint


def merge_lora(pretrained_checkpoint, checkpoint):
    merged_checkpoint = {}
    # If checkpoint is not merged, then all parameter_names are either lora_a or lora_b
    # Otherwise, if checkpoint is already merged, then the parameter_names will match lora and no merging is needed
    for parameter_name, parameter in checkpoint.items():
        if "lora_a" in parameter_name:
            lora_b_parameterName = parameter_name.replace("lora_a", "lora_b")
            lora_b_parameter = checkpoint[lora_b_parameterName]
            rank = lora_b_parameter.shape[0]
            merged_parameterName = parameter_name.replace(
                "lora_a.lora_multiply.weight", "lora.weight"
            )

            original_parameter = pretrained_checkpoint[merged_parameterName]
            # parameter is lora_a_parameter
            lora_parameter = (
                torch.matmul(lora_b_parameter.T, parameter.T).T / rank
                + original_parameter
            )

            merged_checkpoint[merged_parameterName] = lora_parameter
        elif "lora_b" in parameter_name:
            continue
        else:
            print(parameter_name)
            # If checkpoint already has been merged, then just add it back
            merged_checkpoint[parameter_name] = parameter
    return merged_checkpoint


def merge_ia3(pretrained_checkpoint, checkpoint):
    merged_checkpoint = {}
    for parameter_name, ia3_parameter in checkpoint.items():
        # If ia3_vector in parameter_name, then checkpoint has to be merged
        # Otherwise checkpoint does not need to be merged
        if "ia3_vector" in parameter_name:
            assert "ia3_vector" in parameter_name
            merged_parameterName = parameter_name.replace(
                "ia3_layer.ia3_vector", "ia3_linear_layer.weight"
            )
            original_parameter = pretrained_checkpoint[merged_parameterName]
            # Have to move ia3_parameter to cuda since it originally was not saved on gpu
            merged_parameter = torch.mul(
                ia3_parameter.to(original_parameter.device)[:, None], original_parameter
            )
            merged_checkpoint[merged_parameterName] = merged_parameter
        else:
            merged_checkpoint[parameter_name] = ia3_parameter

    return merged_checkpoint


def greedy_generation_encoderDecoder(
    transformer,
    input_ids,
    input_mask,
    bos_tokenId,
    eos_tokenId,
    pad_tokenId,
    max_generationLength,
):
    """
    Assumes model is encoder_decoder model and caches input first

    Args:
        model:
        input_ids:
        input_mask:
        bos_tokenId:
        eos_tokenId:
        pad_tokenId:
        max_generationLength:

    Returns:
        generated_ids: [batch_size, max_generationLength]
    """
    past_key_values = None
    batch_size = input_ids.shape[0]

    # Decode starting with bos_token_id
    # [batch_size, 1]
    current_decoderInputIds = torch.tensor([bos_tokenId] * batch_size)[:, None].to(
        input_ids.device
    )
    # Decoder mask is fixed to always be 1. We don't need to ignore any tokens in the decoder
    # since we just truncate any token after the eos token
    # [batch_size, 1]
    current_decoderMask = torch.ones((batch_size, 1)).to(input_ids.device)

    encoder_outputs = transformer.get_encoder()(input_ids, input_mask)

    generated_ids = current_decoderInputIds

    hasSequence_hitEOS = torch.zeros(size=(batch_size, 1), dtype=torch.int).to(
        input_ids.device
    )

    for i in range(max_generationLength):
        # attention_mask must be passed in for encoder_decoder models, even if we pass the
        # encoder_outputs, since the attention_mask is used to compute the cross_attention mask
        # for encoder decoder models
        output = transformer(
            attention_mask=input_mask,
            decoder_input_ids=current_decoderInputIds,
            decoder_attention_mask=current_decoderMask,
            encoder_outputs=encoder_outputs,
            use_cache=True,
            past_key_values=past_key_values,
        )

        # Update current key values
        past_key_values = output.past_key_values

        predicted_nextToken = torch.argmax(output.logits, -1)

        # If sequence has hit end, then every token afterwards should be a PAD token
        predicted_nextToken = (
            1 - hasSequence_hitEOS
        ) * predicted_nextToken + hasSequence_hitEOS * pad_tokenId

        generated_ids = torch.cat((generated_ids, predicted_nextToken), dim=1)

        # Update whether has sequence has hit end of sequence
        isToken_EOSToken = predicted_nextToken == eos_tokenId
        hasSequence_hitEOS = torch.bitwise_or(hasSequence_hitEOS, isToken_EOSToken)

        # Exit loop if every sequence has hit EOS
        if torch.sum(hasSequence_hitEOS) == batch_size:
            break

        current_decoderInputIds = predicted_nextToken

    return generated_ids
