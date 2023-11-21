import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.utils import round_nestedList, flatten_list

from src.model.utils import computeLogProb_perChoice, greedy_generation_encoderDecoder


class EncoderDecoderWrapper(nn.Module):
    """ """

    def __init__(self, transformer, tokenizer, model_config):
        super().__init__()
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.tokenizer.padding_side = "right"

    def forward(self, batch, average_sequenceLength=True):
        transformer_outputs = self.transformer(
            input_ids=batch["input_ids"],
            attention_mask=batch["input_mask"],
            labels=batch["target_ids"],
        )

        # [batch_size, max_target_len, vocab_size]
        target_logits = transformer_outputs[1].float()
        vocab_size = target_logits.shape[-1]

        # Compute the log probability of the ids for all choices with respect to the logits
        # [batch_size x max_target_len]
        logProbs_ofTargetIds = F.cross_entropy(
            target_logits.reshape(-1, vocab_size),
            batch["target_ids"].reshape(-1),
            reduction="none",
        )
        # Zero out log_probs for target_ids with no loss
        target_mask = batch["target_mask"].reshape(-1)
        logProbs_ofTargetIds_zeroOutPadIds = logProbs_ofTargetIds * target_mask

        if average_sequenceLength:
            loss = torch.sum(logProbs_ofTargetIds_zeroOutPadIds) / torch.sum(
                target_mask
            )
        else:
            loss = torch.sum(logProbs_ofTargetIds_zeroOutPadIds)

        return loss, {"loss": loss.detach().cpu().item()}

    def _broadcast_tensors(self, input_masks, encoder_outputs, num_choices):
        """
        Broadcast the input masks and encoder outputs to account for multiple choices per input

        Args:
            input_masks: [batch_size, max_input_len]
            encoder_outputs: BaseModelOutput object from HuggingFace where the first element is
                            the hidden states of the encoder at the last layer
                            [batch_size, max_input_len, ff_dim]
            num_choices:

        Returns:
            input_masks: [batch_size x num_choices, max_input_len]
            encoder_outputs: BaseModelOutput object from HuggingFace where the first element is
                            the hidden states of the encoder at the last layer
                            [batch_size x num_choices, max_input_len, ff_dim]
        """
        input_masks = torch.repeat_interleave(input_masks, num_choices, dim=0)
        encoder_outputs = (
            torch.repeat_interleave(encoder_outputs[0], num_choices, dim=0),
        )
        return input_masks, encoder_outputs

    def predict_mulChoice(self, batch, length_normalization):
        """

        Args:
            batch:
            length_normalization:

        Returns:
            pred_choice: [batch_size, ]
            score_ofChoices: [batch_size, num_choices]
            logProbs_ofAllChoicesIds: [batch_size, num_choices, max_choice_len]
            len_allChoices: [batch_size]
        """
        encoder_outputs = self.transformer.get_encoder()(
            batch["input_ids"],
            batch["input_mask"],
        )

        num_ofAnswerChoices = (
            batch["answer_choices_ids"].shape[0] // batch["input_mask"].shape[0]
        )

        input_mask, encoder_outputs = self._broadcast_tensors(
            batch["input_mask"], encoder_outputs, num_ofAnswerChoices
        )

        # WARNING: The loss at transformer_outputs[0] is not valid, since allChoices_ids uses a
        # pad token of 0 and so the loss will not be ignored for the pad tokens
        # The input mask is passed in for the cross encoder-decoder attention.
        transformer_outputs = self.transformer(
            attention_mask=input_mask,
            encoder_outputs=encoder_outputs,
            labels=batch["answer_choices_ids"],
        )

        # We used the logits for all choices to compute the log probs per example since
        # the loss returned in transformer_outputs will average the negative log probs across
        # examples
        # [batch_size x num_choices, max_choice_len, vocab_size]
        logits_ofAnswerChoicesIds = transformer_outputs[1].float()
        maxLen_ofAnswerChoices = logits_ofAnswerChoicesIds.shape[1]
        vocab_size = logits_ofAnswerChoicesIds.shape[-1]

        # Compute the log probability of the ids for all choices with respect to the logits
        # [batch_size x num_choices x max_choice_len]
        logProb_ofAnswerChoicesIds = -F.cross_entropy(
            logits_ofAnswerChoicesIds.view(-1, vocab_size),
            batch["answer_choices_ids"].view(-1),
            reduction="none",
        )

        (
            logProb_ofAnswerChoices,
            logProb_ofAnswerChoiceIds_zeroOutPadIds,
            answerChoices_len,
        ) = computeLogProb_perChoice(
            logProb_ofAnswerChoicesIds,
            batch["answer_choices_mask"],
            batch["non_null_answer_choices"],
            num_ofAnswerChoices,
            maxLen_ofAnswerChoices,
            length_normalization,
        )

        _, predicted_choice = torch.max(logProb_ofAnswerChoices, dim=1)

        return (
            predicted_choice.cpu().numpy().tolist(),
            round_nestedList(logProb_ofAnswerChoices.cpu().numpy().tolist(), 5),
            round_nestedList(
                logProb_ofAnswerChoiceIds_zeroOutPadIds.cpu().numpy().tolist(), 4
            ),
            answerChoices_len.cpu().numpy().tolist(),
        )

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, batch, max_generationLength, sample_tokens):
        """

        Args:
            batch:
            max_generationLength:

        Returns:

        """
        generation_output = self.transformer.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["input_mask"],
            max_new_tokens=max_generationLength,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            return_dict_in_generate=True,
            do_sample=sample_tokens,
        )
        generated_txt = self.tokenizer.batch_decode(
            generation_output["sequences"], skip_special_tokens=True
        )

        return generation_output["sequences"].cpu().numpy().tolist(), generated_txt

    def tokenize_fn(self, datapoint_batched, device):
        """
        Tokenizer for the model

        Args:
            datapoint_batched (_type_): a datapoint that already has been batched
            device (_type_):
        """
        keys_toTokenize = ["input", "answer_choices", "target"]

        for key in keys_toTokenize:
            if key in datapoint_batched:
                if key == "answer_choices":
                    text = flatten_list(datapoint_batched[key])
                else:
                    text = datapoint_batched[key]

                tokenized_dict = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding="longest",
                    truncation="longest_first",
                )

                input_ids = tokenized_dict["input_ids"]
                attention_mask = tokenized_dict["attention_mask"]

                if device is not None:
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)

                datapoint_batched[f"{key}_ids"] = input_ids
                datapoint_batched[f"{key}_mask"] = attention_mask

        return datapoint_batched
