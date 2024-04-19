import copy
from dataclasses import dataclass

from torch import FloatTensor, Tensor, nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    ModelOutput,
    Seq2SeqLMOutput,
)
from transformers.models.t5.modeling_t5 import (
    T5Config,
    T5ForConditionalGeneration,
    T5Stack,
)


@dataclass
class SplitWithReconstionOutput(ModelOutput):
    seq2seq_loss: FloatTensor = None
    rec_loss: FloatTensor | None = None


class SplitAndRephraseModel(nn.Module):
    model: T5ForConditionalGeneration

    def __init__(self, model_name: str):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def forward(self, *args, **kwargs) -> SplitWithReconstionOutput:
        outputs: Seq2SeqLMOutput = self.model(*args, **kwargs)
        return SplitWithReconstionOutput(
            seq2seq_loss=outputs.loss,
            rec_loss=None,
        )

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


class SplitAndRephraseWithReconstructorModel(nn.Module):
    enc_dec: T5ForConditionalGeneration
    rec: T5Stack

    def __init__(
        self,
        model_name: str,
        copy_dec_for_rec: bool,
        num_rec_layers: int = None,
    ):
        super().__init__()
        self.enc_dec = T5ForConditionalGeneration.from_pretrained(model_name)
        config: T5Config = self.enc_dec.config

        if copy_dec_for_rec:
            self.rec = copy.deepcopy(self.enc_dec.decoder)
        else:
            rec_config = copy.deepcopy(config)
            rec_config.is_decoder = True
            rec_config.is_encoder_decoder = False
            rec_config.num_layers = num_rec_layers or config.num_decoder_layers
            self.rec = T5Stack(rec_config, self.enc_dec.shared)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        dec_labels: Tensor,
        rec_labels: Tensor,
        **kwargs,
    ) -> SplitWithReconstionOutput:
        outputs: Seq2SeqLMOutput = self.enc_dec(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=dec_labels,
            output_hidden_states=True,
            **kwargs,
        )
        seq2seq_loss = outputs.loss
        dec_last_hidden_state = outputs.decoder_hidden_states[-1]

        # shifts entire sequences right and fills decoder_start_token_id at the first position
        rec_input_ids = self.enc_dec._shift_right(rec_labels)

        # if attention_mask is None, a causal attention mask will be created
        rec_outputs: BaseModelOutputWithPastAndCrossAttentions = self.rec(
            input_ids=rec_input_ids,
            encoder_hidden_states=dec_last_hidden_state,
        )
        rec_last_hidden_state = rec_outputs.last_hidden_state
        rec_logits = self.enc_dec.lm_head(rec_last_hidden_state)
        rec_loss = self.loss_fct(
            rec_logits.view(-1, rec_logits.size(-1)),
            rec_labels.view(-1),
        )

        return SplitWithReconstionOutput(
            seq2seq_loss=seq2seq_loss,
            rec_loss=rec_loss,
        )

    def generate(self, *args, **kwargs):
        return self.enc_dec.generate(*args, **kwargs)


class SplitAndRephraseWithRecursiveReconstructorModel(nn.Module):
    enc_dec: T5ForConditionalGeneration

    def __init__(self, model_name: str):
        super().__init__()
        self.enc_dec = T5ForConditionalGeneration.from_pretrained(model_name)
        self.hidden_size: int = self.enc_dec.config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )
        self.rec = self.enc_dec.decoder

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        dec_labels: Tensor,
        rec_labels: Tensor,
        **kwargs,
    ) -> SplitWithReconstionOutput:
        outputs: Seq2SeqLMOutput = self.enc_dec(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=dec_labels,
            output_hidden_states=True,
            **kwargs,
        )
        seq2seq_loss = outputs.loss

        # shifts entire sequences right and fills decoder_start_token_id at the first position
        rec_input_ids = self.enc_dec._shift_right(rec_labels)

        dec_last_hidden_state = outputs.decoder_hidden_states[-1]
        dec_last_hidden_state = self.mlp(dec_last_hidden_state)

        # if attention_mask is None, a causal attention mask will be created
        rec_outputs: BaseModelOutputWithPastAndCrossAttentions = self.rec(
            input_ids=rec_input_ids,
            encoder_hidden_states=dec_last_hidden_state,
        )

        rec_last_hidden_state = rec_outputs.last_hidden_state
        rec_logits = self.enc_dec.lm_head(rec_last_hidden_state)
        rec_loss = self.loss_fct(
            rec_logits.view(-1, rec_logits.size(-1)),
            rec_labels.view(-1),
        )

        return SplitWithReconstionOutput(
            seq2seq_loss=seq2seq_loss,
            rec_loss=rec_loss,
        )

    def generate(self, *args, **kwargs):
        return self.enc_dec.generate(*args, **kwargs)
