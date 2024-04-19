from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
)
from transformers.modeling_outputs import ModelOutput, SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer

from src import utils


@dataclass
class NLIClassifierOutput(ModelOutput):
    logits: torch.FloatTensor | None = None
    probs: torch.FloatTensor | None = None
    entailment_probs: torch.FloatTensor | None = None
    is_entailment: torch.BoolTensor | None = None

    def cpu(self) -> "NLIClassifierOutput":
        return NLIClassifierOutput(
            logits=self.logits.cpu() if self.logits is not None else None,
            probs=self.probs.cpu() if self.probs is not None else None,
            entailment_probs=self.entailment_probs.cpu() if self.entailment_probs is not None else None,
            is_entailment=self.is_entailment.cpu() if self.is_entailment is not None else None,
        )

    def to(self, device: str) -> "NLIClassifierOutput":
        return NLIClassifierOutput(
            logits=self.logits.to(device) if self.logits is not None else None,
            probs=self.probs.to(device) if self.probs is not None else None,
            entailment_probs=self.entailment_probs.to(device) if self.entailment_probs is not None else None,
            is_entailment=self.is_entailment.to(device) if self.is_entailment is not None else None,
        )


class NLIProcessor:
    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel

    def step(self, *args, **kwargs) -> NLIClassifierOutput:
        raise NotImplementedError

    def collate_fn(self, data_list: list[tuple[str, str]]) -> BatchEncoding:
        raise NotImplementedError


@dataclass
class TRUEProcessor(NLIProcessor):
    MODEL_NAME = "google/t5_xxl_true_nli_mixture"
    max_seq_len: int = 512

    def __post_init__(self) -> None:
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME,
            model_max_length=self.max_seq_len,
        )
        self.model: T5ForConditionalGeneration = (
            T5ForConditionalGeneration.from_pretrained(self.MODEL_NAME).eval().cpu()
        )
        utils.freeze_params(self.model)

    @torch.inference_mode()
    def step(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        **_,
    ) -> NLIClassifierOutput:
        pred_labels: list[list[int]] = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=5,
            do_sample=False,
        ).tolist()
        pred_labels = self.tokenizer.batch_decode(pred_labels, skip_special_tokens=True)

        # TODO: these probabilities are dummy values
        entailment_probs = torch.FloatTensor([float(label.strip() == "1") for label in pred_labels])
        other_probs = torch.FloatTensor([float(label.strip() != "1") for label in pred_labels])
        probs = torch.cat([other_probs, entailment_probs], dim=0)

        is_entailment = torch.BoolTensor([label.strip() == "1" for label in pred_labels])

        return NLIClassifierOutput(
            logits=probs,
            probs=probs,
            entailment_probs=entailment_probs,
            is_entailment=is_entailment,
        )

    def collate_fn(self, data_list: list[tuple[str, str]]) -> BatchEncoding:
        texts = [f"premise: {premise} hypothesis: {hypothesis}" for premise, hypothesis in data_list]
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_seq_len,
        )


@dataclass
class DeBERTaProcessor(NLIProcessor):
    # label mapping from https://huggingface.co/microsoft/deberta-v2-xxlarge-mnli/blob/main/config.json
    ENTAILMENT_ID: int = 2
    model_name = "microsoft/deberta-v2-xxlarge-mnli"
    max_seq_len: int = 512

    def __post_init__(self) -> None:
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=self.max_seq_len,
        )
        self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(self.model_name).eval().cpu()
        utils.freeze_params(self.model)

    @torch.inference_mode()
    def step(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        token_type_ids: torch.LongTensor = None,
        **_,
    ) -> NLIClassifierOutput:
        out: SequenceClassifierOutput = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        probs = out.logits.softmax(dim=-1)
        entailment_probs = probs[:, self.ENTAILMENT_ID]
        other_probs = probs[:, : self.ENTAILMENT_ID].sum(dim=-1)
        is_entailment = entailment_probs > other_probs

        return NLIClassifierOutput(
            logits=out.logits,
            probs=probs,
            entailment_probs=entailment_probs,
            is_entailment=is_entailment,
        )

    def collate_fn(self, data_list: list[tuple[str, str]]) -> BatchEncoding:
        premise, hypothesis = zip(*data_list)
        return self.tokenizer(
            premise,
            hypothesis,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_seq_len,
        )


@dataclass
class RoBERTaProcessor(DeBERTaProcessor):
    model_name = "roberta-large-mnli"


PROCESSORS = {
    # https://huggingface.co/google/t5_xxl_true_nli_mixture
    # The input format for the model is: "premise: PREMISE_TEXT hypothesis: HYPOTHESIS_TEXT".
    "true": TRUEProcessor,
    "deberta": DeBERTaProcessor,
    "roberta": RoBERTaProcessor,
}


@dataclass
class NLIClassifier:
    processor_name: str
    device: str
    batch_size: int = 512
    max_seq_len: int = 512
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        if self.processor_name not in PROCESSORS:
            raise ValueError(f"Model name {self.processor_name} is not supported.")
        processor_class = PROCESSORS[self.processor_name]
        self.processor: NLIProcessor = processor_class()

    def create_loader(self, premise: list[str], hypothesis: list[str]) -> DataLoader:
        return DataLoader(
            list(zip(premise, hypothesis)),
            collate_fn=self.processor.collate_fn,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def concat_outputs(self, outputs: list[NLIClassifierOutput]) -> NLIClassifierOutput:
        logits, probs = [], []
        entailment_probs, is_entailment = [], []
        for output in outputs:
            logits.append(output.logits)
            probs.append(output.probs)
            entailment_probs.append(output.entailment_probs)
            is_entailment.append(output.is_entailment)

        return NLIClassifierOutput(
            logits=torch.cat(logits, dim=0),
            probs=torch.cat(probs, dim=0),
            entailment_probs=torch.cat(entailment_probs, dim=0),
            is_entailment=torch.cat(is_entailment, dim=0),
        )

    @torch.inference_mode()
    def run_nli(
        self,
        premise: list[str],
        hypothesis: list[str],
    ) -> NLIClassifierOutput:
        self.processor.model.to(self.device)
        data_loader = self.create_loader(premise, hypothesis)
        outputs = []

        for batch in tqdm(
            data_loader,
            total=len(data_loader),
            leave=False,
            dynamic_ncols=True,
            desc="NLI Prediction",
        ):
            batch: BatchEncoding = batch.to(self.device)
            with torch.cuda.amp.autocast(dtype=self.dtype):
                out: NLIClassifierOutput = self.processor.step(**batch)
            outputs.append(out.cpu())

        self.processor.model.cpu()
        return self.concat_outputs(outputs)

    @torch.inference_mode()
    def split_and_rephrase_nli(
        self,
        complex: list[str],
        simple: list[str],
        indices: list[int] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        assert len(complex) == len(simple)

        original_indices = []
        comps, simps = [], []
        decomposed_simple = utils.batch_sentence_tokenize(simple)

        if indices is None:
            indices = list(range(len(complex)))

        for idx, comp, decomposed_simp in zip(indices, complex, decomposed_simple):
            original_indices += [idx] * len(decomposed_simp)
            comps += [comp for _ in range(len(decomposed_simp))]
            simps += decomposed_simp

        out: NLIClassifierOutput = self.run_nli(premise=comps, hypothesis=simps)

        micro_results = pd.DataFrame(
            {
                "id": original_indices,
                "is_entailment": out.is_entailment.tolist(),
                "entailment_prob": out.entailment_probs.tolist(),
            }
        )
        micro_results["complex"] = comps
        micro_results["decomposed_simple"] = simps

        macro_results: pd.DataFrame = micro_results.groupby(by="id", as_index=False).agg(
            {
                "entailment_prob": "mean",
                "is_entailment": "all",
            }
        )
        macro_results["complex"] = complex
        macro_results["simple"] = simple

        return {
            "micro": micro_results,
            "macro": macro_results,
        }
