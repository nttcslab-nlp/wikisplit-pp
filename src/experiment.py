from datetime import datetime
from pathlib import Path

import torch
from tap import Tap
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer

from src import utils
from src.evaluations import SplitAndRephraseEvaluator
from src.models import (
    SplitAndRephraseModel,
    SplitAndRephraseWithReconstructorModel,
    SplitAndRephraseWithRecursiveReconstructorModel,
    SplitWithReconstionOutput,
)
from src.nli import NLIClassifier


class Args(Tap):
    method: str = "split"
    model_name: str = "t5-small"
    dataset_dir: Path = "./datasets"
    dataset_name: str = "wiki-split/base"

    # steps: int = 20_000
    steps: int = 2
    batch_size: int = 32
    lr: float = 1e-4
    num_warmup_ratio: float = 0.1
    max_seq_len: int = 512

    rec_loss_weight: float = 1.0
    copy_dec_for_rec: bool = True

    eval_interval: int = 1_000
    eval_batch_size: int = 128

    gen_batch_size: int = 16
    num_beams: int = 10
    no_repeat_ngram_size: int = 3

    nli_processor_name: str = "deberta"
    nli_batch_size: int = 16

    not_amp: bool = False
    device: str = "cuda:0"
    dtype: utils.torch_dtype = "bf16"
    seed: int = None

    def process_args(self):
        date, time = datetime.now().strftime("%Y-%m-%d/%H-%M-%S").split("/")
        self.output_dir = self.make_output_dir(
            "outputs",
            self.method,
            self.dataset_name,
            self.model_name,
            date,
            time,
        )

    def make_output_dir(self, *args) -> Path:
        args = [str(a).replace("/", "__") for a in args]
        output_dir = Path(*args)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @property
    def data_dir(self) -> Path:
        return self.dataset_dir / self.dataset_name


class SplitAndRephraseExperiment:
    args: Args
    model: SplitAndRephraseModel = None
    tokenizer: PreTrainedTokenizer = None
    nli_classifier: NLIClassifier = None

    def __init__(self, args: Args) -> None:
        super().__init__()
        self.args = args

        if args.method in ["split", "split_reverse"]:
            self.model: SplitAndRephraseModel = SplitAndRephraseModel(args.model_name)
            self.step = self.step_split

        elif args.method in ["split_rec", "split_rec_reverse"]:
            self.model: SplitAndRephraseWithReconstructorModel = SplitAndRephraseWithReconstructorModel(
                model_name=args.model_name,
                copy_dec_for_rec=args.copy_dec_for_rec,
            )
            self.step = self.step_split_rec

        elif args.method in ["split_rec2", "split_rec2_reverse"]:
            self.model: SplitAndRephraseWithRecursiveReconstructorModel = (
                SplitAndRephraseWithRecursiveReconstructorModel(model_name=args.model_name)
            )
            self.step = self.step_split_rec

        else:
            raise ValueError(f"Invalid method: {self.args.method}")

        self.model: SplitAndRephraseModel = self.model.eval().to(args.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            model_max_length=args.max_seq_len,
        )

        self.nli_classifier = NLIClassifier(
            processor_name=args.nli_processor_name,
            device=args.device,
            batch_size=args.eval_batch_size,
            max_seq_len=args.max_seq_len,
            dtype=args.dtype,
        )
        self.evaluator = SplitAndRephraseEvaluator(
            nli_classifier=self.nli_classifier,
            batch_size=args.eval_batch_size,
            device=args.device,
        )

    def encode(self, text: list[str]) -> BatchEncoding:
        return self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.args.max_seq_len,
        )

    def collate_fn(self, batch: list[dict]) -> BatchEncoding:
        simple = self.encode([d["simple"] for d in batch])
        complex = self.encode([d["complex"] for d in batch])

        complex["labels"] = simple.input_ids.clone()
        complex["labels"][complex["labels"][:, :] == self.tokenizer.pad_token_id] = -100

        simple["labels"] = complex.input_ids.clone()
        simple["labels"][simple["labels"][:, :] == self.tokenizer.pad_token_id] = -100

        return BatchEncoding(
            {
                "simple": simple,
                "complex": complex,
            }
        )

    def collate_fn_reverse(self, batch: list[dict]) -> BatchEncoding:
        simple = [d["simple"] for d in batch]
        # To avoid becoming a model that simply outputs the input as is, reverse the order of simple sentences.
        simple = [" ".join(utils.sentence_tokenize(simp)[::-1]) for simp in simple]
        simple = self.encode(simple)

        complex = self.encode([d["complex"] for d in batch])

        complex["labels"] = simple.input_ids.clone()
        complex["labels"][complex["labels"][:, :] == self.tokenizer.pad_token_id] = -100

        simple["labels"] = complex.input_ids.clone()
        simple["labels"][simple["labels"][:, :] == self.tokenizer.pad_token_id] = -100

        return BatchEncoding(
            {
                "simple": simple,
                "complex": complex,
            }
        )

    def create_loader(
        self,
        dataset: list[str],
        batch_size: int = None,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> DataLoader:
        if self.args.method in ["split", "split_rec", "split_rec2"]:
            collate_fn = self.collate_fn
        elif self.args.method in ["split_reverse", "split_rec_reverse", "split_rec2_reverse"]:
            collate_fn = self.collate_fn_reverse
        else:
            raise ValueError(f"Invalid method: {self.args.method}")

        return DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=batch_size or self.args.eval_batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            drop_last=drop_last,
        )

    def step_split(self, batch: BatchEncoding) -> torch.FloatTensor:
        batch: BatchEncoding = batch.to(self.args.device)
        out: SplitWithReconstionOutput = self.model.forward(**batch.complex)
        return out.seq2seq_loss

    def step_split_rec(self, batch: BatchEncoding) -> torch.FloatTensor:
        batch: BatchEncoding = batch.to(self.args.device)
        out: SplitWithReconstionOutput = self.model.forward(
            input_ids=batch.complex.input_ids,
            attention_mask=batch.complex.attention_mask,
            dec_labels=batch.complex.labels,
            rec_labels=batch.simple.labels,
        )
        loss = out.seq2seq_loss + self.args.rec_loss_weight * out.rec_loss
        return loss

    def clone_state_dict(self) -> dict:
        return {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}

    @torch.inference_mode()
    def split_and_rephrase(self, complex: list[str]) -> list[str]:
        generated_simple = []
        dataloader = DataLoader(
            complex,
            collate_fn=self.encode,
            batch_size=self.args.gen_batch_size,
            num_workers=4,
            pin_memory=True,
        )
        for batch in tqdm(
            dataloader,
            desc="Generating",
            total=len(dataloader),
            dynamic_ncols=True,
            leave=False,
        ):
            batch: BatchEncoding = batch.to(self.args.device)
            generated_simple += self.model.generate(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                max_length=128,
                num_return_sequences=1,
                num_beams=self.args.num_beams,
                no_repeat_ngram_size=self.args.no_repeat_ngram_size,
            ).tolist()

        generated_simple = self.tokenizer.batch_decode(
            generated_simple,
            skip_special_tokens=True,
        )

        if self.args.method in ["split_reverse", "split_rec_reverse", "split_rec2_reverse"]:
            generated_simple = [" ".join(utils.sentence_tokenize(simp)[::-1]) for simp in generated_simple]

        utils.clear_cache()
        return generated_simple

    @torch.inference_mode()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()

        total_loss = 0
        for batch in tqdm(
            dataloader,
            total=len(dataloader),
            dynamic_ncols=True,
            leave=False,
            desc="Evaluating",
        ):
            with torch.cuda.amp.autocast(enabled=not self.args.not_amp, dtype=self.args.dtype):
                batch: BatchEncoding = batch.to(self.args.device)
                batch_size: int = batch.complex.input_ids.size(0)
                loss: torch.FloatTensor = self.step(batch)
                total_loss += loss.detach().item() * batch_size

        total_loss = total_loss / len(dataloader.dataset)
        utils.clear_cache()
        return total_loss

    @torch.inference_mode()
    def evaluate_split_and_rephrase(self) -> tuple[dict[str, dict], dict[str, list[dict]]]:
        utils.clear_cache()
        return self.evaluator(fn=self.split_and_rephrase)

    def log(self, metrics: dict) -> None:
        utils.log(metrics, self.args.output_dir / "log.csv")
        if metrics["train_loss"] == float("inf"):
            train_loss_log = f"train loss: {metrics['train_loss']:>10}"
        else:
            train_loss_log = f"train loss: {metrics['train_loss']:2.10f}"

        tqdm.write(
            f"epoch: {metrics['epoch']:>3} \t"
            f"step: {metrics['step']:>6} \t"
            f"{train_loss_log}\t"
            f"val loss: {metrics['val_loss']:2.10f}"
        )
