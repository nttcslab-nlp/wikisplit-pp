import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer

from src import utils


@torch.inference_mode()
@torch.cuda.amp.autocast(dtype=torch.bfloat16)
def main():
    model_name = "microsoft/deberta-v2-xxlarge-mnli"
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(model_name).eval()

    def collate_fn(data_list: list[dict[str, int | str | list[str]]]) -> BatchEncoding:
        premise = [d["complex"] for d in data_list]
        hypothesis = [d["simples"][0] for d in data_list]
        return tokenizer(
            premise,
            hypothesis,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )

    dataset = utils.load_jsonl("datasets/hsplit/all.jsonl").to_dict("records")

    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda:3")
    model = model.to(device)

    counter = [0, 0, 0]
    probs = [0, 0, 0]

    for batch in tqdm(data_loader):
        out: ModelOutput = model(**batch.to(device))
        prob = out.logits.softmax(dim=-1).tolist()
        ids = out.logits.argmax(dim=-1).tolist()

        for idx in ids:
            counter[idx] += 1
        for ps in prob:
            probs[0] += ps[0]
            probs[1] += ps[1]
            probs[2] += ps[2]

    print([f"{c * 100/ len(dataset):.2f}" for c in counter])
    print([f"{p * 100/ len(dataset):.2f}" for p in probs])


if __name__ == "__main__":
    main()
