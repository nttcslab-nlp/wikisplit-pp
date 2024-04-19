import math
from itertools import count
from pathlib import Path

import torch
from tqdm import tqdm
from transformers.optimization import get_linear_schedule_with_warmup

from src import utils
from src.experiment import Args, SplitAndRephraseExperiment


def main(args: Args):
    utils.set_seed(args.seed)

    exp = SplitAndRephraseExperiment(args=args)

    train_dataset = utils.load_jsonl(args.data_dir / "train.jsonl").to_dict("records")
    val_dataset = utils.load_jsonl(args.data_dir / "val.jsonl").to_dict("records")
    test_dataset = utils.load_jsonl(args.data_dir / "test.jsonl").to_dict("records")

    train_dataloader = exp.create_loader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = exp.create_loader(val_dataset, batch_size=args.eval_batch_size)
    test_dataloader = exp.create_loader(test_dataset, batch_size=args.eval_batch_size)

    optimizer = torch.optim.AdamW(params=exp.model.parameters(), lr=args.lr)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(args.num_warmup_ratio * args.steps),
        num_training_steps=args.steps,
    )

    best_val_loss: float = exp.evaluate(val_dataloader)
    best_epoch, best_step = 0, 0
    best_state_dict = exp.clone_state_dict()
    val_metrics = {
        "epoch": 0,
        "step": 0,
        "train_loss": float("inf"),
        "val_loss": best_val_loss,
    }
    exp.log(val_metrics)
    train_losses = []

    with tqdm(
        total=args.steps,
        dynamic_ncols=True,
        desc="Training",
    ) as pbar:
        total_steps = 0
        scaler = torch.cuda.amp.GradScaler(enabled=not args.not_amp)
        exp.model.train()

        for epoch in count():
            for batch in train_dataloader:
                with torch.cuda.amp.autocast(enabled=not args.not_amp, dtype=args.dtype):
                    loss = exp.step(batch)
                pbar.set_postfix({"train_loss": loss.detach().item()})
                train_losses.append(loss.detach().item())

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)

                scale = scaler.get_scale()
                scaler.update()
                if scale <= scaler.get_scale():
                    lr_scheduler.step()

                pbar.update(1)
                total_steps += 1

                if (total_steps % args.eval_interval == 0) or (total_steps == args.steps):
                    val_loss = exp.evaluate(val_dataloader)

                    if best_val_loss > val_loss:
                        best_val_loss = val_loss
                        best_epoch, best_step = epoch, total_steps
                        best_state_dict = exp.clone_state_dict()

                    val_metrics = {
                        "epoch": epoch,
                        "step": total_steps,
                        "train_loss": sum(train_losses) / len(train_losses),
                        "val_loss": val_loss,
                    }
                    exp.log(val_metrics)
                    train_losses = []
                    exp.model.train()

                if total_steps == args.steps:
                    break
            else:
                continue
            break

    exp.model.load_state_dict(best_state_dict)
    exp.model.eval().to(args.device)

    test_loss = exp.evaluate(test_dataloader)
    metrics, results = exp.evaluate_split_and_rephrase()

    utils.save_json(
        {
            "general": {
                "best_epoch": best_epoch,
                "best_step": best_step,
                "best_val_loss": best_val_loss,
                "best_val_ppl": math.exp(best_val_loss),
                "test_loss": test_loss,
                "test_ppl": math.exp(test_loss),
            },
            **metrics,
        },
        args.output_dir / "all-metrics.json",
    )

    for dataset_name, dataset_results in results.items():
        dir: Path = args.output_dir / dataset_name
        utils.save_json(metrics[dataset_name], dir / "metrics.json")

        dfs = []
        for result_name, result in dataset_results.items():
            utils.save_jsonl(result, dir / f"{result_name}.jsonl")
            if "micro" not in result_name:
                dfs.append(result)

        utils.save_jsonl(utils.merge_dfs(dfs), dir / "results.jsonl")

    utils.save_config(args, args.output_dir / "config.json")


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
