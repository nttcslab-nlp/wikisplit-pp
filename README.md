# WikiSplit++: Easy Data Refinement for Split and Rephrase

WikiSplit++ enhances the original WikiSplit by applying two techniques: filtering through NLI classification and sentence-order reversing, which help to remove noise and reduce hallucinations compared to the original WikiSplit.  
The preprocessed WikiSplit dataset that formed the basis for this can be found here.


## Description

The train split of WikiSplit++ includes the train, val, and tune splits from WikiSplit.
The origin of each data item, whether it is from the train, val, or tune split of WikiSplit, can be identified by the split entry within the data.
We did not use the test split of WikiSplit as it is being used for the construction of Wiki-BM.
We re-divided the train, val, and tune splits of WikiSplit into new train, val, and test splits for intrinsic evaluations.


## Instrallation & Preparation


```bash
# install dependencies
rye sync
source ./.venv/bin/activate

# download and preprocess the datasets
bash src/download.sh
bash src/create-datasets.sh
```


## Training

```bash
python src/train.py --method "split_reverse" --model_name "t5-small" --dataset_dir "./datasets" --dataset_name "wiki-split/entailment"
```

## License

This software is released under the NTT License, see [LICENSE.txt](https://github.com/nttcslab-nlp/wikisplit-pp/blob/main/LICENSE.txt).  
According to the license, it is not allowed to create pull requests. Please feel free to send issues.

[Our dataset](https://huggingface.co/datasets/cl-nagoya/wikisplit-pp) is publicly available on HuggingFace under the CC BY-SA 4.0 license.