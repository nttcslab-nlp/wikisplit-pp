mkdir -p ./datasets

wiki-split() {
    mkdir -p ./datasets/wiki-split/{raw,detokenized,base}

    cp ./related-works/wiki-split/train.tsv ./datasets/wiki-split/raw/train.tsv &
    cp ./related-works/wiki-split/validation.tsv ./datasets/wiki-split/raw/val.tsv &
    cp ./related-works/wiki-split/tune.tsv ./datasets/wiki-split/raw/tune.tsv &
    # cp ./related-works/wiki-split/test.tsv ./datasets/wiki-split/raw/test.tsv &
    wait

    perl ./src/detokenizer.perl -l en -penn <./datasets/wiki-split/raw/train.tsv >./datasets/wiki-split/detokenized/train.tsv &
    perl ./src/detokenizer.perl -l en -penn <./datasets/wiki-split/raw/val.tsv >./datasets/wiki-split/detokenized/val.tsv &
    perl ./src/detokenizer.perl -l en -penn <./datasets/wiki-split/raw/tune.tsv >./datasets/wiki-split/detokenized/tune.tsv &
    # perl ./src/detokenizer.perl -l en -penn <./datasets/wiki-split/raw/test.tsv >./datasets/wiki-split/detokenized/test.tsv &
    wait

    python src/datasets/wiki_split.py
}

min-wiki-split() {
    mkdir -p ./datasets/min-wiki-split/{raw,detokenized,base}

    cp ./related-works/min-wiki-split/MinWikiSplit_v1_INLG2019.txt ./datasets/min-wiki-split/raw/all.txt
    perl ./src/detokenizer.perl -l en -penn <./datasets/min-wiki-split/raw/all.txt >./datasets/min-wiki-split/detokenized/all.txt

    python src/datasets/min_wiki_split.py
}

bisect() {
    mkdir -p ./datasets/bisect/{raw,detokenized,base}

    cp ./related-works/bisect/bisect/train.src ./datasets/bisect/raw/train.src
    cp ./related-works/bisect/bisect/train.dst ./datasets/bisect/raw/train.dst
    perl ./src/detokenizer.perl -l en -penn <./datasets/bisect/raw/train.src >./datasets/bisect/detokenized/train.src
    perl ./src/detokenizer.perl -l en -penn <./datasets/bisect/raw/train.dst >./datasets/bisect/detokenized/train.dst

    python src/datasets/run_bisect.py
}

hsplit() {
    mkdir -p ./datasets/hsplit/{raw,detokenized}

    cp ./related-works/cocoxu-simplification/data/turkcorpus/GEM/test.8turkers.tok.norm ./datasets/hsplit/raw/complex.txt &
    cp ./related-works/hsplit/HSplit/HSplit1_full ./datasets/hsplit/raw/simple-1.txt &
    cp ./related-works/hsplit/HSplit/HSplit2_full ./datasets/hsplit/raw/simple-2.txt &
    cp ./related-works/hsplit/HSplit/HSplit3_full ./datasets/hsplit/raw/simple-3.txt &
    cp ./related-works/hsplit/HSplit/HSplit4_full ./datasets/hsplit/raw/simple-4.txt &
    wait

    perl ./src/detokenizer.perl -l en -penn <./datasets/hsplit/raw/complex.txt >./datasets/hsplit/detokenized/complex.txt &
    perl ./src/detokenizer.perl -l en -penn <./datasets/hsplit/raw/simple-1.txt >./datasets/hsplit/detokenized/simple-1.txt &
    perl ./src/detokenizer.perl -l en -penn <./datasets/hsplit/raw/simple-2.txt >./datasets/hsplit/detokenized/simple-2.txt &
    perl ./src/detokenizer.perl -l en -penn <./datasets/hsplit/raw/simple-3.txt >./datasets/hsplit/detokenized/simple-3.txt &
    perl ./src/detokenizer.perl -l en -penn <./datasets/hsplit/raw/simple-4.txt >./datasets/hsplit/detokenized/simple-4.txt &
    wait

    python src/datasets/hsplit.py
}

# cont-bm, wiki-bm
small-but-mighty() {
    mkdir -p ./datasets/small-but-mighty/raw

    cp related-works/small-but-mighty/split-and-rephrase-data/benchmarks/contract-benchmark.tsv ./datasets/small-but-mighty/raw/cont-bm.tsv
    cp related-works/small-but-mighty/split-and-rephrase-data/benchmarks/wiki-benchmark.tsv ./datasets/small-but-mighty/raw/wiki-bm.tsv

    python src/datasets/small_but_mighty.py
}

for func in wiki-split min-wiki-split bisect hsplit small-but-mighty; do
    $func &
done

wait

echo "Done preprocessing!"

for dataset_name in wiki-split min-wiki-split bisect; do
    python src/datasets/nli_flitering.py \
        --dataset_name $dataset_name \
        --processor_name roberta \
        --device "cuda:0" &
    python src/datasets/nli_flitering.py \
        --dataset_name $dataset_name \
        --processor_name deberta \
        --device "cuda:1" &
    python src/datasets/nli_flitering.py \
        --dataset_name $dataset_name \
        --processor_name true \
        --batch_size 16 \
        --device "cuda:2" &
    wait
done

python src/datasets/entailment_intersection.py --dataset_dir datasets/wiki-split
python src/datasets/entailment_intersection.py --dataset_dir datasets/min-wiki-split
python src/datasets/entailment_intersection.py --dataset_dir datasets/bisect
