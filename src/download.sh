mkdir -p ./related-works
cd ./related-works

# moses-detokenizer
wget -P ./src https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/detokenizer.perl

# WebSplit: Split and Rephrase [Narayan+'17] (https://github.com/shashiongithub/Split-and-Rephrase)
web-split() {
    git clone https://github.com/shashiongithub/Split-and-Rephrase web-split
    cd web-split
    tar xvf benchmark-v0.1.tar.gz &
    tar xvf benchmark-v1.0.tar.gz &
    wait
}

# download dataset files from here
# https://drive.google.com/drive/folders/1KagOaUv1hlAK8ONYiyH6rXJHjGANk8Qj

## Split and Rephrase: Better Evaluation and Stronger Baselines [Aharoni+'18] (https://aclanthology.org/P18-2114/)
better-web-split() {
    git clone git@github.com:roeeaharoni/sprp-acl2018.git better-web-split
    cd better-web-split
    cd data
    unzip baseline-seq2seq-split-RDFs-relations.zip &
    unzip baseline-seq2seq.zip &
    # unzip complex-sents.zip &
    wait
    rm ./baseline-seq2seq-split-RDFs-relations.zip
    rm ./baseline-seq2seq.zip
    rm ./complex-sents.zip
}

# WikiSplit
# Learning To Split and Rephrase From Wikipedia Edit History [Botha+'18] (https://github.com/google-research-datasets/wiki-split)
wiki-split() {
    git clone https://github.com/google-research-datasets/wiki-split
    cd wiki-split
    unzip ./train.tsv.zip
    rm ./train.tsv.zip
}

# MinWikiSplit
# A Sentence Splitting Corpus with Minimal Propositions [Niklaus+19] (https://github.com/Lambda-3/MinWikiSplit)
min-wiki-split() {
    git clone https://github.com/Lambda-3/MinWikiSplit min-wiki-split
}

# Wiki-BM, Cont-BM
# Small but Mighty: New Benchmarks for Split and Rephrase [Zhang+20] (https://github.com/System-T/TextSimplification)
small-but-mighty() {
    git clone https://github.com/System-T/TextSimplification small-but-mighty
    cd small-but-mighty
    wget https://dax-cdn.cdn.appdomain.cloud/dax-split-and-rephrase/1.0.0/split-and-rephrase-data.tar.gz
    tar xvf split-and-rephrase-data.tar.gz
    rm ./split-and-rephrase-data.tar.gz
}

# HSplit
# BLEU is Not Suitable for the Evaluation of Text Simplification [Sulem+'18] (https://github.com/eliorsulem/HSplit-corpus)
hsplit() {
    git clone https://github.com/eliorsulem/HSplit-corpus hsplit
}

# DiscoFuse: A Large-Scale Dataset for Discourse-Based Sentence Fusion [Geva+'19] (https://github.com/google-research-datasets/discofuse)
discofuse() {
    git clone https://github.com/google-research-datasets/discofuse
    cd discofuse
    wget https://storage.googleapis.com/gresearch/discofuse/discofuse_v1_sports.zip
    wget https://storage.googleapis.com/gresearch/discofuse/discofuse_v1_wikipedia.zip
    unzip ./discofuse_v1_sports.zip &
    unzip ./discofuse_v1_wikipedia.zip &
    wait
    rm ./discofuse_v1_sports.zip
    rm ./discofuse_v1_wikipedia.zip
}

# Semantically Driven Sentence Fusion: Modeling and Evaluation [Ben-David+'20] (https://github.com/eyalbd2/Semantically-Driven-Sentence-Fusion)
aug-discofuse() {
    git clone git@github.com:eyalbd2/Semantically-Driven-Sentence-Fusion.git aug-discofuse
}

# Optimizing Statistical Machine Translation for Text Simplification (https://aclanthology.org/Q16-1029.pdf)
cocoxu-simplification() {
    git clone git@github.com:cocoxu/simplification.git cocoxu-simplification
}

# BISECT: Learning to Split and Rephrase Sentences with Bitexts (https://arxiv.org/pdf/2109.05006.pdf)
bisect() {
    git clone git@github.com:mounicam/BiSECT.git bisect
    gunzip bisect/bisect/train.src.gz &
    gunzip bisect/bisect/train.dst.gz &
    gunzip bisect/bisect/valid.src.gz &
    gunzip bisect/bisect/valid.dst.gz &
    gunzip bisect/bisect/test.src.gz &
    gunzip bisect/bisect/test.dst.gz &
    wait
    rm bisect/bisect/train.src.gz &
    rm bisect/bisect/train.dst.gz &
    rm bisect/bisect/valid.src.gz &
    rm bisect/bisect/valid.dst.gz &
    rm bisect/bisect/test.src.gz &
    rm bisect/bisect/test.dst.gz &
    wait
}

dissim() {
    git clone git@github.com:Lambda-3/DiscourseSimplification.git dissim
}

bleurt() {
    mkdir -p ./data
    wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
    unzip ./BLEURT-20.zip
    rm ./BLEURT-20.zip
    mv ./BLEURT-20 ./data/bleurt
}

# for func in web-split better-web-split ; do
for func in web-split better-web-split wiki-split min-wiki-split small-but-mighty hsplit discofuse aug-discofuse cocoxu-simplification bisect dissim bleurt; do
    $func &
done

wait

poetry run python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
