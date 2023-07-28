# Code repository for the article: "Classification of entire genomes of Neisseria meningitidis with a bag-of-words approach and machine learning"

## Minimal system requirements

- Python >=3.10
- Conda package manager installed on the machine.

## Setup

Clone the repository

    git clone git@github.com:mbodini/genome-predictor.git
    cd genome-predictor

Create a virtual environment and activate it (make sure the conda package manager is installed first):

    conda create --name genpred python=3.10 && conda activate genpred

Install the package in editable mode:

    pip install -e .

Restart the virtual environment to activate the console scripts:

    conda deactivate && conda activate genpred

## Steps to reproduce

All the results presented in the paper are precomputed in the `experiments` folder, and most of them can be reproduced by examining the notebooks in the `notebooks` folder. If, however, you wish to reproduce exactly all the pipeline, please follow the steps below in order.

If you want to run the notebooks with the pre-computed results, please take into account that `01_preprocessing.ipynb` and `05_sp_vocab_analysis.ipynb` will not work since the genomes are not included in this repo, as well as the tokenizers, for space reasons. You can however inspect the end results.

### 1. Download genomes

The following script downloads the genomes from PubMLST. Run:

```console
$ genpred-dl
```

The script will download all the genomes in the `data/raw/fasta/` folder, the gff3 files in `data/raw/gff3` folder, and create a file `data/genomes.csv` to attach the genome metadata to the correct paths.

> ⚠ **Warning:**<br>
> this could take a long time

### 2. Reproduce preprocessing

The following script performs data cleaning, re-balancing to eliminate sequencing biases and splitting. Run:

```console
$ genpred-prep [dataset.name=disease]
```

where `dataset.name` is one of `capsule` or `disease` (see Section Methods of the paper to understand how preprocessing was done).

The script will create a file `data/<dataset.name>/genomes.csv` containing the path of the genomes, their label and the partition (training, validation, test) to which they are assigned.

Works faster on multi-core machines.

### 3. Reproduce tokenization

The following script trains the **sentencepiece tokenizer**. Run:

```console
$ genpred-tok [dataset.name=disease] [dataset.vocab_size=32000]
```

where `dataset.name` is one of `capsule, disease`, and `dataset.vocab_size` is one of `2000, 4000, 8000, 16000, 32000, 64000`.

Once finished, the script will save a file `data/<dataset.name>/sentencepiece/<dataset.vocab_size>/tokenizer.json` containing the metadata to initialize the sentencepiece tokenizer.

> ⚠ **Warning:**<br>
> this could take a long time

### 4. Reproduce vectorization

The following script tokenizes the genomes and transforms them into TF-IDF matrices. Run:

```console
$ genpred-vect [vectorizer=sentencepiece] [dataset.name=disease] [dataset.vocab_size=32000]
```

where `vectorizer` is one of `kmer, sentencepiece`, `dataset.name` is one of `capsule, disease`, and `dataset.vocab_size` is one of `2000, 4000, 8000, 16000, 32000, 64000` if `vectorizer` is `sentencepiece`, or one of `64, 320, 1344, 5440, 21824, 87360` if `vectorizer` is `kmer`.

The script will save the following files in the `data/<dataset.name>/<vectorizer>/<dataset.vocab_size>` folder:

- `training.gz`, the training set matrix (in gzip format);
- `validation.gz`, the validation set matrix (in gzip format);
- `test.gz`, the test set matrix (in gzip format),
- `vectorizer.pkl`, the trained vectorizer.

> ⚠ **Warning:**<br>
> this could take a long time

### 5. Reproduce training

The following script trains the classifier. Run:

```console
$ genpred-train [dataset.name=sentencepiece] [dataset.vectorizer=sentencepiece] [dataset.vocab_size=32000]
```

where `dataset.name`, `dataset.vectorizer`, and `dataset.vocab_size` are defined as specified above. The script will save the following files in the `experiments/<dataset.name>/<dataset.vectorizer>/<dataset.vocab_size>`:

- `.hydra`: a folder with Hydra-specific configuration;
- `config.log`: a tree representation of the experiment config;
- `cv_results.csv`: a file containing the results of the model selection;
- `exec_time.log`: the time eplased by the script;
- `metrics.csv`: the metrics calculated on the test set;
- `model.pkl`: the trained model;
- `predictions.csv`: the predictions on the test set;
- `train.log`: a log file.

> ⚠ **Warning:**<br>
> this could take a long time

### 6. Reproduce capsule Knock-out

The following script will perform capsule knock-out. Run:

```console
$ genpred-ko-capsule dataset.name=capsule dataset.vectorizer=sentencepiece dataset.vocab_size=32000 control=true
```

for the control experiments, and

```console
$ genpred-ko-capsule dataset.name=capsule dataset.vectorizer=sentencepiece dataset.vocab_size=32000 control=false
```

for the knock-out experiments.

The scripts will save the following files in the `experiments/knockout/capsule/sentencepiece/32000/` folder:

- `capsule-control/predictions.csv`: a file with the control knock-out predictions;
- `capsule/predictions.csv`: a file with the capsule knock-out predictions.

Works faster on multi-core machines.

### 7. Reproduce irulence factors knockout

The following script will perform capsule knock-out. Run:

```console
$ genpred-ko-vf dataset.name=disease dataset.vectorizer=sentencepiece dataset.vocab_size=32000 control=true
```

for the control experiments, and

```console
$ genpred-ko-vf dataset.name=disease dataset.vectorizer=sentencepiece dataset.vocab_size=32000 control=false
```

for the knock-out experiments.

The script will save the following files in the `experiments/knockout/disease/sentencepiece/32000` folder:

- `virulence-control/predictions.csv`: a file with the control knock-out predictions;
- `virulence/predictions.csv`: a file with the VF knock-out predictions.

Works faster on multi-core machines.

### 8. Reproduce single gene knock-out

The following script will perform gene knock-out. Run:

```console
$ genpred-ko-gene dataset.name=disease dataset.vectorizer=sentencepiece dataset.vocab_size=32000 gene=<GENE_ID>
```

for the knock-out experiments, where `<GENE_ID>` is a valid gene identifier (see `data/genes/all_genes.txt` to see all possible choices).

The script will save the following files in the `experiments/knockout/disease/sentencepiece/32000/` folder:

- `<GENE_ID>/predictions.csv`: a file with the gene knock-out predictions.

Works faster on multi-core machines.


### 9. Notebooks

We also include 5 notebooks which were used to process the data and analyze the results. All are located in the `notebooks` folder.

### 10. Acknowledgments

This software was developed in collaboration with the University of Pisa.

### 11. Copyright

Copyright 2023 GlaxoSmithKline Biologicals SA. All rights reserved.
